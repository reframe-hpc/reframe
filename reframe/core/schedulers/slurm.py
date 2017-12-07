import itertools
import re
import time

import reframe.core.schedulers as sched
import reframe.utility.os as os_ext

from datetime import datetime
from reframe.core.exceptions import JobSubmissionError, ReframeError
from reframe.core.logging import getlogger
from reframe.core.schedulers.registry import register_scheduler
from reframe.settings import settings


class SlurmJobState(sched.JobState):
    pass


# Slurm Job states
SLURM_JOB_BOOT_FAIL   = SlurmJobState('BOOT_FAIL')
SLURM_JOB_CANCELLED   = SlurmJobState('CANCELLED')
SLURM_JOB_COMPLETED   = SlurmJobState('COMPLETED')
SLURM_JOB_CONFIGURING = SlurmJobState('CONFIGURING')
SLURM_JOB_COMPLETING  = SlurmJobState('COMPLETING')
SLURM_JOB_FAILED      = SlurmJobState('FAILED')
SLURM_JOB_NODE_FAILED = SlurmJobState('NODE_FAILED')
SLURM_JOB_PENDING     = SlurmJobState('PENDING')
SLURM_JOB_PREEMPTED   = SlurmJobState('PREEMPTED')
SLURM_JOB_RESIZING    = SlurmJobState('RESIZING')
SLURM_JOB_RUNNING     = SlurmJobState('RUNNING')
SLURM_JOB_SUSPENDED   = SlurmJobState('SUSPENDED')
SLURM_JOB_TIMEOUT     = SlurmJobState('TIMEOUT')


@register_scheduler('slurm')
class SlurmJob(sched.Job):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prefix  = '#SBATCH'
        self._completion_states = [SLURM_JOB_BOOT_FAIL,
                                   SLURM_JOB_CANCELLED,
                                   SLURM_JOB_COMPLETED,
                                   SLURM_JOB_FAILED,
                                   SLURM_JOB_NODE_FAILED,
                                   SLURM_JOB_PREEMPTED,
                                   SLURM_JOB_TIMEOUT]
        self._pending_states = [SLURM_JOB_CONFIGURING,
                                SLURM_JOB_PENDING]

        # Reasons to cancel a pending job: if the job is expected to remain
        # pending for a much longer time then usual (mostly if a sysadmin
        # intervention is required)
        self._cancel_reasons = ['FrontEndDown',
                                'Licenses',         # May require sysadmin
                                'NodeDown',
                                'PartitionDown',
                                'PartitionInactive',
                                'PartitionNodeLimit',
                                'QOSJobLimit',
                                'QOSResourceLimit',
                                'ReqNodeNotAvail',  # Inaccurate SLURM doc
                                'QOSUsageThreshold']
        self._is_cancelling = False

    def _emit_job_option(self, var, option, builder):
        if var is not None:
            builder.verbatim(self._prefix + ' ' + option.format(var))

    def emit_preamble(self, builder):
        self._emit_job_option(self.name, '--job-name="{0}"', builder)
        self._emit_job_option('%d:%d:%d' % self.time_limit,
                              '--time={0}', builder)
        self._emit_job_option(self.num_tasks, '--ntasks={0}', builder)
        self._emit_job_option(self.num_tasks_per_node,
                              '--ntasks-per-node={0}', builder)
        self._emit_job_option(self.num_tasks_per_core,
                              '--ntasks-per-core={0}', builder)
        self._emit_job_option(self.num_tasks_per_socket,
                              '--ntasks-per-socket={0}', builder)
        self._emit_job_option(self.num_cpus_per_task,
                              '--cpus-per-task={0}', builder)
        self._emit_job_option(self.sched_partition, '--partition={0}', builder)
        self._emit_job_option(self.sched_exclusive_access,
                              '--exclusive', builder)
        self._emit_job_option(self.sched_account, '--account={0}', builder)
        self._emit_job_option(self.sched_nodelist, '--nodelist={0}', builder)
        self._emit_job_option(self.sched_exclude_nodelist,
                              '--exclude={0}', builder)
        if self.use_smt is None:
            hint = None
        else:
            hint = 'multithread' if self.use_smt else 'nomultithread'

        self._emit_job_option(hint, '--hint={0}', builder)
        self._emit_job_option(self.sched_reservation,
                              '--reservation={0}', builder)
        self._emit_job_option(self.stdout, '--output={0}', builder)
        self._emit_job_option(self.stderr, '--error={0}', builder)

        for opt in self.options:
            builder.verbatim('%s %s' % (self._prefix, opt))

        super().emit_preamble(builder)

    def submit(self):
        cmd = 'sbatch %s' % self.script_filename
        completed = os_ext.run_command(
            cmd, check=True, timeout=settings.job_submit_timeout)
        jobid_match = re.search('Submitted batch job (?P<jobid>\d+)',
                                completed.stdout)
        if not jobid_match:
            raise JobSubmissionError(command=cmd,
                                     stdout=completed.stdout,
                                     stderr=completed.stderr,
                                     exitcode=completed.returncode)
        self._jobid = int(jobid_match.group('jobid'))

    def _update_state(self):
        """Check the status of the job."""

        completed = os_ext.run_command(
            'sacct -S %s -P -j %s -o jobid,state,exitcode' %
            (datetime.now().strftime('%F'), self._jobid),
            check=True)
        state_match = re.search(r'^(?P<jobid>\d+)\|(?P<state>\S+)([^\|]*)\|'
                                r'(?P<exitcode>\d+)\:(?P<signal>\d+)',
                                completed.stdout, re.MULTILINE)
        if state_match is None:
            getlogger().debug('job state not matched (stdout follows)\n%s' %
                              completed.stdout)
            return

        self._state = SlurmJobState(state_match.group('state'))
        if self._state in self._completion_states:
            self._exitcode = int(state_match.group('exitcode'))

    def _cancel_if_blocked(self):
        if self._is_cancelling or self._state not in self._pending_states:
            return

        completed = os_ext.run_command('squeue -j %s -o %%r' % self._jobid,
                                       check=True)

        # Get the reason description by removing the header from the result
        try:
            reason_descr = completed.stdout.split('\n')[1]
        except IndexError:
            # Can't retrieve job's state. Perhaps it has finished already and
            # does not show up in the output of squeue
            return

        # The reason description may have two parts as follows:
        # "ReqNodeNotAvail, UnavailableNodes:nid00[408,411-415]"
        try:
            reason, reason_details = reason_descr.split(',', maxsplit=1)
        except ValueError:
            # no reason details
            reason, reason_details = reason_descr, None

        if reason in self._cancel_reasons:
            self.cancel()
            reason_msg = ('job cancelled because it was blocked due to '
                          'a perhaps non-recoverable reason: ' + reason)
            if reason_details is not None:
                reason_msg += ', ' + reason_details

            raise ReframeError(reason_msg)

    def wait(self):
        if self._jobid is None:
            raise ReframeError('no job is spawned yet')

        # Quickly return in case we have finished already
        if self._state in self._completion_states:
            return

        intervals = itertools.cycle(settings.job_poll_intervals)
        self._update_state()
        self._cancel_if_blocked()
        while self._state not in self._completion_states:
            time.sleep(next(intervals))
            self._update_state()
            self._cancel_if_blocked()

    def cancel(self):
        """Cancel job execution.

        This call waits until the job has finished."""
        getlogger().debug('cancelling job (id=%s)' % self._jobid)
        if self._jobid is None:
            raise ReframeError('no job is spawned yet')

        os_ext.run_command('scancel %s' % self._jobid,
                           check=True, timeout=settings.job_submit_timeout)
        self._is_cancelling = True
        self.wait()

    def finished(self):
        try:
            self._update_state()
        except ReframeError as e:
            # We ignore these exceptions at this point and we simply mark the
            # job as unfinished.
            getlogger().debug('ignoring error during polling: %s' % e)
            return False
        else:
            return self._state in self._completion_states
