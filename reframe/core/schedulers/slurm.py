import itertools
import re
import time

import reframe.core.schedulers as sched
import reframe.utility.os as os_ext

from datetime import datetime
from reframe.core.exceptions import (SpawnedProcessError,
                                     JobBlockedError, JobError)
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

    def _run_command(self, cmd, timeout=None):
        """Run command cmd and re-raise any exception as a JobError."""
        try:
            return os_ext.run_command(cmd, check=True, timeout=timeout)
        except SpawnedProcessError as e:
            raise JobError(jobid=self._jobid) from e

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

        prefix_patt = re.compile(r'(#\w+)')
        for opt in self.options:
            if not prefix_patt.match(opt):
                # FIXME: Temporary solution to issue #526. Check if a partition
                #        option is given in site settings which would overwrite
                #        the corresponding command line option.
                if (opt.startswith(('-p', '--partition')) and
                    self.sched_partition):
                    continue
                builder.verbatim('%s %s' % (self._prefix, opt))
            else:
                builder.verbatim(opt)

        super().emit_preamble(builder)

    def prepare(self, builder):
        if self.num_tasks == 0:
            if self.sched_reservation:
                nodes = self._get_reservation_nodes()
                num_nodes = self._count_compatible_nodes(nodes)
                getlogger().debug(
                    'found %s available node(s) in reservation %s' %
                    (num_nodes, self.sched_reservation))
                if num_nodes == 0:
                    raise JobError("could not find any node satisfying the "
                                   "required criteria in reservation '%s'" %
                                   self.sched_reservation)
                num_tasks_per_node = self.num_tasks_per_node or 1
                self._num_tasks = num_nodes * num_tasks_per_node
                getlogger().debug('automatically setting num_tasks to %s' %
                                  self.num_tasks)
            else:
                raise JobError('A reservation has to be specified '
                               'when setting the num_tasks to 0.')

        super().prepare(builder)

    def submit(self):
        cmd = 'sbatch %s' % self.script_filename
        completed = self._run_command(cmd, settings.job_submit_timeout)
        jobid_match = re.search('Submitted batch job (?P<jobid>\d+)',
                                completed.stdout)
        if not jobid_match:
            raise JobError(
                'could not retrieve the job id of the submitted job',
                jobid=None)

        self._jobid = int(jobid_match.group('jobid'))

    def _count_compatible_nodes(self, nodes):
        constraints = set()
        partitions = (set(self.sched_partition.split())
                      if self.sched_partition else set())

        if self.options:
            for optstr in self.options:
                optstr = optstr.strip()
                if optstr.startswith('--'):
                    optstr = optstr.replace('=', ' ', 1)

                option, arg = optstr.split(maxsplit=1)
                if option == '-C' or option == '--constraint':
                    constraints.update(arg.split())

                if option == '-p' or option == '--partition':
                    partitions.update(arg.split())

        num_nodes = 0
        for n in nodes:
            if n.active_features >= constraints and n.partitions >= partitions:
                num_nodes += 1

        return num_nodes

    def _get_reservation_nodes(self):
        command = 'scontrol show res %s' % self.sched_reservation
        completed = os_ext.run_command(command, check=True)
        node_match = reservation_nodes = re.search('(Nodes=\S+)',
                                                   completed.stdout)
        if node_match:
            reservation_nodes = node_match[1]
        else:
            raise JobError("could not extract the nodes names for "
                           "reservation '%s'" % self.sched_reservation)
        completed = os_ext.run_command(
            'scontrol show -o -a %s' % reservation_nodes, check=True)
        node_descriptions = completed.stdout.splitlines()
        return (SlurmNode(descr) for descr in node_descriptions)

    def _update_state(self):
        """Check the status of the job."""

        completed = self._run_command(
            'sacct -S %s -P -j %s -o jobid,state,exitcode' %
            (datetime.now().strftime('%F'), self._jobid)
        )
        state_match = re.search(r'^(?P<jobid>\d+)\|(?P<state>\S+)([^\|]*)\|'
                                r'(?P<exitcode>\d+)\:(?P<signal>\d+)',
                                completed.stdout, re.MULTILINE)
        if state_match is None:
            getlogger().debug('job state not matched (stdout follows)\n%s' %
                              completed.stdout)
            return

        self._state = SlurmJobState(state_match.group('state'))
        self._cancel_if_blocked()
        if self._state in self._completion_states:
            self._exitcode = int(state_match.group('exitcode'))

    def _cancel_if_blocked(self):
        if self._is_cancelling or self._state not in self._pending_states:
            return

        completed = self._run_command('squeue -h -j %s -o %%r' % self._jobid)
        if not completed.stdout:
            # Can't retrieve job's state. Perhaps it has finished already and
            # does not show up in the output of squeue
            return

        self._check_and_cancel(completed.stdout)

    def _check_and_cancel(self, reason_descr):
        """Check if blocking reason ``reason_descr`` is unrecoverable and cancel the
        job in this case."""

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

            raise JobBlockedError(reason_msg, jobid=self._jobid)

    def wait(self):
        if self._jobid is None:
            raise JobError('cannot wait a non spawned job', jobid=None)

        # Quickly return in case we have finished already
        if self._state in self._completion_states:
            return

        intervals = itertools.cycle(settings.job_poll_intervals)
        self._update_state()
        while self._state not in self._completion_states:
            time.sleep(next(intervals))
            self._update_state()

    def cancel(self):
        if self._jobid is None:
            raise JobError('cannot cancel a non spawned job', jobid=None)

        getlogger().debug('cancelling job (id=%s)' % self._jobid)
        self._run_command('scancel %s' % self._jobid,
                          settings.job_submit_timeout)
        self._is_cancelling = True

    def finished(self):
        try:
            self._update_state()
        except JobBlockedError:
            # Job blocked forever; reraise the exception to notify our caller
            raise
        except JobError as e:
            # We ignore these exceptions at this point and we simply mark the
            # job as unfinished.
            getlogger().debug('ignoring error during polling: %s' % e)
            return False
        else:
            return self._state in self._completion_states


@register_scheduler('squeue')
class SqueueJob(SlurmJob):
    """A Slurm job that uses squeue to query its state."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.submit_time = None
        self.squeue_delay = 2
        self._cancelled = False

    def submit(self):
        super().submit()
        self.submit_time = datetime.now()

    def _update_state(self):
        time_from_submit = datetime.now() - self.submit_time
        rem_wait = self.squeue_delay - time_from_submit.total_seconds()
        if rem_wait > 0:
            time.sleep(rem_wait)

        # We don't run the command with check=True, because if the job has
        # finished already, squeue might return an error about an invalid
        # job id.
        completed = self._run_command(
            'squeue -h -j %s -O state,exit_code,reason' % self._jobid)
        output = completed.stdout.strip()
        if not output:
            # Assume that job has finished
            self._state = (SLURM_JOB_CANCELLED if self._cancelled
                           else SLURM_JOB_COMPLETED)

            # Set exit code manually, if not set already by the polling
            if self._exitcode is None:
                self._exitcode = 0

            return

        # There is no reliable way to get the exit code, so we always capture
        # it, just in case we are lucky enough and get its actual value while
        # the job has finished but is still showing up in the queue (e.g., when
        # it is 'COMPLETING')
        state, exitcode, reason = output.split(maxsplit=2)
        self._state = SlurmJobState(state)
        self._exitcode = int(exitcode)
        if not self._is_cancelling and self._state in self._pending_states:
            self._check_and_cancel(reason)

    def cancel(self):
        # There is no reliable way to get the state of the job after it has
        # finished, so we explicitly mark it as cancelled here. The
        # _update_state() will make sure to return the approriate state.
        super().cancel()
        self._cancelled = True


class SlurmNode:
    """Class representing a Slurm node."""

    def __init__(self, node_descr):
        self._name = self._extract_attribute('NodeName', node_descr)
        self._partitions = set(self._extract_attribute(
            'Partitions', node_descr).split(','))
        self._active_features = set(self._extract_attribute(
            'ActiveFeatures', node_descr).split(','))

    @property
    def active_features(self):
        return self._active_features

    @property
    def name(self):
        return self._name

    @property
    def partitions(self):
        return self._partitions

    def _extract_attribute(self, attr_name, node_descr):
        attr_match = re.search(r'%s=(\S+)' % attr_name, node_descr)
        if attr_match:
            return attr_match.group(1)
        else:
            raise JobError("could not extract attribute '%s' from "
                           "node description" % attr_name)
