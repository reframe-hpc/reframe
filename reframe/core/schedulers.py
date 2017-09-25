#
# Scheduler implementations
#

import itertools
import os
import re
import signal
import stat
import subprocess
import time

import reframe.core.debug as debug
import reframe.utility.os as os_ext

from datetime import datetime
from reframe.core.exceptions import (ReframeError,
                                     JobSubmissionError,
                                     JobResourcesError)
from reframe.core.launchers import LocalLauncher
from reframe.settings import settings


class _TimeoutExpired(ReframeError):
    pass


class Job:
    def __init__(self,
                 job_name,
                 job_environ_list,
                 job_script_builder,
                 launcher,
                 num_tasks,
                 script_filename=None,
                 stdout=None,
                 stderr=None,
                 options=[],
                 launcher_options=[],
                 **kwargs):
        self.name = job_name
        self.environs = job_environ_list or []
        self.script_builder = job_script_builder
        self.num_tasks = num_tasks
        self.script_filename = script_filename or '%s.sh' % self.name
        self.options = options
        self.launcher = launcher(self, launcher_options)
        self.stdout = stdout or '%s.out' % self.name
        self.stderr = stderr or '%s.err' % self.name

        # Commands to be run before and after the job is launched
        self.pre_run  = []
        self.post_run = []

        # Live job information; to be filled during job's lifetime
        self.jobid = -1
        self.state = None
        self.exitcode = None
        self._is_cancelling = False

    def __repr__(self):
        return debug.repr(self)

    def emit_preamble(self, builder):
        for e in self.environs:
            e.emit_load_instructions(self.script_builder)

        for stmt in self.pre_run:
            builder.verbatim(stmt)

    def emit_postamble(self, builder):
        for stmt in self.post_run:
            builder.verbatim(stmt)

    def _submit(self, script):
        raise NotImplementedError('Attempt to call an abstract method')

    def wait(self):
        """Wait for the job to finish."""
        raise NotImplementedError('Attempt to call an abstract method')

    def finished(self):
        """Status of the job."""
        raise NotImplementedError('Attempt to call an abstract method')

    def cancel(self):
        """Cancel this job."""
        raise NotImplementedError('Attempt to call an abstract method')

    def submit(self, cmd, workdir='.'):
        # Build the submission script and submit it
        self.emit_preamble(self.script_builder)
        self.script_builder.verbatim('cd %s' % workdir)
        self.launcher.emit_run_command(cmd, self.script_builder)
        self.emit_postamble(self.script_builder)

        script_file = open(self.script_filename, 'w+')
        script_file.write(self.script_builder.finalise())
        script_file.close()
        self._submit(script_file)


class JobState:
    def __init__(self, state):
        self.state = state

    def __repr__(self):
        return debug.repr(self)

    def __eq__(self, other):
        return other is not None and self.state == other.state

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return self.state


class JobResources:
    """Managed job resources.

    Custom resources usually configured per system by the system
    administrators."""

    def __init__(self, resources):
        self.resources = resources

    def __repr__(self):
        return debug.repr(self)

    def get(self, name, **kwargs):
        """Get resource option string for the resource `name'"""
        try:
            return self.resources.format(**kwargs)
        except KeyError:
            return None

    def getall(self, resources_spec):
        """Get all resource option strings for resources in `resource_spec`."""
        ret = []
        for opt, kwargs in resources_spec.items():
            opt_str = self.get(opt, **kwargs)
            if opt_str:
                ret.append(opt_str)

        return ret


# Local job states
class LocalJobState(JobState):
    def __init__(self, state):
        super().__init__(state)


LOCAL_JOB_SUCCESS = LocalJobState('SUCCESS')
LOCAL_JOB_FAILURE = LocalJobState('FAILURE')
LOCAL_JOB_TIMEOUT = LocalJobState('TIMEOUT')


class LocalJob(Job):
    def __init__(self,
                 time_limit=(0, 10, 0),
                 **kwargs):
        super().__init__(num_tasks=1,
                         launcher=LocalLauncher,
                         **kwargs)
        # Launched process
        self.time_limit = time_limit
        self.cancel_grace_period = 2
        self._wait_poll_secs = 0.1
        self._proc = None

    def _submit(self, script):
        # `chmod +x' first, because we will execute the script locally
        os.chmod(script.name, os.stat(script.name).st_mode | stat.S_IEXEC)

        # Run from the absolute path
        self._f_stdout = open(self.stdout, 'w+')
        self._f_stderr = open(self.stderr, 'w+')

        # The new process starts also a new session (session leader), so that
        # we can later kill any other processes that this might spawn by just
        # killing this one.
        self._proc = os_ext.run_command_async(os.path.abspath(script.name),
                                              stdout=self._f_stdout,
                                              stderr=self._f_stderr,
                                              start_new_session=True)
        # Update job info
        self.jobid = self._proc.pid

    def _kill_all(self):
        """Send SIGKILL to all the processes of the spawned job."""
        try:
            os.killpg(self.jobid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            # The process group may already be dead or assigned to a different
            # group, so ignore this error
            pass

    def _term_all(self):
        """Send SIGTERM to all the processes of the spawned job."""
        os.killpg(self.jobid, signal.SIGTERM)

    def _wait_all(self, timeout=0):
        """Wait for all the processes of spawned job to finish.

        Keyword arguments:

        timeout -- Timeout period for this wait call in seconds (may be a real
                   number, too). If `None` or `0`, no timeout will be set.
        """
        t_wait = datetime.now()
        self._proc.wait(timeout=timeout or None)
        t_wait = datetime.now() - t_wait
        try:
            # Wait for all processes in the process group to finish
            while not timeout or t_wait.total_seconds() < timeout:
                t_poll = datetime.now()
                os.killpg(self.jobid, 0)
                time.sleep(self._wait_poll_secs)
                t_poll = datetime.now() - t_poll
                t_wait += t_poll

            # Final check
            os.killpg(self.jobid, 0)
            raise _TimeoutExpired
        except (ProcessLookupError, PermissionError):
            # Ignore also EPERM errors in case this process id is assigned
            # elsewhere and we cannot query its status
            return

    def cancel(self):
        """Cancel the current job.

        The SIGTERM signal will be sent first to all the processes of this job
        and after a grace period (default 2s) the SIGKILL signal will be send.

        This function waits for the spawned process tree to finish.
        """
        if self.jobid == -1:
            return

        self._term_all()

        # Set the time limit to the grace period and let wait() do the final
        # killing
        self.time_limit = (0, 0, self.cancel_grace_period)
        self.wait()

    def wait(self, timeout=None):
        """Wait for the spawned job to finish.

        As soon as the parent job process finishes, all of its spawned
        subprocesses will be forced to finish, too.

        Upon return, the whole process tree of the spawned job process will be
        cleared, unless any of them has called `setsid()`.

        Keyword arguments:

        timeout -- Timeout period for this wait call in seconds. If `None` the
                   `self.time_limit` will be used.
        """
        if self.state is not None:
            # Job has been already waited for
            return

        if timeout is None:
            # Convert time_limit to seconds
            h, m, s = self.time_limit
            timeout = h * 3600 + m * 60 + s

        try:
            self._wait_all(timeout=timeout)
            self.exitcode = self._proc.returncode
            if self.exitcode != 0:
                self.state = LOCAL_JOB_FAILURE
            else:
                self.state = LOCAL_JOB_SUCCESS
        except (_TimeoutExpired, subprocess.TimeoutExpired):
            self.state = LOCAL_JOB_TIMEOUT
        finally:
            # Cleanup all the processes of this job
            self._kill_all()
            self._wait_all()
            self._f_stdout.close()
            self._f_stderr.close()

    def finished(self):
        """Check if the spawned process has finished.

        This function does not wait the process. It just queries its state. If
        the process has finished, you *must* call wait() to properly cleanup
        after it.
        """
        self._proc.poll()
        if self._proc.returncode is None:
            return False

        return True


class SlurmJobState(JobState):
    def __init__(self, state):
        super().__init__(state)


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


class SlurmJob(Job):
    def __init__(self,
                 time_limit=(0, 10, 0),
                 use_smt=None,
                 nodelist=None,
                 exclude=None,
                 partition=None,
                 reservation=None,
                 account=None,
                 num_tasks_per_node=None,
                 num_cpus_per_task=None,
                 num_tasks_per_core=None,
                 num_tasks_per_socket=None,
                 exclusive_access=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.partition   = partition
        self.time_limit  = time_limit
        self.use_smt     = use_smt
        self.exclusive_access   = exclusive_access
        self.nodelist    = nodelist
        self.exclude     = exclude
        self.reservation = reservation
        self.account     = account
        self.prefix      = '#SBATCH'
        self.signal      = None
        self.job_init_poll_num_tries = 0

        self.num_tasks_per_node = num_tasks_per_node
        self.num_cpus_per_task = num_cpus_per_task
        self.num_tasks_per_core = num_tasks_per_core
        self.num_tasks_per_socket = num_tasks_per_socket
        self.completion_states = [SLURM_JOB_BOOT_FAIL,
                                  SLURM_JOB_CANCELLED,
                                  SLURM_JOB_COMPLETED,
                                  SLURM_JOB_FAILED,
                                  SLURM_JOB_NODE_FAILED,
                                  SLURM_JOB_PREEMPTED,
                                  SLURM_JOB_TIMEOUT]
        self.pending_states = [SLURM_JOB_CONFIGURING,
                               SLURM_JOB_PENDING]
        # Reasons to cancel a pending job: if the job is expected to remain
        # pending for a much longer time then usual (mostly if a sysadmin
        # intervention is required)
        self.cancel_reasons = ['FrontEndDown',
                               'Licenses',         # May require sysadmin
                               'NodeDown',
                               'PartitionDown',
                               'PartitionInactive',
                               'PartitionNodeLimit',
                               'QOSJobLimit',
                               'QOSResourceLimit',
                               'ReqNodeNotAvail',  # Inaccurate SLURM doc
                               'QOSUsageThreshold']

    def emit_preamble(self, builder):
        builder.verbatim('%s --job-name="%s"' % (self.prefix, self.name))
        builder.verbatim('%s --time=%s' %
                         (self.prefix, '%d:%d:%d' % self.time_limit))
        builder.verbatim('%s --ntasks=%d' % (self.prefix, self.num_tasks))
        if self.num_tasks_per_node:
            builder.verbatim('%s --ntasks-per-node=%d' %
                             (self.prefix, self.num_tasks_per_node))

        if self.num_cpus_per_task:
            builder.verbatim('%s --cpus-per-task=%d' %
                             (self.prefix, self.num_cpus_per_task))

        if self.num_tasks_per_core:
            builder.verbatim('%s --ntasks-per-core=%d' %
                             (self.prefix, self.num_tasks_per_core))

        if self.num_tasks_per_socket:
            builder.verbatim('%s --ntasks-per-socket=%d' %
                             (self.prefix, self.num_tasks_per_socket))

        if self.partition:
            builder.verbatim('%s --partition=%s' %
                             (self.prefix, self.partition))

        if self.exclusive_access:
            builder.verbatim('%s --exclusive' % self.prefix)

        if self.account:
            builder.verbatim(
                '%s --account=%s' % (self.prefix, self.account))

        if self.nodelist:
            builder.verbatim(
                '%s --nodelist=%s' % (self.prefix, self.nodelist))

        if self.exclude:
            builder.verbatim(
                '%s --exclude=%s' % (self.prefix, self.exclude))

        if self.use_smt is not None:
            hint = 'multithread' if self.use_smt else 'nomultithread'
            builder.verbatim('%s --hint=%s' % (self.prefix, hint))

        if self.reservation:
            builder.verbatim('%s --reservation=%s' % (self.prefix,
                                                      self.reservation))
        if self.stdout:
            builder.verbatim('%s --output="%s"' % (self.prefix, self.stdout))

        if self.stderr:
            builder.verbatim('%s --error="%s"' % (self.prefix, self.stderr))

        for opt in self.options:
            builder.verbatim('%s %s' % (self.prefix, opt))

        super().emit_preamble(builder)

    def _submit(self, script):
        cmd = 'sbatch %s' % script.name
        completed = os_ext.run_command(
            cmd, check=True, timeout=settings.job_submit_timeout)

        jobid_match = re.search('Submitted batch job (?P<jobid>\d+)',
                                completed.stdout)
        if not jobid_match:
            raise JobSubmissionError(command=cmd,
                                     stdout=completed.stdout,
                                     stderr=completed.stderr,
                                     exitcode=completed.returncode)

        # Job id's are treated as string; keep in mind
        self.jobid = jobid_match.group('jobid')
        if not self.stdout:
            self.stdout = 'slurm-%s.out' % self.jobid

        if not self.stderr:
            self.stderr = self.stdout

    def _update_state(self):
        """Check the status of the job."""
        intervals = itertools.cycle(settings.job_init_poll_intervals)
        state_match = None
        max_tries = settings.job_init_poll_max_tries
        while (not state_match and
               self.job_init_poll_num_tries < max_tries):
            # Query job state persistently. When you first submit, the job may
            # not be yet registered in the database; so try some times We
            # restrict the `sacct' query to today (`-S' option), so as to avoid
            # possible older and stale slurm database entries.
            completed = os_ext.run_command(
                'sacct -S %s -P -j %s -o jobid,state,exitcode' %
                (datetime.now().strftime('%F'), self.jobid),
                check=True)
            state_match = re.search(
                '^(?P<jobid>\d+)\|(?P<state>\S+)([^\|]*)\|'
                '(?P<exitcode>\d+)\:(?P<signal>\d+)',
                completed.stdout, re.MULTILINE)
            if not state_match:
                self.job_init_poll_num_tries += 1
                time.sleep(next(intervals))

        if not state_match:
            raise ReframeError('Querying initial job state timed out')

        assert self.jobid == state_match.group('jobid')

        self.state    = SlurmJobState(state_match.group('state'))
        self.exitcode = int(state_match.group('exitcode'))
        self.signal   = int(state_match.group('signal'))

    def _cancel_if_blocked(self):
        if self._is_cancelling or self.state not in self.pending_states:
            return

        completed = os_ext.run_command('squeue -j %s -o "%%i|%%T|%%r" ' %
                                       self.jobid, check=True)
        # Note: the reason may given as "ReqNodeNotAvail,
        # UnavailableNodes:nid00[408,411-415]" by squeue. In this case,
        # we take only the string up to the comma.
        state_match = re.search(
            '^(?P<jobid>\d+)\|(?P<state>\S+)\|'
            '(?P<reason>\w+)(\W+(?P<reason_details>.*))?',
            completed.stdout, re.MULTILINE)
        # If squeue does not return any job info (state_match is empty),
        # it means normally that the job has finished meanwhile. So we
        # can exit this function.
        if not state_match:
            return

        assert self.jobid == state_match.group('jobid')
        # Assure that the job is still in a pending state
        state  = SlurmJobState(state_match.group('state'))
        reason = state_match.group('reason')
        if state in self.pending_states and reason in self.cancel_reasons:
            self.cancel()
            reason_msg = ('job canceled because it was blocked in pending '
                          'state due to the following SLURM reason: ' + reason)
            reason_details = state_match.group('reason_details')
            if reason_details:
                reason_msg += ', ' + reason_details
            raise JobResourcesError(reason_msg)

    def wait(self):
        intervals = itertools.cycle(settings.job_state_poll_intervals)

        # Quickly return in case we have finished already
        if self.state in self.completion_states:
            return

        self._update_state()
        self._cancel_if_blocked()
        while self.state not in self.completion_states:
            time.sleep(next(intervals))
            self._update_state()
            self._cancel_if_blocked()

    def cancel(self):
        """Cancel job execution.

        This call waits until the job has finished."""
        if self.jobid == -1:
            return

        os_ext.run_command('scancel %s' % self.jobid,
                           check=True, timeout=settings.job_submit_timeout)
        self._is_cancelling = True
        self.wait()

    def finished(self):
        self._update_state()
        return self.state in self.completion_states
