#
# Scheduler implementations
#

import abc
import itertools
import os
import re
import numbers
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
from reframe.core.fields import TypedField, TypedListField
from reframe.core.launchers import LocalLauncher
from reframe.core.logging import getlogger
from reframe.settings import settings


class _TimeoutExpired(ReframeError):
    pass


class Job(abc.ABC):
    """A job descriptor.

    Users may not create jobs directly."""

    #: Options to be passed to the backend job scheduler.
    #:
    #: :type: :class:`list` of :class:`str`
    #: :default: ``[]``
    options = TypedListField('options', str)

    #: List of shell commands to execute before launching this job.
    #:
    #: These commands do not execute in the context of ReFrame.
    #: Instead, they are emitted in the generated job script just before the
    #: actual job launch command.
    #:
    #: :type: :class:`list` of :class:`str`
    #: :default: ``[]``
    pre_run  = TypedListField('pre_run', str)

    #: List of shell commands to execute after launching this job.
    #:
    #: See :attr:`pre_run` for a more detailed description of the semantics.
    #:
    #: :type: :class:`list` of :class:`str`
    #: :default: ``[]``
    post_run = TypedListField('post_run', str)

    # FIXME: This is not very meaningful, but allows a meaningful
    # documentation.
    #
    #: The job launcher used to launch this job.
    #:
    #: :type: :class:`reframe.core.launchers.JobLauncher`
    launcher = TypedField('launcher', object)

    def __init__(self,
                 job_name,
                 job_environ_list,
                 job_script_builder,
                 launcher_type,
                 num_tasks,
                 time_limit=(0, 10, 0),
                 script_filename=None,
                 stdout=None,
                 stderr=None,
                 sched_options=[],
                 launcher_options=[],
                 **kwargs):
        # Mutable fields
        self.options = list(sched_options)

        self.launcher = launcher_type(self, launcher_options)

        # Commands to be run before and after the job is launched
        self.pre_run  = []
        self.post_run = []

        self._name = job_name
        self._environs = list(job_environ_list) or []
        self._script_builder = job_script_builder
        self._num_tasks = num_tasks
        self._script_filename = script_filename or '%s.sh' % self._name
        self._stdout = stdout or '%s.out' % self._name
        self._stderr = stderr or '%s.err' % self._name
        self._time_limit = time_limit

        # Live job information; to be filled during job's lifetime
        self._jobid = -1
        self._state = None
        self._exitcode = None

    def __repr__(self):
        return debug.repr(self)

    # Read-only properties
    @property
    def exitcode(self):
        return self._exitcode

    @property
    def jobid(self):
        return self._jobid

    @property
    def name(self):
        return self._name

    @property
    def num_tasks(self):
        return self._num_tasks

    @property
    def script_filename(self):
        return self._script_filename

    @property
    def state(self):
        return self._state

    @property
    def stdout(self):
        return self._stdout

    @property
    def stderr(self):
        return self._stderr

    @property
    def time_limit(self):
        return self._time_limit

    def emit_preamble(self, builder):
        for e in self._environs:
            e.emit_load_instructions(self._script_builder)

        for stmt in self.pre_run:
            builder.verbatim(stmt)

    def emit_postamble(self, builder):
        for stmt in self.post_run:
            builder.verbatim(stmt)

    @abc.abstractmethod
    def _submit(self, script):
        """Submit a script file for execution.

        Keyword arguments:
        script -- the name of a script file to be submitted
        """

    # Wait for the job to finish.
    @abc.abstractmethod
    def wait(self):
        pass

    # Return `True` if job has finished.
    @abc.abstractmethod
    def finished(self):
        pass

    # Cancel the job.
    @abc.abstractmethod
    def cancel(self):
        pass

    def submit(self, cmd, workdir='.'):
        # Build the submission script and submit it
        getlogger().debug('emitting job script: %s' % self._script_filename)
        self.emit_preamble(self._script_builder)
        self._script_builder.verbatim('cd %s' % workdir)
        self.launcher.emit_run_command(cmd, self._script_builder)
        self.emit_postamble(self._script_builder)

        script_file = open(self._script_filename, 'w+')
        script_file.write(self._script_builder.finalise())
        script_file.close()
        self._submit(script_file)


class JobState:
    def __init__(self, state):
        self._state = state

    def __repr__(self):
        return debug.repr(self)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        return self._state == other._state

    def __str__(self):
        return self._state


class JobResources:
    """Managed job resources.

    Custom resources usually configured per system by the system
    administrators.
    """

    def __init__(self, resources):
        self._resources = resources

    def __repr__(self):
        return debug.repr(self)

    def get(self, name, **kwargs):
        """Get resource option string for the resource ``name``."""
        try:
            return self._resources.format(**kwargs)
        except KeyError:
            return None

    def getall(self, resources_spec):
        """Get all resource option strings for resources in ``resource_spec``."""
        ret = []
        for opt, kwargs in resources_spec.items():
            opt_str = self.get(opt, **kwargs)
            if opt_str:
                ret.append(opt_str)

        return ret


# Local job states
class LocalJobState(JobState):
    pass


LOCAL_JOB_SUCCESS = LocalJobState('SUCCESS')
LOCAL_JOB_FAILURE = LocalJobState('FAILURE')
LOCAL_JOB_TIMEOUT = LocalJobState('TIMEOUT')


class LocalJob(Job):
    def __init__(self,
                 time_limit=(0, 10, 0),
                 **kwargs):
        super().__init__(num_tasks=1,
                         launcher_type=LocalLauncher,
                         **kwargs)
        # Launched process
        self.cancel_grace_period = 2
        self._wait_poll_secs = 0.1
        self._proc = None

    def _submit(self, script):
        # `chmod +x' first, because we will execute the script locally
        os.chmod(script.name, os.stat(script.name).st_mode | stat.S_IEXEC)

        # Run from the absolute path
        self._f_stdout = open(self._stdout, 'w+')
        self._f_stderr = open(self._stderr, 'w+')

        # The new process starts also a new session (session leader), so that
        # we can later kill any other processes that this might spawn by just
        # killing this one.
        self._proc = os_ext.run_command_async(os.path.abspath(script.name),
                                              stdout=self._f_stdout,
                                              stderr=self._f_stderr,
                                              start_new_session=True)
        # Update job info
        self._jobid = self._proc.pid

    def _kill_all(self):
        """Send SIGKILL to all the processes of the spawned job."""
        try:
            os.killpg(self._jobid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            # The process group may already be dead or assigned to a different
            # group, so ignore this error
            getlogger().debug(
                'pid %s already dead or assigned elsewhere' % self._jobid)

    def _term_all(self):
        """Send SIGTERM to all the processes of the spawned job."""
        os.killpg(self._jobid, signal.SIGTERM)

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
                os.killpg(self._jobid, 0)
                time.sleep(self._wait_poll_secs)
                t_poll = datetime.now() - t_poll
                t_wait += t_poll

            # Final check
            os.killpg(self._jobid, 0)
            raise _TimeoutExpired
        except (ProcessLookupError, PermissionError):
            # Ignore also EPERM errors in case this process id is assigned
            # elsewhere and we cannot query its status
            getlogger().debug(
                'pid %s already dead or assigned elsewhere' % self._jobid)
            return

    def cancel(self):
        """Cancel the current job.

        The SIGTERM signal will be sent first to all the processes of this job
        and after a grace period (default 2s) the SIGKILL signal will be send.

        This function waits for the spawned process tree to finish.
        """
        if self._jobid == -1:
            return

        self._term_all()

        # Set the time limit to the grace period and let wait() do the final
        # killing
        self._time_limit = (0, 0, self.cancel_grace_period)
        self.wait()

    def wait(self, timeout=None):
        """Wait for the spawned job to finish.

        As soon as the parent job process finishes, all of its spawned
        subprocesses will be forced to finish, too.

        Upon return, the whole process tree of the spawned job process will be
        cleared, unless any of them has called `setsid()`.

        Keyword arguments:

        timeout -- Timeout period for this wait call in seconds. If `None` the
                   default time limit will be used.
        """
        if self._state is not None:
            # Job has been already waited for
            return

        if timeout is None:
            # Convert time_limit to seconds
            h, m, s = self._time_limit
            timeout = h * 3600 + m * 60 + s

        try:
            self._wait_all(timeout=timeout)
            self._exitcode = self._proc.returncode
            if self._exitcode != 0:
                self._state = LOCAL_JOB_FAILURE
            else:
                self._state = LOCAL_JOB_SUCCESS
        except (_TimeoutExpired, subprocess.TimeoutExpired):
            getlogger().debug('job timed out')
            self._state = LOCAL_JOB_TIMEOUT
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
                 use_smt=None,
                 sched_nodelist=None,
                 sched_exclude=None,
                 sched_partition=None,
                 sched_reservation=None,
                 sched_account=None,
                 num_tasks_per_node=None,
                 num_cpus_per_task=None,
                 num_tasks_per_core=None,
                 num_tasks_per_socket=None,
                 exclusive_access=False,
                 **kwargs):
        super().__init__(**kwargs)
        self._partition   = sched_partition
        self._use_smt     = use_smt
        self._exclusive_access = exclusive_access
        self._nodelist    = sched_nodelist
        self._exclude     = sched_exclude
        self._reservation = sched_reservation
        self._account     = sched_account
        self._prefix      = '#SBATCH'
        self._signal      = None
        self._job_init_poll_num_tries = 0

        self._num_tasks_per_node = num_tasks_per_node
        self._num_cpus_per_task  = num_cpus_per_task
        self._num_tasks_per_core = num_tasks_per_core
        self._num_tasks_per_socket = num_tasks_per_socket
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

    @property
    def account(self):
        return self._account

    @property
    def exclude_list(self):
        return self._exclude

    @property
    def exclusive_access(self):
        return self._exclusive_access

    @property
    def nodelist(self):
        return self._nodelist

    @property
    def num_cpus_per_task(self):
        return self._num_cpus_per_task

    @property
    def num_tasks_per_core(self):
        return self._num_tasks_per_core

    @property
    def num_tasks_per_node(self):
        return self._num_tasks_per_node

    @property
    def num_tasks_per_socket(self):
        return self._num_tasks_per_socket

    @property
    def partition(self):
        return self._partition

    @property
    def reservation(self):
        return self._reservation

    @property
    def use_smt(self):
        return self._use_smt

    @property
    def signal(self):
        return self._signal

    def emit_preamble(self, builder):
        builder.verbatim('%s --job-name="%s"' % (self._prefix, self._name))
        builder.verbatim('%s --time=%s' %
                         (self._prefix, '%d:%d:%d' % self._time_limit))
        builder.verbatim('%s --ntasks=%d' % (self._prefix, self._num_tasks))
        if self._num_tasks_per_node:
            builder.verbatim('%s --ntasks-per-node=%d' %
                             (self._prefix, self._num_tasks_per_node))

        if self._num_cpus_per_task:
            builder.verbatim('%s --cpus-per-task=%d' %
                             (self._prefix, self._num_cpus_per_task))

        if self._num_tasks_per_core:
            builder.verbatim('%s --ntasks-per-core=%d' %
                             (self._prefix, self._num_tasks_per_core))

        if self._num_tasks_per_socket:
            builder.verbatim('%s --ntasks-per-socket=%d' %
                             (self._prefix, self._num_tasks_per_socket))

        if self._partition:
            builder.verbatim('%s --partition=%s' %
                             (self._prefix, self._partition))

        if self._exclusive_access:
            builder.verbatim('%s --exclusive' % self._prefix)

        if self._account:
            builder.verbatim(
                '%s --account=%s' % (self._prefix, self._account))

        if self._nodelist:
            builder.verbatim(
                '%s --nodelist=%s' % (self._prefix, self._nodelist))

        if self._exclude:
            builder.verbatim(
                '%s --exclude=%s' % (self._prefix, self._exclude))

        if self._use_smt is not None:
            hint = 'multithread' if self._use_smt else 'nomultithread'
            builder.verbatim('%s --hint=%s' % (self._prefix, hint))

        if self._reservation:
            builder.verbatim('%s --reservation=%s' % (self._prefix,
                                                      self._reservation))
        if self._stdout:
            builder.verbatim('%s --output="%s"' % (self._prefix, self._stdout))

        if self._stderr:
            builder.verbatim('%s --error="%s"' % (self._prefix, self._stderr))

        for opt in self.options:
            builder.verbatim('%s %s' % (self._prefix, opt))

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
        self._jobid = jobid_match.group('jobid')
        if not self._stdout:
            self._stdout = 'slurm-%s.out' % self._jobid

        if not self._stderr:
            self._stderr = self._stdout

    def _update_state(self):
        """Check the status of the job."""
        intervals = itertools.cycle(settings.job_init_poll_intervals)
        state_match = None
        max_tries = settings.job_init_poll_max_tries
        while (not state_match and
               self._job_init_poll_num_tries < max_tries):
            # Query job state persistently. When you first submit, the job may
            # not be yet registered in the database; so try some times We
            # restrict the `sacct' query to today (`-S' option), so as to avoid
            # possible older and stale slurm database entries.
            completed = os_ext.run_command(
                'sacct -S %s -P -j %s -o jobid,state,exitcode' %
                (datetime.now().strftime('%F'), self._jobid),
                check=True)
            state_match = re.search(
                '^(?P<jobid>\d+)\|(?P<state>\S+)([^\|]*)\|'
                '(?P<exitcode>\d+)\:(?P<signal>\d+)',
                completed.stdout, re.MULTILINE)
            if not state_match:
                getlogger().debug('job state not matched (stdout follows)\n%s' %
                                  completed.stdout)
                self._job_init_poll_num_tries += 1
                time.sleep(next(intervals))

        if not state_match:
            raise ReframeError('querying initial job state timed out')

        assert self._jobid == state_match.group('jobid')

        self._state    = SlurmJobState(state_match.group('state'))
        self._exitcode = int(state_match.group('exitcode'))
        self._signal   = int(state_match.group('signal'))

    def _cancel_if_blocked(self):
        if self._is_cancelling or self._state not in self._pending_states:
            return

        completed = os_ext.run_command('squeue -j %s -o "%%i|%%T|%%r" ' %
                                       self._jobid, check=True)
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

        assert self._jobid == state_match.group('jobid')

        # Ensure that the job is still in a pending state
        state  = SlurmJobState(state_match.group('state'))
        reason = state_match.group('reason')
        if state in self._pending_states and reason in self._cancel_reasons:
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
        if self._state in self._completion_states:
            return

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
        if self._jobid == -1:
            return

        os_ext.run_command('scancel %s' % self._jobid,
                           check=True, timeout=settings.job_submit_timeout)
        self._is_cancelling = True
        self.wait()

    def finished(self):
        try:
            self._update_state()
        except ReframeError as e:
            # We postpone exception handling: we ignore the exception at this
            # point and mark the job as unfinished in order to deal with it
            # later
            getlogger().debug('ignoring error during polling: %s' % e)
            return False
        else:
            return self._state in self._completion_states
