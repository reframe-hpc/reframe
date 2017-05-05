#
# Scheduler implementations
#

import itertools
import os
import re
import stat
import subprocess
import time

import reframe.utility.os as os_ext

from datetime import datetime
from reframe.core.exceptions import ReframeError, JobSubmissionError
from reframe.core.launchers import LocalLauncher
from reframe.core.shell import BashScriptBuilder
from reframe.settings import settings


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
        self.environs = job_environ_list if job_environ_list else []
        self.script_builder = job_script_builder
        self.num_tasks = num_tasks
        self.script_filename = script_filename \
                               if script_filename else '%s.sh' % self.name
        self.options = options
        self.launcher = launcher(self, launcher_options)
        self.stdout = stdout if stdout else '%s.out' % self.name
        self.stderr = stderr if stderr else '%s.err' % self.name

        # Commands to be run before and after the job is launched
        self.pre_run  = []
        self.post_run = []

        # Live job information; to be filled during job's lifetime
        self.jobid    = None
        self.state    = None
        self.exitcode = None


    def emit_preamble(self, builder):
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


    def submit(self, cmd, workdir = '.'):
        # Build the submission script and submit it
        self.emit_preamble(self.script_builder)
        for e in self.environs:
            e.emit_load_instructions(self.script_builder)

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


    def __eq__(self, other):
        return other != None and self.state == other.state


    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return self.state


class JobResources:
    """Managed job resources.

    Custom resources usually configured per system by the system administrators.
    """
    def __init__(self, resources):
        self.resources = resources


    def get(self, name, **kwargs):
        """Get resource option string for the resource `name'"""
        try:
            return self.resources.format(**kwargs)
        except KeyError:
            return None


    def getall(self, resources_spec):
        """
        Return a list of resource option strings for all the resources specified in
        `resourse_spec'
        """
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
                 time_limit = (0, 10, 0),
                 **kwargs):
        super().__init__(num_tasks=1,
                         launcher=LocalLauncher,
                         **kwargs)
        # Process launched
        self.proc = None
        self.time_limit = time_limit


    def _submit(self, script):
        # `chmod +x' first, because we will execute the script locally
        os.chmod(script.name, os.stat(script.name).st_mode | stat.S_IEXEC);

        # Run from the absolute path
        self._stdout = open(self.stdout, 'w+')
        self._stderr = open(self.stderr, 'w+')
        self.proc = os_ext.run_command_async(os.path.abspath(script.name),
                                             stdout=self._stdout,
                                             stderr=self._stderr)

        # update job info
        self.jobid = self.proc.pid


    def wait(self):
        # convert timeout to seconds
        h, m, s = self.time_limit
        timeout = h * 3600 + m * 60 + s
        # wait for spawned process to finish
        try:
            self.proc.wait(timeout=timeout)
            self.exitcode = self.proc.returncode
            if self.exitcode != 0:
                self.state = LOCAL_JOB_FAILURE
            else:
                self.state = LOCAL_JOB_SUCCESS
        except subprocess.TimeoutExpired:
            self.proc.kill()
            # we need the wait to avoid zombie processes
            self.proc.wait()
            self.state = LOCAL_JOB_TIMEOUT

        # close stdout/stderr
        finally:
            self._stdout.close()
            self._stderr.close()


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
SLURM_JOB_RUNNING     = SlurmJobState('RUNNING')
SLURM_JOB_RESIZING    = SlurmJobState('RESIZING')
SLURM_JOB_SUSPENDED   = SlurmJobState('SUSPENDED')
SLURM_JOB_TIMEOUT     = SlurmJobState('TIMEOUT')


class SlurmJob(Job):
    def __init__(self,
                 time_limit = (0, 10, 0),
                 use_smt = False,
                 exclusive = True,
                 nodelist = None,
                 exclude = None,
                 partition = None,
                 reservation = None,
                 account = None,
                 num_tasks_per_node=None,
                 num_cpus_per_task=None,
                 num_tasks_per_core=None,
                 num_tasks_per_socket=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.partition   = partition
        self.time_limit  = time_limit
        self.use_smt     = use_smt
        self.exclusive   = exclusive
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

    def emit_preamble(self, builder):
        builder.verbatim('%s --job-name="%s"' % (self.prefix, self.name))
        builder.verbatim('%s --time=%s' % (self.prefix,
                                           '%d:%d:%d' % self.time_limit))
        builder.verbatim('%s --ntasks=%d' % (self.prefix, self.num_tasks))
        if self.num_tasks_per_node:
            builder.verbatim('%s --ntasks-per-node=%d' % (self.prefix,
                                                          self.num_tasks_per_node))
        if self.num_cpus_per_task:
            builder.verbatim('%s --cpus-per-task=%d' % (self.prefix,
                                                          self.num_cpus_per_task))
        if self.num_tasks_per_core:
            builder.verbatim('%s --ntasks-per-core=%d' % (self.prefix,
                                                          self.num_tasks_per_core))
        if self.num_tasks_per_socket:
            builder.verbatim('%s --ntasks-per-socket=%d' % (self.prefix,
                                                          self.num_tasks_per_socket))
        if self.partition:
            builder.verbatim('%s --partition=%s' % (self.prefix, self.partition))

        if self.exclusive:
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

        if self.use_smt:
            builder.verbatim('%s --hint=multithread' % self.prefix)
        else:
            builder.verbatim('%s --hint=nomultithread' % self.prefix)

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
        """
        Check the status of the job.
        """
        intervals = itertools.cycle(settings.job_init_poll_intervals)
        state_match = None
        while not state_match and \
              self.job_init_poll_num_tries < settings.job_init_poll_max_tries:
            # Query job state persistently. When you first submit, the job may
            # not be yet registered in the database; so try some times We
            # restrict the `sacct' query to today (`-S' option), so as to avoid
            # possible older and stale slurm database entries.
            completed = os_ext.run_command(
                'sacct -S %s -P -j %s -o jobid,state,exitcode' % \
                (datetime.now().strftime('%F'), self.jobid),
                check=True)
            state_match = re.search(
                '^(?P<jobid>\d+)\|(?P<state>\S+)\|'
                '(?P<exitcode>\d+)\:(?P<signal>\d+)',
                completed.stdout, re.MULTILINE)
            if not state_match:
                self.job_init_poll_num_tries += 1
                time.sleep(next(intervals))

        if not state_match:
            raise ReframeError('Querying initial job state timed out')

        if state_match.group('jobid') != self.jobid:
            # this shouldn't happen
            raise RegressionFatalError(
                'Oops: job ids do not match. Expected %s, got %s' % \
                (self.jobid, state_match.group('jobid')))

        self.state    = JobState(state_match.group('state'))
        self.exitcode = int(state_match.group('exitcode'))
        self.signal   = int(state_match.group('signal'))

    def wait(self, states = [ SLURM_JOB_BOOT_FAIL,
                              SLURM_JOB_CANCELLED,
                              SLURM_JOB_COMPLETED,
                              SLURM_JOB_FAILED,
                              SLURM_JOB_NODE_FAILED,
                              SLURM_JOB_PREEMPTED,
                              SLURM_JOB_TIMEOUT ]):
        if not states:
            raise RuntimeError('No state was specified to wait for.')

        intervals = itertools.cycle(settings.job_state_poll_intervals)

        self._update_state()
        while not self.state or not self.state in states:
            time.sleep(next(intervals))
            self._update_state()
