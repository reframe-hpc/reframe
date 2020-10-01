# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import signal
import socket
import stat
import subprocess
import time

import reframe.core.schedulers as sched
import reframe.utility.os_ext as os_ext
from reframe.core.backends import register_scheduler
from reframe.core.exceptions import ReframeError
from reframe.core.logging import getlogger


class _TimeoutExpired(ReframeError):
    pass


class _LocalJob(sched.Job):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._proc = None
        self._f_stdout = None
        self._f_stderr = None

    @property
    def proc(self):
        return self._proc

    @property
    def f_stdout(self):
        return self._f_stdout

    @property
    def f_stderr(self):
        return self._f_stderr


@register_scheduler('local', local=True)
class LocalJobScheduler(sched.JobScheduler):
    CANCEL_GRACE_PERIOD = 2
    WAIT_POLL_SECS = 0.1

    def make_job(self, *args, **kwargs):
        return _LocalJob(*args, **kwargs)

    def submit(self, job):
        # `chmod +x' first, because we will execute the script locally
        os.chmod(job.script_filename,
                 os.stat(job.script_filename).st_mode | stat.S_IEXEC)

        # Run from the absolute path
        f_stdout = open(job.stdout, 'w+')
        f_stderr = open(job.stderr, 'w+')

        # The new process starts also a new session (session leader), so that
        # we can later kill any other processes that this might spawn by just
        # killing this one.
        proc = os_ext.run_command_async(
            os.path.abspath(job.script_filename),
            stdout=f_stdout,
            stderr=f_stderr,
            start_new_session=True
        )

        # Update job info
        job._jobid = proc.pid
        job._nodelist = [socket.gethostname()]
        job._proc = proc
        job._f_stdout = f_stdout
        job._f_stderr = f_stderr
        job._submit_time = time.time()
        job._state = 'RUNNING'

    def emit_preamble(self, job):
        return []

    def allnodes(self):
        return [_LocalNode(socket.gethostname())]

    def filternodes(self, job, nodes):
        return [_LocalNode(socket.gethostname())]

    def _kill_all(self, job):
        '''Send SIGKILL to all the processes of the spawned job.'''
        try:
            os.killpg(job.jobid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            # The process group may already be dead or assigned to a different
            # group, so ignore this error
            getlogger().debug(
                f'pid {job.jobid} already dead or assigned elsewhere'
            )

    def _term_all(self, job):
        '''Send SIGTERM to all the processes of the spawned job.'''
        os.killpg(job.jobid, signal.SIGTERM)

    def cancel(self, job):
        '''Cancel job.

        The SIGTERM signal will be sent first to all the processes of this job
        and after a grace period (default 2s) the SIGKILL signal will be send.

        This function waits for the spawned process tree to finish.
        '''
        self._term_all(job)

        # Set the time limit to the grace period and let wait() do the final
        # killing
        job.time_limit = time.time() - job.submit_time + self.CANCEL_GRACE_PERIOD

    def wait(self, job):
        '''Wait for the spawned job to finish.

        As soon as the parent job process finishes, all of its spawned
        subprocesses will be forced to finish, too.

        Upon return, the whole process tree of the spawned job process will be
        cleared, unless any of them has called `setsid()`.
        '''

        while not self.finished(job):
            self.poll(job)
            time.sleep(self.WAIT_POLL_SECS)

        job.f_stdout.close()
        job.f_stderr.close()

    def finished(self, job):
        '''Check if the spawned process has finished.

        This function does not wait the process. It just queries its state. If
        the process has finished, you *must* call wait() to properly cleanup
        after it.
        '''
        return job.state in ['SUCCESS', 'FAILURE', 'TIMEOUT']

    def poll(self, *jobs):
        for job in jobs:
            self._poll_job(job)

    def _poll_job(self, job):
        if job.jobid is None:
            return

        try:
            os.killpg(job.jobid, 0)
        except (ProcessLookupError, PermissionError):
            # Spawned session has finished; call wait on the
            # subprocess object to get the exit code, etc.
            job.proc.wait()
            job._exitcode = job.proc.returncode
            job._state = 'FAILURE' if job.exitcode != 0 else 'SUCCESS'
            job.f_stdout.close()
            job.f_stderr.close()
        else:
            # Job is still alive; check if it should time out
            t_elapsed = time.time() - job.submit_time
            if job.time_limit and t_elapsed > job.time_limit:
                self._kill_all(job)
                job.proc.wait()
                job._exitcode = job.proc.returncode
                job._state = 'TIMEOUT'
                job.f_stdout.close()
                job.f_stderr.close()


class _LocalNode(sched.Node):
    def __init__(self, name):
        self._name = name

    def in_state(self, state):
        return state.casefold() == 'idle'
