# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import errno
import os
import signal
import socket
import time

import reframe.core.schedulers as sched
import reframe.utility.osext as osext
from reframe.core.backends import register_scheduler
from reframe.core.exceptions import ReframeError


class _TimeoutExpired(ReframeError):
    pass


class _LocalJob(sched.Job):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._proc = None
        self._f_stdout = None
        self._f_stderr = None
        self._signal = None
        self._cancel_time = None

    @property
    def proc(self):
        return self._proc

    @property
    def f_stdout(self):
        return self._f_stdout

    @property
    def f_stderr(self):
        return self._f_stderr

    @property
    def signal(self):
        return self._signal

    @property
    def cancel_time(self):
        return self._cancel_time


@register_scheduler('local', local=True)
class LocalJobScheduler(sched.JobScheduler):
    CANCEL_GRACE_PERIOD = 2
    WAIT_POLL_SECS = 0.001

    def make_job(self, *args, **kwargs):
        return _LocalJob(*args, **kwargs)

    def submit(self, job):
        # Run from the absolute path
        f_stdout = open(job.stdout, 'w+')
        f_stderr = open(job.stderr, 'w+')

        # The new process starts also a new session (session leader), so that
        # we can later kill any other processes that this might spawn by just
        # killing this one.
        proc = osext.run_command_async(
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
            job._signal = signal.SIGKILL
        except (ProcessLookupError, PermissionError):
            # The process group may already be dead or assigned to a different
            # group, so ignore this error
            self.log(f'pid {job.jobid} already dead or assigned elsewhere')
        finally:
            # Close file handles
            job.f_stdout.close()
            job.f_stderr.close()
            job._state = 'FAILURE'

    def _term_all(self, job):
        '''Send SIGTERM to all the processes of the spawned job.'''
        os.killpg(job.jobid, signal.SIGTERM)
        job._signal = signal.SIGTERM

    def cancel(self, job):
        '''Cancel job.

        The SIGTERM signal will be sent first to all the processes of this job
        and after a grace period (default 2s) the SIGKILL signal will be send.

        This function waits for the spawned process tree to finish.
        '''
        self._term_all(job)
        job._cancel_time = time.time()

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

    def finished(self, job):
        '''Check if the spawned process has finished.

        This function does not wait the process. It just queries its state. If
        the process has finished, you *must* call wait() to properly cleanup
        after it.
        '''
        if job.exception:
            raise job.exception

        return job.state in ['SUCCESS', 'FAILURE', 'TIMEOUT']

    def poll(self, *jobs):
        for job in jobs:
            self._poll_job(job)

    def _poll_job(self, job):
        if job is None or job.jobid is None:
            return

        try:
            pid, status = os.waitpid(job.jobid, os.WNOHANG)
        except OSError as e:
            if e.errno == errno.ECHILD:
                # No unwaited children
                self.log('no more unwaited children')
                return
            else:
                raise e

        if job.cancel_time:
            # Job has been cancelled; give it a grace period and kill it
            self.log(f'Job {job.jobid} has been cancelled; '
                     f'giving it a grace period')
            t_rem = self.CANCEL_GRACE_PERIOD - (time.time() - job.cancel_time)
            if t_rem > 0:
                time.sleep(t_rem)

            self._kill_all(job)
            return

        if not pid:
            # Job has not finished; check if we have reached a timeout
            t_elapsed = time.time() - job.submit_time
            if job.time_limit and t_elapsed > job.time_limit:
                self.log(f'Job {job.jobid} timed out; kill it')
                self._kill_all(job)
                job._state = 'TIMEOUT'

            return

        # Job has finished; kill the whole session
        self._kill_all(job)

        # Retrieve the status of the job and return
        if os.WIFEXITED(status):
            job._exitcode = os.WEXITSTATUS(status)
            job._state = 'FAILURE' if job.exitcode != 0 else 'SUCCESS'
        elif os.WIFSIGNALED(status):
            job._state = 'FAILURE'
            job._signal = os.WTERMSIG(status)


class _LocalNode(sched.Node):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    def in_state(self, state):
        return state.casefold() == 'idle'
