# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import errno
import os
import signal
import socket
import time

import reframe.core.schedulers as sched
import reframe.utility.osext as osext
from reframe.core.backends import register_scheduler
from reframe.core.exceptions import JobError


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
        hostname = socket.gethostname()
        if self.get_option('unqualified_hostnames'):
            job._nodelist = [hostname.split('.')[0]]
        else:
            job._nodelist = [hostname]

        job._proc = proc
        job._f_stdout = f_stdout
        job._f_stderr = f_stderr
        job._submit_time = time.time()
        job._state = 'RUNNING'

    def emit_preamble(self, job):
        return []

    def allnodes(self):
        return [sched.AlwaysIdleNode(socket.gethostname())]

    def filternodes(self, job, nodes):
        return [sched.AlwaysIdleNode(socket.gethostname())]

    def _kill_all(self, job):
        '''Send SIGKILL to all the processes of the spawned job and wait for
        any children to finish'''
        try:
            os.killpg(job._jobid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            # The process group may already be dead or assigned to a different
            # group, so ignore this error
            self.log(f'pid {job.jobid} already dead')
        finally:
            # Close file handles
            job.f_stdout.close()
            job.f_stderr.close()
            with contextlib.suppress(ChildProcessError):
                os.waitpid(0, 0)

    def _term_all(self, job):
        '''Send SIGTERM to all the processes of the spawned job.'''
        try:
            os.killpg(job._jobid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            # Job has finished already, close file handles
            self.log(f'pid {job.jobid} already dead')
            job.f_stdout.close()
            job.f_stderr.close()

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
        return job.exitcode is not None or job.signal is not None

    def poll(self, *jobs):
        for job in jobs:
            self._poll_job(job)

    def _poll_job(self, job):
        if job is None or job._jobid is None:
            return

        try:
            pid, status = os.waitpid(job._jobid, os.WNOHANG)
        except OSError as e:
            if e.errno == errno.ECHILD:
                # No unwaited children
                self.log('no more unwaited children')
                return
            else:
                raise e

        if pid:
            # Job has finished

            # Forcefully kill the whole session once the parent process exits
            self._kill_all(job)

            # Call wait() in the underlying Popen object to avoid false
            # positive warnings
            job._proc.wait()

            # Retrieve the status of the job and return
            if os.WIFEXITED(status):
                job._exitcode = os.WEXITSTATUS(status)
                if job._state == 'RUNNING':
                    job._state = 'FAILURE' if job._exitcode != 0 else 'SUCCESS'
            elif os.WIFSIGNALED(status):
                if job._state == 'RUNNING':
                    job._state = 'FAILURE'

                job._signal = os.WTERMSIG(status)
        else:
            # Job has not finished; check for timeouts
            now = time.time()
            t_elapsed = now - job.submit_time
            if job.cancel_time:
                t_rem = self.CANCEL_GRACE_PERIOD - (now - job.cancel_time)
                self.log(f'Job {job.jobid} has been cancelled; '
                         f'giving it a grace period of {t_rem} seconds')
                if t_rem <= 0:
                    self._kill_all(job)
            elif job.time_limit and t_elapsed > job.time_limit:
                self.cancel(job)
                job._state = 'TIMEOUT'
                job._exception = JobError(
                    f'job timed out ({t_elapsed:.6f}s > {job.time_limit}s)',
                    job.jobid
                )
