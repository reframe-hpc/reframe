# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import os
import signal
import socket
import time
import psutil

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

    async def submit(self, job):
        # Run from the absolute path
        f_stdout = open(job.stdout, 'w+')
        f_stderr = open(job.stderr, 'w+')

        # The new process starts also a new session (session leader), so that
        # we can later kill any other processes that this might spawn by just
        # killing this one.
        proc = await osext.run_command_asyncio(
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
        '''Send SIGKILL to all the processes of the spawned job.'''
        try:
            # Get the process with psutil because we need to cancel the group
            p = psutil.Process(job.jobid)
            # Get the children of this group
            job.children = p.children(recursive=True)
            children = job.children
        except psutil.NoSuchProcess:
            try:
                # Maybe the main process was already killed/terminated
                # but the children were not
                children = job.children
            except AttributeError:
                children = []

        try:
            for child in children:
                if child.is_running():
                    child.send_signal(signal.SIGKILL)
                    job._signal = signal.SIGKILL
                else:
                    self.log(f'child pid {child.pid} already dead')
            job.proc.send_signal(signal.SIGKILL)
            job._signal = signal.SIGKILL
        except (ProcessLookupError, PermissionError):
            # The process group may already be dead or assigned to a different
            # group, so ignore this error
            self.log(f'pid {job.jobid} already dead')
        finally:
            # Close file handles
            job.f_stdout.close()
            job.f_stderr.close()
            job._state = 'FAILURE'

    def _term_all(self, job):
        '''Send SIGTERM to all the processes of the spawned job.'''

        try:
            p = psutil.Process(job.jobid)
            # Get the chilldren of the process
            job.children = p.children(recursive=True)
        except psutil.NoSuchProcess:
            job.children = []

        try:
            job.proc.send_signal(signal.SIGTERM)
            job._signal = signal.SIGTERM
            # Here, we don't know if it was ignored or not
            for child in job.children:
                # try to kill the children
                try:
                    child.send_signal(signal.SIGTERM)
                except (ProcessLookupError, PermissionError,
                        psutil.NoSuchProcess):
                    # The process group may already be dead or assigned
                    # to a different group, so ignore this error
                    self.log(f'child pid {child.pid} already dead')

        except (ProcessLookupError, PermissionError):
            # Job has finished already, close file handles
            self.log(f'pid {job.jobid} already dead')
            job.f_stdout.close()
            job.f_stderr.close()
            job._state = 'FAILURE'

    def cancel(self, job):
        '''Cancel job.

        The SIGTERM signal will be sent first to all the processes of this job
        and after a grace period (default 2s) the SIGKILL signal will be send.

        This function waits for the spawned process tree to finish.
        '''
        self._term_all(job)
        job._cancel_time = time.time()

    async def wait(self, job):
        '''Wait for the spawned job to finish.

        As soon as the parent job process finishes, all of its spawned
        subprocesses will be forced to finish, too.

        Upon return, the whole process tree of the spawned job process will be
        cleared, unless any of them has called `setsid()`.
        '''

        while not self.finished(job):
            await self.poll(job)
            await asyncio.sleep(self.WAIT_POLL_SECS)

    def finished(self, job):
        '''Check if the spawned process has finished.

        This function does not wait the process. It just queries its state. If
        the process has finished, you *must* call wait() to properly cleanup
        after it.
        '''
        if job.exception:
            raise job.exception

        return job.state in ['SUCCESS', 'FAILURE', 'TIMEOUT']

    async def poll(self, *jobs):
        for job in jobs:
            await self._poll_job(job)

    async def _poll_job(self, job):
        if (job is None or job.jobid is None or job.finished()):
            return

        if job.cancel_time:
            # Job has been cancelled; give it a grace period and kill it
            self.log(f'Job {job.jobid} has been cancelled; '
                     f'giving it a grace period')
            t_rem = self.CANCEL_GRACE_PERIOD - (time.time() - job.cancel_time)
            if t_rem > 0:
                await asyncio.sleep(t_rem)

            self._kill_all(job)
            return

        if job.proc.returncode is None:
            # Job has not finished; check if we have reached a timeout
            t_elapsed = time.time() - job.submit_time
            if job.time_limit and t_elapsed > job.time_limit:
                self._kill_all(job)
                job._state = 'TIMEOUT'
                job._exception = JobError(
                    f'job timed out ({t_elapsed:.6f}s > {job.time_limit}s)',
                    job.jobid
                )

            return

        # Job has finished; kill the whole session
        self._kill_all(job)

        # Retrieve the status of the job and return
        if job.proc.returncode >= 0:
            job._exitcode = job.proc.returncode
            job._state = 'FAILURE' if job.exitcode != 0 else 'SUCCESS'
        else:
            job._state = 'FAILURE'
            job._signal = job.proc.returncode
