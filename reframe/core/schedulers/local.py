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
from datetime import datetime, timedelta

import reframe.core.schedulers as sched
import reframe.utility.os_ext as os_ext
from reframe.core.backends import register_scheduler
from reframe.core.exceptions import ReframeError
from reframe.core.logging import getlogger


class _TimeoutExpired(ReframeError):
    pass


@register_scheduler('local', local=True)
class LocalJobScheduler(sched.JobScheduler):
    CANCEL_GRACE_PERIOD = 2
    WAIT_POLL_SECS = 0.1

    def completion_time(self, job):
        return None

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
        job.jobid = proc.pid
        job.nodelist = [socket.gethostname()]
        job.proc = proc
        job.f_stdout = f_stdout
        job.f_stderr = f_stderr

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
            getlogger().debug('pid %s already dead or assigned elsewhere' %
                              job.jobid)

    def _term_all(self, job):
        '''Send SIGTERM to all the processes of the spawned job.'''
        os.killpg(job.jobid, signal.SIGTERM)

    def _wait_all(self, job, timeout=0):
        '''Wait for all the processes of spawned job to finish.

        Keyword arguments:

        timeout -- Timeout period for this wait call in seconds (may be a real
                   number, too). If `None` or `0`, no timeout will be set.
        '''
        t_wait = datetime.now()
        job.proc.wait(timeout=timeout or None)
        t_wait = datetime.now() - t_wait
        try:
            # Wait for all processes in the process group to finish
            while not timeout or t_wait.total_seconds() < timeout:
                t_poll = datetime.now()
                os.killpg(job.jobid, 0)
                time.sleep(self.WAIT_POLL_SECS)
                t_poll = datetime.now() - t_poll
                t_wait += t_poll

            # Final check
            os.killpg(job.jobid, 0)
            raise _TimeoutExpired
        except (ProcessLookupError, PermissionError):
            # Ignore also EPERM errors in case this process id is assigned
            # elsewhere and we cannot query its status
            getlogger().debug('pid %s already dead or assigned elsewhere' %
                              job.jobid)
            return

    def cancel(self, job):
        '''Cancel job.

        The SIGTERM signal will be sent first to all the processes of this job
        and after a grace period (default 2s) the SIGKILL signal will be send.

        This function waits for the spawned process tree to finish.
        '''
        self._term_all(job)

        # Set the time limit to the grace period and let wait() do the final
        # killing
        job.time_limit = timedelta(seconds=self.CANCEL_GRACE_PERIOD)
        self.wait(job)

    def wait(self, job):
        '''Wait for the spawned job to finish.

        As soon as the parent job process finishes, all of its spawned
        subprocesses will be forced to finish, too.

        Upon return, the whole process tree of the spawned job process will be
        cleared, unless any of them has called `setsid()`.
        '''
        if job.state is not None:
            # Job has been already waited for
            return

        # Convert job's time_limit to seconds
        if job.time_limit is not None:
            timeout = job.time_limit.total_seconds()
        else:
            timeout = 0

        try:
            self._wait_all(job, timeout)
            job.exitcode = job.proc.returncode
            if job.exitcode != 0:
                job.state = 'FAILURE'
            else:
                job.state = 'SUCCESS'
        except (_TimeoutExpired, subprocess.TimeoutExpired):
            getlogger().debug('job timed out')
            job.state = 'TIMEOUT'
        finally:
            # Cleanup all the processes of this job
            self._kill_all(job)
            self._wait_all(job)
            job.f_stdout.close()
            job.f_stderr.close()

    def finished(self, job):
        '''Check if the spawned process has finished.

        This function does not wait the process. It just queries its state. If
        the process has finished, you *must* call wait() to properly cleanup
        after it.
        '''
        return job.proc.returncode is not None

    def poll(self, *jobs):
        for job in jobs:
            if job.jobid and job.proc:
                job.proc.poll()


class _LocalNode(sched.Node):
    def __init__(self, name):
        self._name = name

    def in_state(self, state):
        return state.casefold() == 'idle'
