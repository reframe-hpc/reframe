import os
import signal
import socket
import stat
import subprocess
import time
from datetime import datetime

import reframe.core.schedulers as sched
import reframe.utility.os_ext as os_ext
from reframe.core.exceptions import ReframeError
from reframe.core.logging import getlogger
from reframe.core.schedulers.registry import register_scheduler


class _TimeoutExpired(ReframeError):
    pass


@register_scheduler('local', local=True)
class LocalJob(sched.Job):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cancel_grace_period = 2
        self._wait_poll_secs = 0.1
        self._proc = None  # Launched process

    def submit(self):
        # `chmod +x' first, because we will execute the script locally
        os.chmod(self._script_filename,
                 os.stat(self._script_filename).st_mode | stat.S_IEXEC)

        # Run from the absolute path
        self._f_stdout = open(self.stdout, 'w+')
        self._f_stderr = open(self.stderr, 'w+')

        # The new process starts also a new session (session leader), so that
        # we can later kill any other processes that this might spawn by just
        # killing this one.
        self._proc = os_ext.run_command_async(
            os.path.abspath(self._script_filename),
            stdout=self._f_stdout,
            stderr=self._f_stderr,
            start_new_session=True)

        # Update job info
        self._jobid = self._proc.pid
        self._nodelist = [socket.gethostname()]

    def emit_preamble(self):
        return []

    def get_all_nodes(self):
        raise NotImplementedError(
            'local scheduler does not support listing of available nodes')

    def filter_nodes(self, nodes, options):
        raise NotImplementedError(
            'local scheduler does not support filtering of available nodes')

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
        """Cancel job.

        The SIGTERM signal will be sent first to all the processes of this job
        and after a grace period (default 2s) the SIGKILL signal will be send.

        This function waits for the spawned process tree to finish.
        """
        super().cancel()
        self._term_all()

        # Set the time limit to the grace period and let wait() do the final
        # killing
        self._time_limit = (0, 0, self.cancel_grace_period)
        self.wait()

    def wait(self):
        """Wait for the spawned job to finish.

        As soon as the parent job process finishes, all of its spawned
        subprocesses will be forced to finish, too.

        Upon return, the whole process tree of the spawned job process will be
        cleared, unless any of them has called `setsid()`.
        """
        super().wait()
        if self._state is not None:
            # Job has been already waited for
            return

        # Convert job's time_limit to seconds
        if self.time_limit is not None:
            h, m, s = self.time_limit
            timeout = h * 3600 + m * 60 + s
        else:
            timeout = 0

        try:
            self._wait_all(timeout)
            self._exitcode = self._proc.returncode
            if self._exitcode != 0:
                self._state = 'FAILURE'
            else:
                self._state = 'SUCCESS'
        except (_TimeoutExpired, subprocess.TimeoutExpired):
            getlogger().debug('job timed out')
            self._state = 'TIMEOUT'
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
        super().finished()
        self._proc.poll()
        if self._proc.returncode is None:
            return False

        return True
