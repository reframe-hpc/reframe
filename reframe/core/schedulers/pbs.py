#
# PBS backend
#
# - Initial version submitted by Rafael Escovar, ASML
#

import os
import itertools
import re
import time
from datetime import datetime

import reframe.core.schedulers as sched
import reframe.utility.os_ext as os_ext
from reframe.core.config import settings
from reframe.core.exceptions import (SpawnedProcessError, JobError)
from reframe.core.logging import getlogger
from reframe.core.schedulers.registry import register_scheduler


# Time to wait after a job is finished for its standard output/error to be
# written to the corresponding files.
PBS_OUTPUT_WRITEBACK_WAIT = 3


@register_scheduler('pbs')
class PbsJob(sched.Job):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prefix  = '#PBS'
        self._time_finished = None

        # Optional part of the job id refering to the PBS server
        self._pbs_server = None

    def _emit_lselect_option(self, builder):
        num_tasks_per_node = self._num_tasks_per_node or 1
        num_cpus_per_task = self._num_cpus_per_task or 1
        num_nodes = self._num_tasks // num_tasks_per_node
        num_cpus_per_node = num_tasks_per_node * num_cpus_per_task
        ret = '-l select=%s:mpiprocs=%s:ncpus=%s' % (num_nodes,
                                                     num_tasks_per_node,
                                                     num_cpus_per_node)
        if self.options:
            ret += ':' + ':'.join(self.options)

        self._emit_job_option(ret, builder)

    def _emit_job_option(self, option, builder):
        builder.verbatim(self._prefix + ' ' + option)

    def _run_command(self, cmd, timeout=None):
        """Run command cmd and re-raise any exception as a JobError."""
        try:
            return os_ext.run_command(cmd, check=True, timeout=timeout)
        except SpawnedProcessError as e:
            raise JobError(jobid=self._jobid) from e

    def emit_preamble(self, builder):
        self._emit_job_option('-N "%s"' % self.name, builder)
        self._emit_lselect_option(builder)
        self._emit_job_option('-l walltime=%d:%d:%d' % self.time_limit,
                              builder)
        if self.sched_partition:
            self._emit_job_option('-q %s' % self.sched_partition, builder)

        self._emit_job_option('-o %s' % self.stdout, builder)
        self._emit_job_option('-e %s' % self.stderr, builder)

        # PBS starts the job in the home directory by default
        builder.verbatim('cd %s' % self.workdir)

    def submit(self):
        # `-o` and `-e` options are only recognized in command line by the PBS
        # Slurm wrappers.
        cmd = 'qsub -o %s -e %s %s' % (self.stdout, self.stderr,
                                       self.script_filename)
        completed = self._run_command(cmd, settings().job_submit_timeout)
        jobid_match = re.search('^(?P<jobid>\S+)', completed.stdout)
        if not jobid_match:
            raise JobError('could not retrieve the job id '
                           'of the submitted job')

        jobid, *info = jobid_match.group('jobid').split('.', maxsplit=2)
        self._jobid = int(jobid)
        if info:
            self._pbs_server = info[0]

    def wait(self):
        super().wait()
        intervals = itertools.cycle(settings().job_poll_intervals)
        while not self.finished():
            time.sleep(next(intervals))

    def cancel(self):
        super().cancel()

        # Recreate the full job id
        jobid = str(self._jobid)
        if self._pbs_server:
            jobid += '.' + self._pbs_server

        getlogger().debug('cancelling job (id=%s)' % jobid)
        self._run_command('qdel %s' % jobid, settings().job_submit_timeout)

    def finished(self):
        super().finished()
        done = os.path.exists(self.stdout) and os.path.exists(self.stderr)
        if done:
            t_now = datetime.now()
            self._time_finished = self._time_finished or t_now
            time_from_finish = (t_now - self._time_finished).total_seconds()

        return done and time_from_finish > PBS_OUTPUT_WRITEBACK_WAIT
