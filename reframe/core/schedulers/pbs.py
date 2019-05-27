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
from reframe.core.exceptions import SpawnedProcessError, JobError
from reframe.core.logging import getlogger
from reframe.core.schedulers.registry import register_scheduler


# Time to wait after a job is finished for its standard output/error to be
# written to the corresponding files.
PBS_OUTPUT_WRITEBACK_WAIT = 3


@register_scheduler('pbs')
class PbsJob(sched.Job):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prefix = '#PBS'
        self._time_finished = None

        # Optional part of the job id refering to the PBS server
        self._pbs_server = None

    def _emit_lselect_option(self):
        num_tasks_per_node = self._num_tasks_per_node or 1
        num_cpus_per_task = self._num_cpus_per_task or 1
        num_nodes = self._num_tasks // num_tasks_per_node
        num_cpus_per_node = num_tasks_per_node * num_cpus_per_task
        select_opt = '-l select=%s:mpiprocs=%s:ncpus=%s' % (num_nodes,
                                                            num_tasks_per_node,
                                                            num_cpus_per_node)

        # Options starting with `-` are emitted in separate lines
        rem_opts = []
        verb_opts = []
        for opt in (*self.sched_access, *self.options):
            if opt.startswith('-'):
                rem_opts.append(opt)
            elif opt.startswith('#'):
                verb_opts.append(opt)
            else:
                select_opt += ':' + opt

        return [self._format_option(select_opt),
                *(self._format_option(opt) for opt in rem_opts),
                *verb_opts]

    def _format_option(self, option):
        return self._prefix + ' ' + option

    def _run_command(self, cmd, timeout=None):
        """Run command cmd and re-raise any exception as a JobError."""
        try:
            return os_ext.run_command(cmd, check=True, timeout=timeout)
        except SpawnedProcessError as e:
            raise JobError(jobid=self._jobid) from e

    def emit_preamble(self):
        preamble = [
            self._format_option('-N "%s"' % self.name),
            self._format_option('-o %s' % self.stdout),
            self._format_option('-e %s' % self.stderr),
        ]

        if self.time_limit is not None:
            preamble.append(
                self._format_option('-l walltime=%d:%d:%d' % self.time_limit))

        if self.sched_partition:
            preamble.append(
                self._format_option('-q %s' % self.sched_partition))

        preamble += self._emit_lselect_option()

        # PBS starts the job in the home directory by default
        preamble.append('cd %s' % self.workdir)
        return preamble

    def get_all_nodes(self):
        raise NotImplementedError('pbs backend does not support node listing')

    def filter_nodes(self, nodes, options):
        raise NotImplementedError('pbs backend does not support '
                                  'node filtering')

    def submit(self):
        # `-o` and `-e` options are only recognized in command line by the PBS
        # Slurm wrappers.
        cmd = 'qsub -o %s -e %s %s' % (self.stdout, self.stderr,
                                       self.script_filename)
        completed = self._run_command(cmd, settings().job_submit_timeout)
        jobid_match = re.search(r'^(?P<jobid>\S+)', completed.stdout)
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
        with os_ext.change_dir(self.workdir):
            done = os.path.exists(self.stdout) and os.path.exists(self.stderr)

        if done:
            t_now = datetime.now()
            self._time_finished = self._time_finished or t_now
            time_from_finish = (t_now - self._time_finished).total_seconds()

        return done and time_from_finish > PBS_OUTPUT_WRITEBACK_WAIT
