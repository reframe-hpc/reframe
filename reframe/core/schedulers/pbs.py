# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# PBS backend
#
# - Initial version submitted by Rafael Escovar, ASML
#

import functools
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
from reframe.utility import seconds_to_hms


# Time to wait after a job is finished for its standard output/error to be
# written to the corresponding files.
PBS_OUTPUT_WRITEBACK_WAIT = 3


_run_strict = functools.partial(os_ext.run_command, check=True)


@register_scheduler('pbs')
class PbsJobScheduler(sched.JobScheduler):
    def __init__(self):
        self._prefix = '#PBS'
        self._time_finished = None

        # Optional part of the job id refering to the PBS server
        self._pbs_server = None

    def completion_time(self, job):
        return None

    def _emit_lselect_option(self, job):
        num_tasks_per_node = job.num_tasks_per_node or 1
        num_cpus_per_task = job.num_cpus_per_task or 1
        num_nodes = job.num_tasks // num_tasks_per_node
        num_cpus_per_node = num_tasks_per_node * num_cpus_per_task
        select_opt = '-l select=%s:mpiprocs=%s:ncpus=%s' % (num_nodes,
                                                            num_tasks_per_node,
                                                            num_cpus_per_node)

        # Options starting with `-` are emitted in separate lines
        rem_opts = []
        verb_opts = []
        for opt in (*job.sched_access, *job.options):
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

    def emit_preamble(self, job):
        preamble = [
            self._format_option('-N "%s"' % job.name),
            self._format_option('-o %s' % job.stdout),
            self._format_option('-e %s' % job.stderr),
        ]

        if job.time_limit is not None:
            h, m, s = seconds_to_hms(job.time_limit.total_seconds())
            preamble.append(
                self._format_option('-l walltime=%d:%d:%d' % (h, m, s)))

        if job.sched_partition:
            preamble.append(
                self._format_option('-q %s' % job.sched_partition))

        preamble += self._emit_lselect_option(job)

        # PBS starts the job in the home directory by default
        preamble.append('cd %s' % job.workdir)
        return preamble

    def allnodes(self):
        raise NotImplementedError('pbs backend does not support node listing')

    def filternodes(self, job, nodes):
        raise NotImplementedError('pbs backend does not support '
                                  'node filtering')

    def submit(self, job):
        # `-o` and `-e` options are only recognized in command line by the PBS
        # Slurm wrappers.
        cmd = 'qsub -o %s -e %s %s' % (job.stdout, job.stderr,
                                       job.script_filename)
        completed = _run_strict(cmd, timeout=settings().job_submit_timeout)
        jobid_match = re.search(r'^(?P<jobid>\S+)', completed.stdout)
        if not jobid_match:
            raise JobError('could not retrieve the job id '
                           'of the submitted job')

        jobid, *info = jobid_match.group('jobid').split('.', maxsplit=2)
        job.jobid = int(jobid)
        if info:
            self._pbs_server = info[0]

    def wait(self, job):
        intervals = itertools.cycle(settings().job_poll_intervals)
        while not self.finished(job):
            time.sleep(next(intervals))

    def cancel(self, job):
        # Recreate the full job id
        jobid = str(job.jobid)
        if self._pbs_server:
            jobid += '.' + self._pbs_server

        getlogger().debug('cancelling job (id=%s)' % jobid)
        _run_strict('qdel %s' % jobid, timeout=settings().job_submit_timeout)

    def finished(self, job):
        with os_ext.change_dir(job.workdir):
            done = os.path.exists(job.stdout) and os.path.exists(job.stderr)

        if done:
            t_now = datetime.now()
            self._time_finished = self._time_finished or t_now
            time_from_finish = (t_now - self._time_finished).total_seconds()

        return done and time_from_finish > PBS_OUTPUT_WRITEBACK_WAIT
