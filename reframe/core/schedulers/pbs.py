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

import reframe.core.runtime as rt
import reframe.core.schedulers as sched
import reframe.utility.os_ext as os_ext
from reframe.core.backends import register_scheduler
from reframe.core.config import settings
from reframe.core.exceptions import SpawnedProcessError, JobError
from reframe.core.logging import getlogger
from reframe.utility import seconds_to_hms


# Time to wait after a job is finished for its standard output/error to be
# written to the corresponding files.
PBS_OUTPUT_WRITEBACK_WAIT = 3


# Minimum amount of time between its submission and its cancellation. If you
# immediately cancel a PBS job after submission, its output files may never
# appear in the output causing the wait() to hang.
PBS_CANCEL_DELAY = 3


_run_strict = functools.partial(os_ext.run_command, check=True)


@register_scheduler('pbs')
class PbsJobScheduler(sched.JobScheduler):
    TASKS_OPT = ('-l select={num_nodes}:mpiprocs={num_tasks_per_node}'
                 ':ncpus={num_cpus_per_node}')

    def __init__(self):
        self._prefix = '#PBS'
        self._time_finished = None
        self._job_submit_timeout = rt.runtime().get_option(
            f'schedulers/@{self.registered_name}/job_submit_timeout'
        )
        self._cancelled = False

        # Optional part of the job id refering to the PBS server
        self._pbs_server = None

    def completion_time(self, job):
        return None

    def _emit_lselect_option(self, job):
        num_tasks_per_node = job.num_tasks_per_node or 1
        num_cpus_per_task = job.num_cpus_per_task or 1
        num_nodes = job.num_tasks // num_tasks_per_node
        num_cpus_per_node = num_tasks_per_node * num_cpus_per_task
        select_opt = self.TASKS_OPT.format(
            num_nodes=num_nodes,
            num_tasks_per_node=num_tasks_per_node,
            num_cpus_per_node=num_cpus_per_node
        )

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
        completed = _run_strict(cmd, timeout=self._job_submit_timeout)
        jobid_match = re.search(r'^(?P<jobid>\S+)', completed.stdout)
        if not jobid_match:
            raise JobError('could not retrieve the job id '
                           'of the submitted job')

        jobid, *info = jobid_match.group('jobid').split('.', maxsplit=1)
        job.jobid = int(jobid)
        if info:
            self._pbs_server = info[0]

        self._submit_time = datetime.now()

    def wait(self, job):
        intervals = itertools.cycle([1, 2, 3])
        while not self.finished(job):
            time.sleep(next(intervals))

    def cancel(self, job):
        self._cancelled = True

        # Recreate the full job id
        jobid = str(job.jobid)
        if self._pbs_server:
            jobid += '.' + self._pbs_server

        time_from_submit = (datetime.now() - self._submit_time).total_seconds()
        if time_from_submit < PBS_CANCEL_DELAY:
            time.sleep(PBS_CANCEL_DELAY - time_from_submit)

        getlogger().debug('cancelling job (id=%s)' % jobid)
        _run_strict('qdel %s' % jobid, timeout=self._job_submit_timeout)

    def finished(self, job):
        with os_ext.change_dir(job.workdir):
            output_ready = (os.path.exists(job.stdout) and
                            os.path.exists(job.stderr))

        done = self._cancelled or output_ready

        if done:
            t_now = datetime.now()
            self._time_finished = self._time_finished or t_now
            time_from_finish = (t_now - self._time_finished).total_seconds()

        return done and time_from_finish > PBS_OUTPUT_WRITEBACK_WAIT
