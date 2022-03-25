# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# LSF backend
#
# - Initial version submitted by Ryan Goodner, UNM (based on PBS backend)
# - Extended and fixed by Jonathan Frawley and Mark Turner
#

import functools
import re
import time

import reframe.core.runtime as rt
import reframe.utility.osext as osext
from reframe.core.backends import register_scheduler
from reframe.core.exceptions import JobSchedulerError
from reframe.core.schedulers.pbs import PbsJobScheduler

_run_strict = functools.partial(osext.run_command, check=True)


@register_scheduler('lsf')
class LsfJobScheduler(PbsJobScheduler):
    def __init__(self):
        self._prefix = '#BSUB'
        self._submit_timeout = rt.runtime().get_option(
            f'schedulers/@{self.registered_name}/job_submit_timeout'
        )

    def _format_option(self, var, option):
        if var is not None:
            return self._prefix + ' ' + option.format(var)
        else:
            return ''

    def emit_preamble(self, job):
        preamble = [
            self._format_option(job.name, '-J {0}'),
            self._format_option(job.stdout, '-o {0}'),
            self._format_option(job.stderr, '-e {0}'),
        ]

        if job.num_tasks_per_node is not None:
            num_nodes = job.num_tasks // job.num_tasks_per_node
            preamble.append(self._format_option(num_nodes, '-nnodes {0}'))
        else:
            preamble.append(self._format_option(job.num_tasks, '-n {0}'))

        if job.num_cpus_per_task is not None:
            preamble.append(self._format_option(job.num_cpus_per_task,
                                                '-R "affinity[core({0})]"'))

        # add job time limit in minutes
        if job.time_limit is not None:
            preamble.append(
                f'{self._prefix} -W {int(job.time_limit // 60)}'
            )

        for opt in job.sched_access:
            preamble.append(f'{self._prefix} {opt}')

        # emit the rest of the options
        options = job.options + job.cli_options
        for opt in options:
            if opt.startswith('#'):
                preamble.append(opt)
            else:
                preamble.append(self._prefix + ' ' + opt)

        if job.exclusive_access:
            preamble.append(f'{self._prefix} -x')

        # Filter out empty statements before returning
        return list(filter(None, preamble))

    def submit(self, job):
        with open(job.script_filename, 'r') as fp:
            completed = _run_strict('bsub', stdin=fp)
        jobid_match = re.search(r'^Job <(?P<jobid>\S+)> is submitted',
                                completed.stdout)
        if not jobid_match:
            raise JobSchedulerError('could not retrieve the job id '
                                    'of the submitted job')
        job._jobid = jobid_match.group('jobid')
        job._submit_time = time.time()

    def poll(self, *jobs):
        if jobs:
            # filter out non-jobs
            jobs = [job for job in jobs if job is not None]

        if not jobs:
            return

        completed = _run_strict(
            f'bjobs -noheader {" ".join(job.jobid for job in jobs)}'
        )
        job_status = {}
        job_status_lines = completed.stdout.split('\n')

        for line in job_status_lines:
            job_regex = (r'(?P<jobid>\d+)\s+'
                         r'(?P<user>\S+)\s+'
                         r'(?P<status>\S+)\s+'
                         r'(?P<queue>\S+)')
            job_match = re.search(job_regex, line)
            if job_match:
                job_status[job_match['jobid']] = job_match['status']

        for job in jobs:
            if job.jobid not in job_status:
                # job id not found
                self.log(f'Job {job.jobid} not known to scheduler, '
                         f'assuming job completed')
                job._state = 'COMPLETED'
                job._completed = True
            elif job_status[job.jobid] in ('DONE', 'EXIT'):
                # job done
                job._state = 'COMPLETED'
                job._completed = True
            elif job_status[job.jobid] == 'RUN':
                # job running
                job._state = 'RUNNING'
            elif job_status[job.jobid] == 'PEND':
                # job pending
                job._state = 'PENDING'
            elif job_status[job.jobid] in ['PSUSP', 'SSUSP', 'USUSP']:
                # job suspended
                job._state = 'SUSPENDED'
            else:
                # job status unknown
                self.log(f'Job {job_status[job.jobid]} not known, '
                         f'assuming job completed')
                job._state = 'COMPLETED'
                job._completed = True

    def finished(self, job):
        if job.exception:
            raise job.exception

        return job.state == 'COMPLETED'
