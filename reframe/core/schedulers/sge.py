# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# SGE backend
#
# - Initial version submitted by Mosè Giordano, UCL (based on the PBS backend)
#

import functools
import re
import time
import xml.etree.ElementTree as ET

import reframe.utility.osext as osext
from reframe.core.backends import register_scheduler
from reframe.core.exceptions import JobSchedulerError
from reframe.core.schedulers.pbs import PbsJobScheduler
from reframe.utility import seconds_to_hms

_run_strict = functools.partial(osext.run_command, check=True)


@register_scheduler('sge')
class SgeJobScheduler(PbsJobScheduler):
    def __init__(self):
        self._prefix = '#$'
        self._submit_timeout = self.get_option('job_submit_timeout')

    def emit_preamble(self, job):
        preamble = [
            self._format_option(f'-N "{job.name}"'),
            self._format_option(f'-o {job.stdout}'),
            self._format_option(f'-e {job.stderr}'),
            self._format_option(f'-wd {job.workdir}')
        ]

        if job.time_limit is not None:
            h, m, s = seconds_to_hms(job.time_limit)
            preamble.append(
                self._format_option(f'-l h_rt=%d:%d:%d' % (h, m, s))
            )

        # Emit the rest of the options
        options = job.options + job.cli_options
        for opt in options:
            if opt.startswith('#'):
                preamble.append(opt)
            else:
                preamble.append(self._format_option(opt))

        return preamble

    def submit(self, job):
        # `-o` and `-e` options are only recognized in command line by the PBS,
        # SGE, and Slurm wrappers.
        cmd = f'qsub -o {job.stdout} -e {job.stderr} {job.script_filename}'
        completed = _run_strict(cmd, timeout=self._submit_timeout)
        jobid_match = re.search(r'^Your job (?P<jobid>\S+)', completed.stdout)
        if not jobid_match:
            raise JobSchedulerError('could not retrieve the job id '
                                    'of the submitted job')

        job._jobid = jobid_match.group('jobid')
        job._submit_time = time.time()

    def poll(self, *jobs):
        if jobs:
            # Filter out non-jobs
            jobs = [job for job in jobs if job is not None]

        if not jobs:
            return

        user = osext.osuser()
        completed = osext.run_command(f'qstat -xml -u {user}')
        if completed.returncode != 0:
            raise JobSchedulerError(
                f'qstat failed with exit code {completed.returncode} '
                f'(standard error follows):\n{completed.stderr}'
            )

        # Index the jobs to poll on their jobid
        jobs_to_poll = {job.jobid: job for job in jobs}

        # Parse the XML
        root = ET.fromstring(completed.stdout)

        # We are iterating over the returned XML and update the status of the
        # jobs relevant to ReFrame; the naming convention of variables matches
        # that of SGE's XML output

        known_jobs = set()  # jobs known to the SGE scheduler
        for queue_info in root:
            # Reads the XML and prints jobs with status belonging to user.
            if queue_info is None:
                raise JobSchedulerError('could not retrieve queue information')

            for job_list in queue_info:
                if job_list.find("JB_owner").text != user:
                    # Not a job of this user.
                    continue

                jobid = job_list.find("JB_job_number").text
                if jobid not in jobs_to_poll:
                    # Not a reframe job
                    continue

                state = job_list.find("state").text
                job = jobs_to_poll[jobid]
                known_jobs.add(job)

                # For the list of known statuses see `man 5 sge_status`
                # (https://arc.liv.ac.uk/SGE/htmlman/htmlman5/sge_status.html)
                if state in ['r', 'hr', 't', 'Rr', 'Rt']:
                    job._state = 'RUNNING'
                elif state in ['qw', 'Rq', 'hqw', 'hRwq']:
                    job._state = 'PENDING'
                elif state in ['s', 'ts', 'S', 'tS', 'T', 'tT', 'Rs',
                               'Rts', 'RS', 'RtS', 'RT', 'RtT']:
                    job._state = 'SUSPENDED'
                elif state in ['Eqw', 'Ehqw', 'EhRqw']:
                    job._state = 'ERROR'
                elif state in ['dr', 'dt', 'dRr', 'dRt', 'ds',
                               'dS', 'dT', 'dRs', 'dRS', 'dRT']:
                    job._state = 'DELETING'
                elif state == 'z':
                    job._state = 'COMPLETED'

        # Mark any "unknown" job as completed
        unknown_jobs = set(jobs) - known_jobs
        for job in unknown_jobs:
            self.log(f'Job {job.jobid} not known to scheduler, '
                     f'assuming job completed')
            job._state = 'COMPLETED'

    def finished(self, job):
        if job.exception:
            raise job.exception

        return job.state == 'COMPLETED'
