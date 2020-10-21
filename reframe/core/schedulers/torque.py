# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Torque backend
#
# - Initial version submitted by Samuel Moors, Vrije Universiteit Brussel (VUB)
#

import re
import os
import time

import reframe.utility.osext as osext
from reframe.core.backends import register_scheduler
from reframe.core.exceptions import JobError, JobSchedulerError
from reframe.core.logging import getlogger
from reframe.core.schedulers.pbs import PbsJobScheduler, _run_strict


JOB_STATES = {
    'Q': 'QUEUED',
    'H': 'HELD',
    'R': 'RUNNING',
    'E': 'EXITING',
    'T': 'MOVED',
    'W': 'WAITING',
    'S': 'SUSPENDED',
    'C': 'COMPLETED',
}


@register_scheduler('torque')
class TorqueJobScheduler(PbsJobScheduler):
    TASKS_OPT = '-l nodes={num_nodes}:ppn={num_cpus_per_node}'

    def _update_nodelist(self, job, nodespec):
        if job.nodelist is not None:
            return

        job._nodelist = [x.split('/')[0] for x in nodespec.split('+')]
        job._nodelist.sort()

    def poll(self, *jobs):
        if jobs:
            # Filter out non-jobs
            jobs = [job for job in jobs if job is not None]

        if not jobs:
            return

        completed = osext.run_command(
            f'qstat -f {" ".join(job.jobid for job in jobs)}'
        )

        # Depending on the configuration, completed jobs will remain on the job
        # list for a limited time, or be removed upon completion.
        # If qstat cannot find any of the job IDs, it will return 153.
        # Otherwise, it will return with return code 0 and print information
        # only for the jobs it could find.
        if completed.returncode == 153:
            getlogger().debug(
                'return code = 153: jobids not known by scheduler, '
                'assuming all jobs completed'
            )
            for job in jobs:
                job._state = 'COMPLETED'

            return

        if completed.returncode != 0:
            raise JobSchedulerError(
                f'qstat failed with exit code {completed.returncode} '
                f'(standard error follows):\n{completed.stderr}'
            )

        # Store information for each job separately
        jobinfo = {}
        for job_raw_info in completed.stdout.split('\n\n'):
            jobid_match = re.search(
                r'^Job Id:\s*(?P<jobid>\S+)', job_raw_info, re.MULTILINE
            )
            if jobid_match:
                jobid = jobid_match.group('jobid')
                jobinfo[jobid] = job_raw_info

        for job in jobs:
            if job.jobid not in jobinfo:
                getlogger().debug(
                    f'jobid {job.jobid} not known to scheduler, '
                    f'assuming job completed'
                )
                job._state = 'COMPLETED'
                job._completed = True
                continue

            info = jobinfo[job.jobid]
            state_match = re.search(
                r'^\s*job_state = (?P<state>[A-Z])', info, re.MULTILINE
            )
            if not state_match:
                getlogger().debug(
                    f'job state not found (job info follows):\n{info}'
                )
                continue

            state = state_match.group('state')
            job._state = JOB_STATES[state]
            nodelist_match = re.search(
                r'exec_host = (?P<nodespec>[\S\t\n]+)',
                info, re.MULTILINE
            )
            if nodelist_match:
                nodespec = nodelist_match.group('nodespec')
                nodespec = re.sub(r'[\n\t]*', '', nodespec)
                self._update_nodelist(job, nodespec)

            if job.state == 'COMPLETED':
                exitcode_match = re.search(
                    r'^\s*exit_status = (?P<code>\d+)',
                    info, re.MULTILINE,
                )
                if exitcode_match:
                    job._exitcode = int(exitcode_match.group('code'))

                # We report a job as finished only when its stdout/stderr are
                # written back to the working directory
                stdout = os.path.join(job.workdir, job.stdout)
                stderr = os.path.join(job.workdir, job.stderr)
                out_ready = os.path.exists(stdout) and os.path.exists(stderr)
                done = job.cancelled or out_ready
                if done:
                    job._completed = True
            elif (job.state in ['QUEUED', 'HELD', 'WAITING'] and
                  job.max_pending_time):
                if (time.time() - job.submit_time >= job.max_pending_time):
                    self.cancel(job)
                    job._exception = JobError('maximum pending time exceeded')
