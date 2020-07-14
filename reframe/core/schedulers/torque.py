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
from datetime import datetime

import reframe.utility.os_ext as os_ext
from reframe.core.backends import register_scheduler
from reframe.core.exceptions import JobError
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

    def _set_nodelist(self, job, nodespec):
        if job.nodelist is not None:
            return

        job.nodelist = [x.split('/')[0] for x in nodespec.split('+')]
        job.nodelist.sort()

    def poll_jobs(self, jobs):
        '''Update the status of the jobs.'''

        if not jobs:
            return

        jobids = [str(job.jobid) for job in jobs]
        completed = os_ext.run_command(f"qstat -f {' '.join(jobids)}")

        # Depending on the configuration, completed jobs will remain on the job
        # list for a limited time, or be removed upon completion.
        # If qstat cannot find the jobid, it returns code 153.
        if completed.returncode == 153:
            getlogger().debug(
                'jobids not known by scheduler, assuming all jobs completed'
            )
            for job in jobs:
                job.state = 'COMPLETED'
            return

        if completed.returncode != 0:
            for job in jobs:
                job.exception = JobError('qstat failed: %s' % completed.stderr, jobids)

            return

        jobs_stdout = {}
        for jobout in completed.stdout.split('\n\n'):
            jobid_match = re.search(
                r'^Job Id: (?P<jobid>\S+)', completed.stdout, re.MULTILINE
            )
            if jobid_match:
                jobs_stdout[jobid_match.group('jobid')] = jobout

        for job in jobs:
            if job.jobid not in jobs_stdout:
                getlogger().debug(
                    'jobid not known by scheduler, assuming job completed'
                )
                job.state = 'COMPLETED'
                continue

            stdout = jobs_stdout[job.jobid]
            nodelist_match = re.search(
                r'^\s*exec_host = (?P<nodespec>\S+)', stdout, re.MULTILINE
            )
            if nodelist_match:
                nodespec = nodelist_match.group('nodespec')
                self._set_nodelist(job, nodespec)

            state_match = re.search(
                r'^\s*job_state = (?P<state>[A-Z])', stdout, re.MULTILINE
            )
            if not state_match:
                getlogger().debug(
                    'job state not found (stdout follows)\n%s' % stdout
                )
                continue

            state = state_match.group('state')
            job.state = JOB_STATES[state]
            if job.state == 'COMPLETED':
                code_match = re.search(
                    r'^\s*exit_status = (?P<code>\d+)',
                    stdout,
                    re.MULTILINE,
                )
                if not code_match:
                    continue

                job.exitcode = int(code_match.group('code'))

    def finished(self, job):
        if job.exception:
            try:
                raise job.exception
            except JobError as e:
                # We ignore these exceptions at this point and we simply mark the
                # job as unfinished.
                getlogger().debug('ignoring error during polling: %s' % e)
                return False
            finally:
                job.exception = None

        if job.max_pending_time and job.state in ['QUEUED',
                                                  'HELD',
                                                  'WAITING']:
            if datetime.now() - job._submit_time >= job.max_pending_time:
                self.cancel(job)
                raise JobError('maximum pending time exceeded',
                                jobid=job.jobid)

        stdout = os.path.join(job.workdir, job.stdout)
        stderr = os.path.join(job.workdir, job.stderr)
        output_ready = os.path.exists(stdout) and os.path.exists(stderr)
        done = job.jobid in self._cancelled or output_ready
        return job.state == 'COMPLETED' and done
