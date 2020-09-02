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

    def poll(self, *jobs):
        '''Update the status of the jobs.'''
        jobids = [str(job.jobid) for job in jobs]
        if not jobids:
            return

        completed = os_ext.run_command(f"qstat -f {' '.join(jobids)}")

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
                job.state = 'COMPLETED'
            return

        if completed.returncode != 0:
            raise JobError(f'qstat failed: {completed.stderr}')

        jobs_info = {}
        for job_raw_info in completed.stdout.split('\n\n'):
            jobid_match = re.search(
                r'^Job Id:\s*(?P<jobid>\S+)', job_raw_info, re.MULTILINE
            )
            if jobid_match:
                jobid = int(jobid_match.group('jobid'))
                jobs_info[jobid] = job_raw_info

        for job in jobs:
            if job.jobid not in jobs_info:
                getlogger().debug(
                    f'jobid {job.jobid} not known by scheduler, '
                    'assuming job completed'
                )
                job.state = 'COMPLETED'
                continue

            stdout = jobs_info[job.jobid]
            nodelist_match = re.search(
                r'exec_host = (?P<nodespec>[\S\t\n]+)',
                completed.stdout,
                re.MULTILINE
            )
            if nodelist_match:
                nodespec = nodelist_match.group('nodespec')
                nodespec = re.sub(r'[\n\t]*', '', nodespec)
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
                exitcode_match = re.search(
                    r'^\s*exit_status = (?P<code>\d+)',
                    stdout,
                    re.MULTILINE,
                )
                if not exitcode_match:
                    continue

                job.exitcode = int(exitcode_match.group('code'))

        for job in jobs:
            stdout = os.path.join(job.workdir, job.stdout)
            stderr = os.path.join(job.workdir, job.stderr)
            output_ready = os.path.exists(stdout) and os.path.exists(stderr)
            done = job.jobid in self._cancelled or output_ready
            if job.state == 'COMPLETED' and done:
                self._finished_jobs.add(job)

    def finished(self, job):
        if job.exception:
            try:
                raise job.exception
            except JobError as e:
                # We ignore these exceptions at this point and we simply mark
                # the job as unfinished.
                getlogger().debug('ignoring error during polling: %s' % e)
                return False
            finally:
                job.exception = None

        if job.max_pending_time and job.state in ['QUEUED',
                                                  'HELD',
                                                  'WAITING']:
            if (datetime.now() - self._job_submit_time[job] >=
                job.max_pending_time):
                self.cancel(job)
                raise JobError('maximum pending time exceeded',
                               jobid=job.jobid)

        getlogger().debug(f"finished: {job in self._finished_jobs}")
        return job in self._finished_jobs
