# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# OAR backend
#
# - Initial version submitted by Mahendra Paipuri, INRIA
#

import functools
import os
import re
import time

import reframe.utility.osext as osext
from reframe.core.backends import register_scheduler
from reframe.core.exceptions import JobError, JobSchedulerError
from reframe.core.schedulers.pbs import PbsJobScheduler
from reframe.utility import seconds_to_hms


# States can be found here:
# https://github.com/oar-team/oar/blob/0fccc4fc3bb86ee935ce58effc5aec514a3e155d/sources/core/qfunctions/oarstat#L293
def oar_state_completed(state):
    completion_states = {
        'Error',
        'Terminated',
    }
    if state:
        return all(s in completion_states for s in state.split(','))

    return False


def oar_state_pending(state):
    pending_states = {
        'Waiting',
        'toLaunch',
        'Launching',
        'Hold',
        'Running',
        'toError',
        'Finishing',
        'Suspended',
        'Resuming',
    }
    if state:
        return any(s in pending_states for s in state.split(','))

    return False


_run_strict = functools.partial(osext.run_command, check=True)


@register_scheduler('oar')
class OarJobScheduler(PbsJobScheduler):
    def __init__(self):
        self._prefix = '#OAR'
        self._submit_timeout = self.get_option('job_submit_timeout')

    def emit_preamble(self, job):
        # host is de-facto nodes and core is number of cores requested per node
        # number of sockets can also be specified using cpu={num_sockets}
        tasks_opt = '-l /host={num_nodes}/core={num_tasks_per_node}'

        # Same reason as oarsub, we give full path to output and error files to
        # avoid writing them in the working dir
        preamble = [
            self._format_option(f'-n "{job.name}"'),
            self._format_option(f'-O {job.stdout}'),
            self._format_option(f'-E {job.stderr}'),
        ]

        if job.time_limit is not None:
            h, m, s = seconds_to_hms(job.time_limit)
            tasks_opt += ',walltime=%d:%d:%d' % (h, m, s)

        # Get number of nodes in the reservation
        num_tasks_per_node = job.num_tasks_per_node or 1
        num_nodes = job.num_tasks // num_tasks_per_node

        # Emit main resource reservation option
        options = [tasks_opt.format(
            num_nodes=num_nodes, num_tasks_per_node=num_tasks_per_node,
        )]

        # Emit the rest of the options
        options += job.sched_access + job.options + job.cli_options
        for opt in options:
            if opt.startswith('#'):
                preamble.append(opt)
            else:
                preamble.append(self._format_option(opt))

        return preamble

    def submit(self, job):
        # OAR batch submission mode needs full path to the job script
        job_script_fullpath = os.path.join(job.workdir, job.script_filename)

        # OAR needs -S to submit job in batch mode
        cmd = f'oarsub -S {job_script_fullpath}'
        completed = _run_strict(cmd, timeout=self._submit_timeout)
        jobid_match = re.search(r'.*OAR_JOB_ID=(?P<jobid>\S+)',
                                completed.stdout)
        if not jobid_match:
            raise JobSchedulerError('could not retrieve the job id '
                                    'of the submitted job')

        job._jobid = jobid_match.group('jobid')
        job._submit_time = time.time()

    def cancel(self, job):
        _run_strict(f'oardel {job.jobid}', timeout=self._submit_timeout)
        job._cancelled = True

    def poll(self, *jobs):
        if jobs:
            # Filter out non-jobs
            jobs = [job for job in jobs if job is not None]

        if not jobs:
            return

        for job in jobs:
            completed = _run_strict(
                f'oarstat -fj {job.jobid}'
            )

            # Store information for each job separately
            jobinfo = {}

            # Typical oarstat -fj <job_id> output:
            # https://github.com/oar-team/oar/blob/0fccc4fc3bb86ee935ce58effc5aec514a3e155d/sources/core/qfunctions/oarstat#L310
            job_raw_info = completed.stdout
            jobid_match = re.search(
                r'^Job_Id:\s*(?P<jobid>\S+)', completed.stdout, re.MULTILINE
            )
            if jobid_match:
                jobid = jobid_match.group('jobid')
                jobinfo[jobid] = job_raw_info

            if job.jobid not in jobinfo:
                self.log(f'Job {job.jobid} not known to scheduler, '
                         f'assuming job completed')
                job._state = 'Terminated'
                job._completed = True
                continue

            info = jobinfo[job.jobid]
            state_match = re.search(
                r'^\s*state = (?P<state>[A-Z]\S+)', info, re.MULTILINE
            )
            if not state_match:
                self.log(f'Job state not found (job info follows):\n{info}')
                continue

            job._state = state_match.group('state')
            if oar_state_completed(job.state):
                exitcode_match = re.search(
                    r'^\s*exit_code = (?P<code>\d+)',
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
            elif oar_state_pending(job.state) and job.max_pending_time:
                if time.time() - job.submit_time >= job.max_pending_time:
                    self.cancel(job)
                    job._exception = JobError('maximum pending time exceeded',
                                              job.jobid)
