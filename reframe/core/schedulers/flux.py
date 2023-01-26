# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Flux-Framework backend
#
# - Initial version submitted by Vanessa Sochat,
#   Lawrence Livermore National Lab
#

import itertools
import os
import time

from reframe.core.backends import register_scheduler
from reframe.core.exceptions import JobError
from reframe.core.schedulers import JobScheduler, Job

# Just import flux once
try:
    import flux
    import flux.job
    from flux.job import JobspecV1
except ImportError:
    error = 'no flux Python bindings found'
else:
    error = None

WAITING_STATES = ('QUEUED', 'HELD', 'WAITING', 'PENDING')


class _FluxJob(Job):
    def __init__(self, *args, **kwargs):
        '''Create the flux job (and future) to watch.'''
        super().__init__(*args, **kwargs)

        # Generate the flux job
        self.fluxjob = JobspecV1.from_command(
            command=['/bin/bash', self.script_filename],
            num_tasks=self.num_tasks_per_core or 1,
            cores_per_task=self.num_cpus_per_task or 1,
        )

        # We must use absolute paths for Flux
        out = os.path.join(os.path.abspath(self.workdir), self.stdout)
        err = os.path.join(os.path.abspath(self.workdir), self.stderr)

        # A duration of zero (the default) means unlimited
        self.fluxjob.duration = self.time_limit or 0
        self.fluxjob.stdout = out
        self.fluxjob.stderr = err
        self.fluxjob.cwd = os.path.abspath(self.workdir)
        self.fluxjob.environment = dict(os.environ)
        self._completed = False

    @property
    def completed(self):
        return self._completed


@register_scheduler('flux', error=error)
class FluxJobScheduler(JobScheduler):
    def __init__(self):
        self._fexecutor = flux.job.FluxExecutor()
        self._submit_timeout = self.get_option('job_submit_timeout')

    def emit_preamble(self, job):
        # We don't need to submit with a file, so we don't need a preamble.
        return []

    def make_job(self, *args, **kwargs):
        return _FluxJob(*args, **kwargs)

    def submit(self, job):
        '''Submit a job to the flux executor.'''

        flux_future = self._fexecutor.submit(job.fluxjob)
        job._jobid = str(flux_future.jobid())
        job._submit_time = time.time()
        job._flux_future = flux_future

    def cancel(self, job):
        '''Cancel a running Flux job.'''

        # Job future cannot cancel once running or completed
        if not job._flux_future.cancel():
            # This will raise JobException with event=cancel (on poll)
            flux.job.cancel(flux.Flux(), job._flux_future.jobid())

    def poll(self, *jobs):
        '''Poll running Flux jobs for updated states.'''

        if jobs:
            # filter out non-jobs
            jobs = [job for job in jobs if job is not None]

        if not jobs:
            return

        # Loop through active jobs and act on status
        for job in jobs:
            if job._flux_future.done():
                try:
                    # The exit code can help us determine if the job was
                    # successful
                    exit_code = job._flux_future.result(0)
                except flux.job.JobException:
                    # Currently the only state we see is cancelled here
                    self.log(f'Job {job.jobid} was likely cancelled.')
                    job._state = 'CANCELLED'
                except RuntimeError:
                    # Assume some runtime issue (suspended)
                    self.log(f'Job {job.jobid} was likely suspended.')
                    job._state = 'SUSPENDED'
                else:
                    # the job finished (but possibly with nonzero exit code)
                    job._state = 'COMPLETED'
                    if exit_code != 0:
                        self.log(
                            f'Job {job.jobid} did not finish successfully'
                        )

                job._completed = True
            elif job.state in WAITING_STATES and job.max_pending_time:
                if time.time() - job.submit_time >= job.max_pending_time:
                    self.cancel(job)
                    job._exception = JobError(
                        'maximum pending time exceeded', job.jobid
                    )
            else:
                # Otherwise, we are still running
                job._state = 'RUNNING'

    def allnodes(self):
        raise NotImplementedError('flux backend does not support node listing')

    def filternodes(self, job, nodes):
        raise NotImplementedError(
            'flux backend does not support node filtering'
        )

    def wait(self, job):
        '''Wait until a job is finished.'''

        intervals = itertools.cycle([1, 2, 3])
        while not self.finished(job):
            self.poll(job)
            time.sleep(next(intervals))

    def finished(self, job):
        if job.exception:
            raise job.exception

        return job.completed
