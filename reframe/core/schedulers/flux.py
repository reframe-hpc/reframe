# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Flux-Framework backend
#
# - Initial version submitted by Vanessa Sochat, Lawrence Livermore National Lab
#

import os
import time

import reframe.core.runtime as rt
from reframe.core.backends import register_scheduler
from reframe.core.exceptions import JobSchedulerError, JobError
from reframe.core.schedulers.pbs import PbsJobScheduler


# Just import flux once
try:
    import flux
    import flux.job
    from flux.job import JobspecV1
except ImportError:
    flux = None

waiting_states = ["QUEUED", "HELD", "WAITING", "PENDING"]


@register_scheduler("flux")
class FluxJobScheduler(PbsJobScheduler):
    def __init__(self):
        if not flux:
            raise JobSchedulerError(
                "Cannot import flux. Is a cluster available to you with Python bindings?"
            )
        self._fexecutor = flux.job.FluxExecutor()
        self._submit_timeout = rt.runtime().get_option(
            f"schedulers/@{self.registered_name}/job_submit_timeout"
        )

    def emit_preamble(self, job):
        # We don't need to submit with a file, so we don't need a preamble.
        return []

    def submit(self, job):
        """
        Submit a job to the flux executor.
        """
        # Output and error files
        script_prefix = job.script_filename.split(".")[0]
        output = os.path.join(job.workdir, f"{script_prefix}.out")
        error = os.path.join(job.workdir, f"{script_prefix}.err")

        # Generate the flux job
        # Assume the filename includes a hashbang
        # flux does not support mem_mb, disk_mb
        fluxjob = JobspecV1.from_command(
            command=["/bin/bash", job.script_filename],
            num_tasks=job.num_tasks_per_core or 1,
            cores_per_task=job.num_cpus_per_task or 1,
        )

        # A duration of zero (the default) means unlimited
        fluxjob.duration = job.time_limit or 0
        fluxjob.stdout = output
        fluxjob.stderr = error

        # This doesn't seem to be used?
        fluxjob.cwd = job.workdir
        fluxjob.environment = dict(os.environ)
        flux_future = self._fexecutor.submit(fluxjob)
        job._jobid = str(flux_future.jobid())
        job._submit_time = time.time()
        job._flux_future = flux_future

    def cancel(self, job):
        # Job cannot cancel once running or completed
        if not job._flux_future.cancel():
            # This will raise JobException with event=cancel (on poll)
            flux.job.cancel(flux.Flux(), job._flux_future.jobid())
        job._is_cancelling = True

    def poll(self, *jobs):
        if jobs:
            # filter out non-jobs
            jobs = [job for job in jobs if job is not None]

        if not jobs:
            return

        # Loop through active jobs and act on status
        for job in jobs:

            if job._flux_future.done():
                # The exit code can help us determine if the job was successful
                try:
                    exit_code = job._flux_future.result(0)

                # Currently the only state we see is cancelled here
                except flux.job.JobException:
                    self.log(f"Job {job.jobid} was likely cancelled.")
                    job._state = "CANCELLED"
                    job._cancelled = True

                except RuntimeError:
                    # Assume some runtime issue (suspended)
                    self.log(f"Job {job.jobid} was likely suspended.")
                    job._state = "SUSPENDED"
                else:
                    # the job finished (but possibly with nonzero exit code)
                    if exit_code != 0:
                        self.log(f"Job {job.jobid} did not finish successfully")
                    job._state = "COMPLETED"
                job._completed = True

            elif job.state in waiting_states and job.max_pending_time:
                if time.time() - job.submit_time >= job.max_pending_time:
                    self.cancel(job)
                    job._exception = JobError(
                        "maximum pending time exceeded", job.jobid
                    )

            # Otherwise, we are still running
            else:
                job._state = "RUNNING"

    def finished(self, job):
        if job.exception:
            raise job.exception
        return job.state in ["COMPLETED", "CANCELLED", "SUSPENDED"]
