# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
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

import reframe.core.runtime as rt
import reframe.core.schedulers as sched
import reframe.utility.osext as osext
from reframe.core.backends import register_scheduler
from reframe.core.exceptions import JobError, JobSchedulerError
from reframe.utility import seconds_to_hms, toalphanum


# Time to wait after a job is finished for its standard output/error to be
# written to the corresponding files.
# FIXME: Consider making this a configuration parameter
PBS_OUTPUT_WRITEBACK_WAIT = 3


# Minimum amount of time between its submission and its cancellation. If you
# immediately cancel a PBS job after submission, its output files may never
# appear in the output causing the wait() to hang.
# FIXME: Consider making this a configuration parameter
PBS_CANCEL_DELAY = 3


_run_strict = functools.partial(osext.run_command, check=True)


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


class _PbsJob(sched.Job):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cancelled = False

        # This is set by the scheduler when both the job's state is
        # 'COMPLETED' and the job's stdout and stderr are written back
        self._completed = False

    @property
    def cancelled(self):
        return self._cancelled

    @property
    def completed(self):
        return self._completed


@register_scheduler('pbs')
class PbsJobScheduler(sched.JobScheduler):
    TASKS_OPT = ('-l select={num_nodes}:mpiprocs={num_tasks_per_node}'
                 ':ncpus={num_cpus_per_node}')

    def __init__(self):
        self._prefix = '#PBS'
        self._submit_timeout = rt.runtime().get_option(
            f'schedulers/@{self.registered_name}/job_submit_timeout'
        )

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
        for opt in (*job.sched_access, *job.options, *job.cli_options):
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

    def make_job(self, *args, **kwargs):
        return _PbsJob(*args, **kwargs)

    def emit_preamble(self, job):
        # The job name is a string of up to 15 alphanumeric characters
        # where the first character is alphabetic
        job_name = toalphanum(job.name)[:15]
        preamble = [
            self._format_option(f'-N {job_name}'),
            self._format_option(f'-o {job.stdout}'),
            self._format_option(f'-e {job.stderr}'),
        ]

        if job.time_limit is not None:
            h, m, s = seconds_to_hms(job.time_limit)
            preamble.append(
                self._format_option('-l walltime=%d:%d:%d' % (h, m, s)))

        preamble += self._emit_lselect_option(job)

        # PBS starts the job in the home directory by default
        preamble.append(f'cd {job.workdir}')
        return preamble

    def allnodes(self):
        raise NotImplementedError('pbs backend does not support node listing')

    def filternodes(self, job, nodes):
        raise NotImplementedError('pbs backend does not support '
                                  'node filtering')

    def submit(self, job):
        # `-o` and `-e` options are only recognized in command line by the PBS
        # Slurm wrappers.
        cmd = f'qsub -o {job.stdout} -e {job.stderr} {job.script_filename}'
        completed = _run_strict(cmd, timeout=self._submit_timeout)
        jobid_match = re.search(r'^(?P<jobid>\S+)', completed.stdout)
        if not jobid_match:
            raise JobSchedulerError('could not retrieve the job id '
                                    'of the submitted job')

        job._jobid = jobid_match.group('jobid')
        job._submit_time = time.time()

    def wait(self, job):
        intervals = itertools.cycle([1, 2, 3])
        while not self.finished(job):
            self.poll(job)
            time.sleep(next(intervals))

    def cancel(self, job):
        time_from_submit = time.time() - job.submit_time
        if time_from_submit < PBS_CANCEL_DELAY:
            time.sleep(PBS_CANCEL_DELAY - time_from_submit)

        _run_strict(f'qdel {job.jobid}', timeout=self._submit_timeout)
        job._cancelled = True

    def finished(self, job):
        if job.exception:
            raise job.exception

        return job.completed

    def _update_nodelist(self, job, nodespec):
        if job.nodelist is not None:
            return

        job._nodelist = [x.split('/')[0] for x in nodespec.split('+')]
        job._nodelist.sort()

    def poll(self, *jobs):
        def output_ready(job):
            # We report a job as finished only when its stdout/stderr are
            # written back to the working directory
            stdout = os.path.join(job.workdir, job.stdout)
            stderr = os.path.join(job.workdir, job.stderr)
            return os.path.exists(stdout) and os.path.exists(stderr)

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
        if completed.returncode in (153, 35):
            self.log(f'Return code is {completed.returncode}')
            for job in jobs:
                job._state = 'COMPLETED'
                if job.cancelled or output_ready(job):
                    self.log(f'Assuming job {job.jobid} completed')
                    job._completed = True

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
                self.log(f'Job {job.jobid} not known to scheduler')
                job._state = 'COMPLETED'
                if job.cancelled or output_ready(job):
                    self.log(f'Assuming job {job.jobid} completed')
                    job._completed = True

                continue

            info = jobinfo[job.jobid]
            state_match = re.search(
                r'^\s*job_state = (?P<state>[A-Z])', info, re.MULTILINE
            )
            if not state_match:
                self.log(f'Job state not found (job info follows):\n{info}')
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
                done = job.cancelled or output_ready(job)
                if done:
                    job._completed = True
            elif (job.state in ['QUEUED', 'HELD', 'WAITING'] and
                  job.max_pending_time):
                if (time.time() - job.submit_time >= job.max_pending_time):
                    self.cancel(job)
                    job._exception = JobError('maximum pending time exceeded',
                                              job.jobid)


@register_scheduler('torque')
class TorqueJobScheduler(PbsJobScheduler):
    TASKS_OPT = '-l nodes={num_nodes}:ppn={num_cpus_per_node}'
