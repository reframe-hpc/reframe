# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

from reframe.core.launchers import JobLauncher
from reframe.core.launchers.registry import register_launcher
from reframe.utility import seconds_to_hms


@register_launcher('srun')
class SrunLauncher(JobLauncher):
    def command(self, job):
        return ['srun']


@register_launcher('ibrun')
class IbrunLauncher(JobLauncher):
    '''TACC's custom parallel job launcher.'''

    def command(self, job):
        return ['ibrun']


@register_launcher('alps')
class AlpsLauncher(JobLauncher):
    def command(self, job):
        cmd = ['aprun', '-n', str(job.num_tasks)]
        if job.num_tasks_per_node:
            cmd += ['-N', str(job.num_tasks_per_node)]

        if job.num_cpus_per_task:
            cmd += ['-d', str(job.num_cpus_per_task)]

        if job.use_smt:
            cmd += ['-j', '0']

        return cmd


@register_launcher('mpirun')
class MpirunLauncher(JobLauncher):
    def command(self, job):
        return ['mpirun', '-np', str(job.num_tasks)]


@register_launcher('mpiexec')
class MpiexecLauncher(JobLauncher):
    def command(self, job):
        return ['mpiexec', '-n', str(job.num_tasks)]


@register_launcher('srunalloc')
class SrunAllocationLauncher(JobLauncher):
    def command(self, job):
        ret = ['srun']
        if job.name:
            ret += ['--job-name=%s' % job.name]

        if job.time_limit:
            h, m, s = seconds_to_hms(job.time_limit.total_seconds())
            ret += ['--time=%d:%d:%d' % (h, m, s)]

        if job.stdout:
            ret += ['--output=%s' % job.stdout]

        if job.stderr:
            ret += ['--error=%s' % job.stderr]

        if job.num_tasks:
            ret += ['--ntasks=%s' % str(job.num_tasks)]

        if job.num_tasks_per_node:
            ret += ['--ntasks-per-node=%s' % str(job.num_tasks_per_node)]

        if job.num_tasks_per_core:
            ret += ['--ntasks-per-core=%s' % str(job.num_tasks_per_core)]

        if job.num_tasks_per_socket:
            ret += ['--ntasks-per-socket=%s' % str(job.num_tasks_per_socket)]

        if job.num_cpus_per_task:
            ret += ['--cpus-per-task=%s' % str(job.num_cpus_per_task)]

        if job.sched_partition:
            ret += ['--partition=%s' % str(job.sched_partition)]

        if job.sched_exclusive_access:
            ret += ['--exclusive']

        if job.use_smt is not None:
            hint = 'multithread' if job.use_smt else 'nomultithread'
            ret += ['--hint=%s' % hint]

        if job.sched_partition:
            ret += ['--partition=%s' % str(job.sched_partition)]

        if job.sched_account:
            ret += ['--account=%s' % str(job.sched_account)]

        if job.sched_nodelist:
            ret += ['--nodelist=%s' % str(job.sched_nodelist)]

        if job.sched_exclude_nodelist:
            ret += ['--exclude=%s' % str(job.sched_exclude_nodelist)]

        for opt in job.options:
            if opt.startswith('#'):
                continue

            ret.append(opt)

        return ret
