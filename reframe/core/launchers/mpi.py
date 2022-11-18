# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import semver
import re

import reframe.utility.osext as osext
from reframe.core.backends import register_launcher
from reframe.core.launchers import JobLauncher
from reframe.core.logging import getlogger
from reframe.utility import seconds_to_hms


@register_launcher('srun')
class SrunLauncher(JobLauncher):
    def __init__(self):
        self.options = []
        self.use_cpus_per_task = True
        try:
            out = osext.run_command('srun --version')
            match = re.search('slurm (\d+)\.(\d+)\.(\d+)', out.stdout)
            if match:
                # We cannot pass to semver strings like 22.05.1 directly
                # because it is not a valid version string for semver. We
                # need to remove all the leading zeros.
                slurm_version = (
                    semver.VersionInfo(
                        match.group(1), match.group(2), match.group(3)
                    )
                )
                if slurm_version < semver.VersionInfo(22, 5, 0):
                    self.use_cpus_per_task = False
            else:
                getlogger().warning(
                    'could not get version of Slurm, --cpus-per-task will be '
                    'set according to the num_cpus_per_task attribute'
                )
        except Exception:
            getlogger().warning(
                'could not get version of Slurm, --cpus-per-task will be set '
                'according to the num_cpus_per_task attribute'
            )

    def command(self, job):
        ret = ['srun']
        if self.use_cpus_per_task and job.num_cpus_per_task:
            ret.append(f'--cpus-per-task={job.num_cpus_per_task}')

        return ret


@register_launcher('ibrun')
class IbrunLauncher(JobLauncher):
    '''TACC's custom parallel job launcher.'''

    def command(self, job):
        return ['ibrun']


@register_launcher('upcrun')
class UpcrunLauncher(JobLauncher):
    '''Launcher for UPC applications.'''

    def command(self, job):
        cmd = ['upcrun']
        if job.num_tasks_per_node:
            num_nodes = job.num_tasks // job.num_tasks_per_node
            cmd += ['-N', str(num_nodes)]

        cmd += ['-n', str(job.num_tasks)]
        return cmd


@register_launcher('upcxx-run')
class UpcxxrunLauncher(JobLauncher):
    '''Launcher for UPC++ applications.'''

    def command(self, job):
        cmd = ['upcxx-run']
        if job.num_tasks_per_node:
            num_nodes = job.num_tasks // job.num_tasks_per_node
            cmd += ['-N', str(num_nodes)]

        cmd += ['-n', str(job.num_tasks)]
        return cmd


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
            h, m, s = seconds_to_hms(job.time_limit)
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

        if job.exclusive_access:
            ret += ['--exclusive']

        if job.use_smt is not None:
            hint = 'multithread' if job.use_smt else 'nomultithread'
            ret += ['--hint=%s' % hint]

        for opt in job.options + job.cli_options:
            if opt.startswith('#'):
                continue

            ret.append(opt)

        return ret


@register_launcher('lrun')
class LrunLauncher(JobLauncher):
    '''LLNL's custom parallel job launcher'''

    def command(self, job):
        num_tasks_per_node = job.num_tasks_per_node or 1
        num_nodes = job.num_tasks // num_tasks_per_node
        return ['lrun', '-N', str(num_nodes),
                '-T', str(num_tasks_per_node)]


@register_launcher('lrun-gpu')
class LrungpuLauncher(LrunLauncher):
    '''LLNL's custom parallel job launcher w/ CUDA aware Spectum MPI'''

    def command(self, job):
        return super().command(job) + ['-M "-gpu"']
