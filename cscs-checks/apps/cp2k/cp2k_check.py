# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
from hpctestlib.apps.cp2k import Cp2k


REFERENCE_CPU_PERFORMANCE_SMALL = {
    'dom:mc': {
        'maint': (202.2, None, 0.05, 's'),
        'prod': (202.2, None, 0.05, 's')
    },
    'daint:mc': {
        'maint': (180.9, None, 0.08, 's'),
        'prod': (180.9, None, 0.08, 's')
    },
    'eiger:mc': {
        'maint': (70.0, None, 0.08, 's'),
        'prod': (46.0, None, 0.05, 's')
    },
    'pilatus:mc': {
        'maint': (70.0, None, 0.08, 's'),
        'prod': (70.0, None, 0.08, 's')
    },
}

REFERENCE_CPU_PERFORMANCE_LARGE = {
    'daint:mc': {
        'maint': (141.0, None, 0.05, 's'),
        'prod': (113.0, None, 0.05, 's')
    },
    'eiger:mc': {
        'maint': (46.0, None, 0.05, 's'),
        'prod': (46.0, None, 0.05, 's')
    },
    'pilatus:mc': {
        'maint': (46.0, None, 0.05, 's'),
        'prod': (46.0, None, 0.05, 's')
    },
}

REFERENCE_GPU_PERFORMANCE_SMALL = {
    'dom:mc': {
        'maint': (251.8, None, 0.15, 's'),
        'prod': (240.0, None, 0.05, 's')
    },
    'daint:mc': {
        'maint': (241.3, None, 0.05, 's'),
        'prod': (241.3, None, 0.05, 's')
    }
}

REFERENCE_GPU_PERFORMANCE_LARGE = {
    'daint:mc': {
        'maint': (199.6, None, 0.06, 's'),
        'prod': (199.6, None, 0.06, 's')
    }
}


class Cp2kCheck(Cp2k):
    modules = ['CP2K']
    executable = 'cp2k.psmp'
    executable_opts = ['H2O-256.inp']
    maintainers = ['LM']
    tags = {'scs'}
    strict_check = False
    extra_resources = {
        'switches': {
            'num_switches': 1
        }
    }
    scale = parameter(['small', 'large'])
    benchmark = parameter(['prod', 'maint'])
    energy_value = -4404.2323
    energy_tolerance = 1E-04

    @run_after('init')
    def env_define(self):
        if self.current_system.name in ['eiger', 'pilatus']:
            self.valid_prog_environs = ['cpeGNU']
        else:
            self.valid_prog_environs = ['builtin']

    @run_after('setup')
    def set_generic_perf_references(self):
        self.reference.update({'*': {
            self.benchmark: (0, None, None, 'timesteps/s')
        }})

    @run_after('setup')
    def set_perf_patterns(self):
        self.perf_patterns = {
            self.benchmark: sn.extractsingle(
                r'^ CP2K(\s+[\d\.]+){4}\s+(?P<perf>\S+)',
                self.stdout, 'perf', float)
        }

    @run_after('init')
    def set_tags(self):
        self.tags |= {'maintenance' if self.benchmark == 'maint'
                      else 'production'}


@rfm.simple_test
class Cp2kCpuCheck(Cp2kCheck):
    valid_systems = ['daint:mc', 'eiger:mc', 'pilatus:mc']

    @run_after('init')
    def set_description(self):
        self.descr = (f'CP2K CPU check (version: {self.scale}, '
                      f'{self.benchmark})')

    @run_after('init')
    def set_reference(self):
        if self.scale == 'small':
            self.reference = REFERENCE_CPU_PERFORMANCE_SMALL
        else:
            self.reference = REFERENCE_CPU_PERFORMANCE_LARGE

    @run_after('init')
    def set_num_tasks(self):
        if self.scale == 'small':
            self.valid_systems += ['dom:mc']
            if self.current_system.name in ['daint', 'dom']:
                self.num_tasks = 216
                self.num_tasks_per_node = 36
            elif self.current_system.name in ['eiger', 'pilatus']:
                self.num_tasks = 96
                self.num_tasks_per_node = 16
                self.num_cpus_per_task = 16
                self.num_tasks_per_core = 1
                self.use_multithreading = False
                self.variables = {
                    'MPICH_OFI_STARTUP_CONNECT': '1',
                    'OMP_NUM_THREADS': '8',
                    'OMP_PLACES': 'cores',
                    'OMP_PROC_BIND': 'close'
                }

        else:
            if self.current_system.name in ['daint', 'dom']:
                self.num_tasks = 576
                self.num_tasks_per_node = 36
            elif self.current_system.name in ['eiger', 'pilatus']:
                self.num_tasks = 256
                self.num_tasks_per_node = 16
                self.num_cpus_per_task = 16
                self.num_tasks_per_core = 1
                self.use_multithreading = False
                self.variables = {
                    'MPICH_OFI_STARTUP_CONNECT': '1',
                    'OMP_NUM_THREADS': '8',
                    'OMP_PLACES': 'cores',
                    'OMP_PROC_BIND': 'close'
                }

    @run_before('run')
    def set_task_distribution(self):
        self.job.options = ['--distribution=block:block']

    @run_before('run')
    def set_cpu_binding(self):
        self.job.launcher.options = ['--cpu-bind=cores']


@rfm.simple_test
class Cp2kGpuCheck(Cp2kCheck):
    valid_systems = ['daint:gpu']
    num_gpus_per_node = 1
    num_tasks_per_node = 6
    num_cpus_per_task = 2

    @run_after('init')
    def set_description(self):
        self.descr = (f'CP2K GPU check (version: {self.scale}, '
                      f'{self.benchmark})')

    @run_after('init')
    def set_reference(self):
        if self.scale == 'small':
            self.reference = REFERENCE_GPU_PERFORMANCE_SMALL
        else:
            self.reference = REFERENCE_GPU_PERFORMANCE_LARGE

    @run_after('init')
    def set_num_tasks(self):
        if self.scale == 'small':
            self.valid_systems += ['dom:gpu']
            self.num_tasks = 36
        else:
            self.num_tasks = 96

        self.variables = {
            'CRAY_CUDA_MPS': '1',
            'OMP_NUM_THREADS': str(self.num_cpus_per_task)
        }
