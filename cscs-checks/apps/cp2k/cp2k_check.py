# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
from hpctestlib.apps.cp2k.nve import Cp2k_NVE


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

REFERENCE_CPU_PERFORMANCE = {
    'small': REFERENCE_CPU_PERFORMANCE_SMALL,
    'large': REFERENCE_CPU_PERFORMANCE_LARGE,
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

REFERENCE_GPU_PERFORMANCE = {
    'small': REFERENCE_GPU_PERFORMANCE_SMALL,
    'large': REFERENCE_GPU_PERFORMANCE_LARGE,
}

REFERENCE_PERFORMANCE = {
    'cpu': REFERENCE_CPU_PERFORMANCE,
    'gpu': REFERENCE_GPU_PERFORMANCE,
}


@rfm.simple_test
class cp2k_check(Cp2k_NVE):
    modules = ['CP2K']
    maintainers = ['LM']
    tags = {'scs'}
    strict_check = False
    extra_resources = {
        'switches': {
            'num_switches': 1
        }
    }
    scale = parameter(['small', 'large'])
    mode = parameter(['prod', 'maint'])

    @run_after('init')
    def env_define(self):
        if self.current_system.name in ['eiger', 'pilatus']:
            self.valid_prog_environs = ['cpeGNU']
        else:
            self.valid_prog_environs = ['builtin']

    @run_after('init')
    def set_tags(self):
        self.tags |= {'maintenance' if self.mode == 'maint'
                      else 'production'}

    @run_after('init')
    def set_valid_systems(self):
        if self.platform_name == 'cpu':
            self.valid_systems = ['daint:mc', 'eiger:mc', 'pilatus:mc']
        else:
            self.valid_systems = ['daint:gpu']

    @run_after('init')
    def set_description(self):
        if self.platform_name == 'cpu':
            self.descr = (f'CP2K CPU check (version: {self.scale}, '
                          f'{self.mode})')
        else:
            self.descr = (f'CP2K GPU check (version: {self.scale}, '
                          f'{self.mode})')

    @run_after('init')
    def set_num_tasks(self):
        if self.platform_name == 'cpu':
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
        else:
            self.num_gpus_per_node = 1
            self.num_tasks_per_node = 6
            self.num_cpus_per_task = 2
            if self.scale == 'small':
                self.valid_systems += ['dom:gpu']
                self.num_tasks = 36
            else:
                self.num_tasks = 96

            self.variables = {
                'CRAY_CUDA_MPS': '1',
                'OMP_NUM_THREADS': str(self.num_cpus_per_task)
            }

    @run_after('setup')
    def set_reference(self):
        self.reference = REFERENCE_PERFORMANCE[self.platform_name][self.scale]

    @run_before('run')
    def set_task_distribution(self):
        if self.platform_name == 'cpu':
            self.job.options = ['--distribution=block:block']

    @run_before('run')
    def set_cpu_binding(self):
        if self.platform_name == 'cpu':
            self.job.launcher.options = ['--cpu-bind=cores']
