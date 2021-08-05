# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
from hpctestlib.apps.quantumespresso import QuantumESPRESSO


REFERENCE_CPU_PERFORMANCE_SMALL = {
    'dom:mc': {
        'maint': (115.0, None, 0.05, 's'),
        'prod': (115.0, None, 0.05, 's')
    },
    'daint:mc': {
        'maint': (115.0, None, 0.10, 's'),
        'prod': (115.0, None, 0.10, 's')
    },
    'eiger:mc': {
        'maint': (66.0, None, 0.10, 's'),
        'prod': (66.0, None, 0.10, 's')
    },
    'pilatus:mc': {
        'maint': (66.0, None, 0.10, 's'),
        'prod': (66.0, None, 0.10, 's')
    },
}

REFERENCE_CPU_PERFORMANCE_LARGE = {
    'daint:mc': {
        'maint': (115.0, None, 0.10, 's'),
        'prod': (115.0, None, 0.10, 's')
    },
    'eiger:mc': {
        'maint': (53.0, None, 0.10, 's'),
        'prod': (53.0, None, 0.10, 's')
    },
    'pilatus:mc': {
        'maint': (53.0, None, 0.10, 's'),
        'prod': (53.0, None, 0.10, 's')
    },
}

REFERENCE_GPU_PERFORMANCE_SMALL = {
    'dom:mc': {
        'maint': (61.0, None, 0.05, 's'),
        'prod': (61.0, None, 0.05, 's')
    },
    'daint:mc': {
        'maint': (61.0, None, 0.05, 's'),
        'prod': (61.0, None, 0.05, 's')
    }
}

REFERENCE_GPU_PERFORMANCE_LARGE = {
    'daint:mc': {
        'maint': (54.0, None, 0.05, 's'),
        'prod': (54.0, None, 0.05, 's')
    }
}


class QuantumESPRESSOCheck(QuantumESPRESSO):
    scale = parameter(['small', 'large'])
    benchmark = parameter(['maint', 'prod'])

    modules = ['QuantumESPRESSO']
    executable = 'pw.x'
    input_file = 'ausurf.in'
    maintainers = ['LM']
    tags = {'scs'}
    strict_check = False
    extra_resources = {
        'switches': {
            'num_switches': 1
        }
    }

    @run_after('init')
    def env_define(self):
        if self.current_system.name in ['eiger', 'pilatus']:
            self.valid_prog_environs = ['cpeIntel']
        else:
            self.valid_prog_environs = ['builtin']

    @run_after('init')
    def set_reference(self):
        self.reference = references[self.benchmark]

    @run_after('init')
    def set_tags(self):
        self.tags |= {
            'maintenance' if self.benchmark == 'maint' else 'production'
        }

    @run_after('setup')
    def set_generic_perf_references(self):
        self.reference.update({'*': {
            self.benchmark: (0, None, None, 's')
        }})

    @run_after('setup')
    def set_perf_patterns(self):
        self.perf_patterns = {
            self.benchmark: sn.extractsingle(
                r'electrons.+\s(?P<wtime>\S+)s WALL',
                self.stdout, 'wtime', float)
        }

@rfm.simple_test
class QuantumESPRESSOCpuCheck(QuantumESPRESSOCheck):
    valid_systems = ['daint:mc', 'eiger:mc', 'pilatus:mc']

    @run_after('init')
    def set_reference(self):
        if self.scale == 'small':
            self.reference = REFERENCE_CPU_PERFORMANCE_SMALL
            self.energy_value = -11427.09017168
            self.energy_tolerance = 1E-06
        else:
            self.reference = REFERENCE_CPU_PERFORMANCE_LARGE
            self.energy_value = -11427.09017152
            self.energy_tolerance = 1E-06

    @run_after('init')
    def set_description(self):
        self.descr = (f'QuantumESPRESSO CPU check (version: {self.scale}, '
                      f'{self.benchmark})')

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
                self.variables = {
                    'MPICH_OFI_STARTUP_CONNECT': '1',
                    'OMP_NUM_THREADS': '8',
                    'OMP_PLACES': 'cores',
                    'OMP_PROC_BIND': 'close'
                }
        else:
            if self.current_system.name in ['daint']:
                self.num_tasks = 576
                self.num_tasks_per_node = 36
            elif self.current_system.name in ['eiger', 'pilatus']:
                self.num_tasks = 256
                self.num_tasks_per_node = 16
                self.num_cpus_per_task = 16
                self.num_tasks_per_core = 1
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
class QuantumESPRESSOGpuCheck(QuantumESPRESSOCheck):
    valid_systems = ['daint:gpu']
    num_gpus_per_node = 1
    num_tasks_per_node = 1
    num_cpus_per_task = 12

    @run_after('init')
    def set_reference(self):
        if self.scale == 'small':
            self.reference = REFERENCE_GPU_PERFORMANCE_SMALL
            self.energy_value = -11427.09017168
            self.energy_tolerance = 1E-07
        else:
            self.reference = REFERENCE_GPU_PERFORMANCE_LARGE
            self.energy_value = -11427.09017179
            self.energy_tolerance = 1E-07

    @run_after('init')
    def set_description(self):
        self.descr = (f'QuantumESPRESSO GPU check (version: {self.scale}, '
                      f'{self.benchmark})')

    @run_after('init')
    def set_num_tasks(self):
        if self.scale == 'small':
            self.valid_systems += ['dom:gpu']
            self.num_tasks = 6
        else:
            self.num_tasks = 16
