# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


class QuantumESPRESSOCheck(rfm.RunOnlyRegressionTest):
    scale = parameter(['small', 'large'])
    variant = parameter(['maint', 'prod'])

    def __init__(self):
        if self.current_system.name in ['eiger', 'pilatus']:
            self.valid_prog_environs = ['cpeIntel']
        else:
            self.valid_prog_environs = ['builtin']

        self.modules = ['QuantumESPRESSO']
        self.executable = 'pw.x'
        self.executable_opts = ['-in', 'ausurf.in']

        self.sanity_patterns = sn.all([
            sn.assert_found(r'convergence has been achieved', self.stdout),
        ])

        self.perf_patterns = {
            'time': sn.extractsingle(r'electrons.+\s(?P<wtime>\S+)s WALL',
                                     self.stdout, 'wtime', float)
        }

        self.maintainers = ['LM']
        self.tags = {'scs'}
        self.strict_check = False
        self.extra_resources = {
            'switches': {
                'num_switches': 1
            }
        }


@rfm.simple_test
class QuantumESPRESSOCpuCheck(QuantumESPRESSOCheck):
    def __init__(self):
        super().__init__()
        self.descr = (f'QuantumESPRESSO CPU check (version: {self.scale}, '
                      f'{self.variant})')
        self.valid_systems = ['daint:mc', 'eiger:mc', 'pilatus:mc']
        if self.scale == 'small':
            self.valid_systems += ['dom:mc']
            energy_reference = -11427.09017218
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
                    'OMP_NUM_THREADS': str(self.num_cpus_per_task),
                    'OMP_PLACES': 'cores',
                    'OMP_PROC_BIND': 'close'
                }
        else:
            energy_reference = -11427.09017152
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
                    'OMP_NUM_THREADS': str(self.num_cpus_per_task),
                    'OMP_PLACES': 'cores',
                    'OMP_PROC_BIND': 'close'
                }

        energy = sn.extractsingle(r'!\s+total energy\s+=\s+(?P<energy>\S+) Ry',
                                  self.stdout, 'energy', float)
        energy_diff = sn.abs(energy-energy_reference)
        self.sanity_patterns = sn.all([
            self.sanity_patterns,
            sn.assert_lt(energy_diff, 1e-6)
        ])

        references = {
            'maint': {
                'small': {
                    'dom:mc': {'time': (115.0, None, 0.05, 's')},
                    'daint:mc': {'time': (115.0, None, 0.10, 's')},
                    'eiger:mc': {'time': (66.0, None, 0.10, 's')},
                    'pilatus:mc': {'time': (66.0, None, 0.10, 's')}
                },
                'large': {
                    'daint:mc': {'time': (115.0, None, 0.10, 's')},
                    'eiger:mc': {'time': (53.0, None, 0.10, 's')},
                    'pilatus:mc': {'time': (53.0, None, 0.10, 's')}
                }
            },
            'prod': {
                'small': {
                    'dom:mc': {'time': (115.0, None, 0.05, 's')},
                    'daint:mc': {'time': (115.0, None, 0.10, 's')},
                    'eiger:mc': {'time': (66.0, None, 0.10, 's')},
                    'pilatus:mc': {'time': (66.0, None, 0.10, 's')}
                },
                'large': {
                    'daint:mc': {'time': (115.0, None, 0.10, 's')},
                    'eiger:mc': {'time': (53.0, None, 0.10, 's')},
                    'pilatus:mc': {'time': (53.0, None, 0.10, 's')}
                }
            }
        }

        self.reference = references[self.variant][self.scale]
        self.tags |= {
            'maintenance' if self.variant == 'maint' else 'production'
        }

    @run_before('run')
    def set_task_distribution(self):
        self.job.options = ['--distribution=block:block']

    @run_before('run')
    def set_cpu_binding(self):
        self.job.launcher.options = ['--cpu-bind=cores']


@rfm.simple_test
class QuantumESPRESSOGpuCheck(QuantumESPRESSOCheck):
    def __init__(self):
        super().__init__()
        self.descr = (f'QuantumESPRESSO GPU check (version: {self.scale}, '
                      f'{self.variant})')
        self.valid_systems = ['daint:gpu']
        self.num_gpus_per_node = 1
        if self.scale == 'small':
            self.valid_systems += ['dom:gpu']
            self.num_tasks = 6
            energy_reference = -11427.09017168
        else:
            self.num_tasks = 16
            energy_reference = -11427.09017179

        self.num_tasks_per_node = 1
        self.num_cpus_per_task = 12

        energy = sn.extractsingle(r'!\s+total energy\s+=\s+(?P<energy>\S+) Ry',
                                  self.stdout, 'energy', float)
        energy_diff = sn.abs(energy-energy_reference)
        self.sanity_patterns = sn.all([
            self.sanity_patterns,
            sn.assert_lt(energy_diff, 1e-7)
        ])

        references = {
            'maint': {
                'small': {
                    'dom:gpu': {'time': (61.0, None, 0.05, 's')},
                    'daint:gpu': {'time': (61.0, None, 0.05, 's')}
                },
                'large': {
                    'daint:gpu': {'time': (54.0, None, 0.05, 's')}
                }
            },
            'prod': {
                'small': {
                    'dom:gpu': {'time': (61.0, None, 0.05, 's')},
                    'daint:gpu': {'time': (61.0, None, 0.05, 's')}
                },
                'large': {
                    'daint:gpu': {'time': (54.0, None, 0.05, 's')}
                }
            }
        }

        self.reference = references[self.variant][self.scale]
        self.tags |= {
            'maintenance' if self.variant == 'maint' else 'production'
        }
