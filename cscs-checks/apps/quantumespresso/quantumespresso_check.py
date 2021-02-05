# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


class QuantumESPRESSOCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.valid_prog_environs = ['builtin']
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


@rfm.parameterized_test(*([s, v]
                          for s in ['small', 'large']
                          for v in ['maint', 'prod']))
class QuantumESPRESSOCpuCheck(QuantumESPRESSOCheck):
    def __init__(self, scale, variant):
        super().__init__()
        self.descr = f'QuantumESPRESSO CPU check (version: {scale}, {variant})'
        self.valid_systems = ['daint:mc', 'eiger:mc']
        self.modules = ['QuantumESPRESSO']
        if scale == 'small':
            self.valid_systems += ['dom:mc']
            energy_reference = -11427.09017218
            if self.current_system.name in ['daint', 'dom']:
                self.num_tasks = 216
                self.num_tasks_per_node = 36
            elif self.current_system.name == 'eiger':
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
            energy_reference = -11427.09017152
            if self.current_system.name in ['daint', 'dom']:
                self.num_tasks = 576
                self.num_tasks_per_node = 36
            elif self.current_system.name in ['eiger']:
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

        energy = sn.extractsingle(r'!\s+total energy\s+=\s+(?P<energy>\S+) Ry',
                                  self.stdout, 'energy', float)
        energy_diff = sn.abs(energy-energy_reference)
        self.sanity_patterns = sn.all([
            self.sanity_patterns,
            # FIXME temporarily increase energy difference
            # (different QE default on Dom and Daint)
            sn.assert_lt(energy_diff, 1e-6)
        ])

        references = {
            'maint': {
                'small': {
                    'dom:mc': {'time': (115.0, None, 0.05, 's')},
                    'daint:mc': {'time': (115.0, None, 0.10, 's')},
                    'eiger:mc': {'time': (66.0, None, 0.10, 's')}
                },
                'large': {
                    'daint:mc': {'time': (115.0, None, 0.10, 's')},
                    'eiger:mc': {'time': (53.0, None, 0.10, 's')}
                }
            },
            'prod': {
                'small': {
                    'dom:mc': {'time': (115.0, None, 0.05, 's')},
                    'daint:mc': {'time': (115.0, None, 0.10, 's')},
                    'eiger:mc': {'time': (66.0, None, 0.10, 's')}
                },
                'large': {
                    'daint:mc': {'time': (115.0, None, 0.10, 's')},
                    'eiger:mc': {'time': (53.0, None, 0.10, 's')}
                }
            }
        }

        self.reference = references[variant][scale]
        self.tags |= {'maintenance' if variant == 'maint' else 'production'}

    @rfm.run_before('run')
    def set_task_distribution(self):
        self.job.options = ['--distribution=block:block']

    @rfm.run_before('run')
    def set_cpu_binding(self):
        self.job.launcher.options = ['--cpu-bind=cores']

@rfm.parameterized_test(*([s, v]
                          for s in ['small', 'large']
                          for v in ['maint', 'prod']))
class QuantumESPRESSOGpuCheck(QuantumESPRESSOCheck):
    def __init__(self, scale, variant):
        super().__init__()
        self.descr = f'QuantumESPRESSO GPU check (version: {scale}, {variant})'
        self.valid_systems = ['daint:gpu']
        self.modules = ['QuantumESPRESSO']
        self.num_gpus_per_node = 1
        if scale == 'small':
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
            # FIXME temporarily increase energy difference
            # (different CUDA default on Dom and Daint)
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

        self.reference = references[variant][scale]
        self.tags |= {'maintenance' if variant == 'maint' else 'production'}
