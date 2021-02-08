# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


class Cp2kCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.valid_prog_environs = ['builtin']
        self.modules = ['CP2K']
        self.executable = 'cp2k.psmp'
        self.executable_opts = ['H2O-256.inp']

        energy = sn.extractsingle(
            r'\s+ENERGY\| Total FORCE_EVAL \( QS \) '
            r'energy [\[\(]a\.u\.[\]\)]:\s+(?P<energy>\S+)',
            self.stdout, 'energy', float, item=-1
        )
        energy_reference = -4404.2323
        energy_diff = sn.abs(energy-energy_reference)
        self.sanity_patterns = sn.all([
            sn.assert_found(r'PROGRAM STOPPED IN', self.stdout),
            sn.assert_eq(sn.count(sn.extractall(
                r'(?i)(?P<step_count>STEP NUMBER)',
                self.stdout, 'step_count')), 10),
            sn.assert_lt(energy_diff, 1e-4)
        ])

        self.perf_patterns = {
            'time': sn.extractsingle(r'^ CP2K(\s+[\d\.]+){4}\s+(?P<perf>\S+)',
                                     self.stdout, 'perf', float)
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
class Cp2kCpuCheck(Cp2kCheck):
    def __init__(self, scale, variant):
        super().__init__()
        self.descr = 'CP2K CPU check (version: %s, %s)' % (scale, variant)
        self.valid_systems = ['daint:mc', 'eiger:mc']
        if scale == 'small':
            self.valid_systems += ['dom:mc']
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

        references = {
            'maint': {
                'small': {
                    'dom:mc': {'time': (202.2, None, 0.05, 's')},
                    'daint:mc': {'time': (180.9, None, 0.08, 's')},
                    'eiger:mc': {'time': (70.0, None, 0.08, 's')}
                },
                'large': {
                    'daint:mc': {'time': (141.0, None, 0.05, 's')},
                    'eiger:mc': {'time': (46.0, None, 0.05, 's')}
                }
            },
            'prod': {
                'small': {
                    'dom:mc': {'time': (202.2, None, 0.05, 's')},
                    'daint:mc': {'time': (180.9, None, 0.08, 's')},
                    'eiger:mc': {'time': (70.0, None, 0.08, 's')}
                },
                'large': {
                    'daint:mc': {'time': (113.0, None, 0.05, 's')},
                    'eiger:mc': {'time': (46.0, None, 0.05, 's')}
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
class Cp2kGpuCheck(Cp2kCheck):
    def __init__(self, scale, variant):
        super().__init__()
        self.descr = 'CP2K GPU check (version: %s, %s)' % (scale, variant)
        self.valid_systems = ['daint:gpu']
        self.num_gpus_per_node = 1
        if scale == 'small':
            self.valid_systems += ['dom:gpu']
            self.num_tasks = 36
        else:
            self.num_tasks = 96

        self.num_tasks_per_node = 6
        self.num_cpus_per_task = 2
        self.variables = {
            'CRAY_CUDA_MPS': '1',
            'OMP_NUM_THREADS': str(self.num_cpus_per_task)
        }
        references = {
            'maint': {
                'small': {
                    'dom:gpu': {'time': (251.8, None, 0.15, 's')},
                    'daint:gpu': {'time': (241.3, None, 0.05, 's')}
                },
                'large': {
                    'daint:gpu': {'time': (199.6, None, 0.06, 's')}
                }
            },
            'prod': {
                'small': {
                    'dom:gpu': {'time': (240.0, None, 0.05, 's')},
                    'daint:gpu': {'time': (241.3, None, 0.05, 's')}
                },
                'large': {
                    'daint:gpu': {'time': (199.6, None, 0.06, 's')}
                }
            }
        }
        self.reference = references[variant][scale]
        self.tags |= {'maintenance' if variant == 'maint' else 'production'}
