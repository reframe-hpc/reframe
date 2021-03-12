# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


class VASPCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        if self.current_system.name == 'pilatus':
            self.valid_prog_environs = ['cpeIntel']
        else:
            self.valid_prog_environs = ['builtin']

        self.modules = ['VASP']
        force = sn.extractsingle(r'1 F=\s+(?P<result>\S+)',
                                 self.stdout, 'result', float)
        self.sanity_patterns = sn.assert_reference(
            force, -.85026214E+03, -1e-5, 1e-5
        )
        self.keep_files = ['OUTCAR']
        self.perf_patterns = {
            'time': sn.extractsingle(r'Total CPU time used \(sec\):'
                                     r'\s+(?P<time>\S+)', 'OUTCAR',
                                     'time', float)
        }
        self.maintainers = ['LM']
        self.tags = {'scs'}
        self.strict_check = False
        self.extra_resources = {
            'switches': {
                'num_switches': 1
            }
        }


@rfm.parameterized_test(*([v] for v in ['maint', 'prod']))
class VASPCpuCheck(VASPCheck):
    def __init__(self, variant):
        super().__init__()
        self.descr = f'VASP CPU check (variant: {variant})'
        self.valid_systems = ['daint:mc', 'dom:mc', 'eiger:mc', 'pilatus:mc']

        self.executable = 'vasp_std'
        if self.current_system.name == 'dom':
            self.num_tasks = 72
            self.num_tasks_per_node = 12
            self.use_multithreading = True
        elif self.current_system.name in ['eiger', 'pilatus']:
            self.num_tasks = 64
            self.num_tasks_per_node = 4
            self.num_cpus_per_task = 8
            self.num_tasks_per_core = 1
            self.use_multithreading = False
            self.variables = {
                'MPICH_OFI_STARTUP_CONNECT': '1',
                'OMP_NUM_THREADS': str(self.num_cpus_per_task),
                'OMP_PLACES': 'cores',
                'OMP_PROC_BIND': 'close'
            }
        else:
            self.num_tasks = 32
            self.num_tasks_per_node = 2
            self.use_multithreading = True

        references = {
            'maint': {
                'dom:mc': {'time': (148.7, None, 0.05, 's')},
                'daint:mc': {'time': (105.3, None, 0.20, 's')},
                'eiger:mc': {'time': (100.0, None, 0.10, 's')},
                'pilatus:mc': {'time': (100.0, None, 0.10, 's')}
            },
            'prod': {
                'dom:mc': {'time': (148.7, None, 0.05, 's')},
                'daint:mc': {'time': (105.3, None, 0.20, 's')},
                'eiger:mc': {'time': (100.0, None, 0.10, 's')},
                'pilatus:mc': {'time': (100.0, None, 0.10, 's')}
            }
        }
        self.reference = references[variant]
        self.tags |= {'maintenance' if variant == 'maint' else 'production'}

    @rfm.run_before('run')
    def set_task_distribution(self):
        self.job.options = ['--distribution=block:block']

    @rfm.run_before('run')
    def set_cpu_binding(self):
        self.job.launcher.options = ['--cpu-bind=cores']


@rfm.parameterized_test(*([v] for v in ['maint', 'prod']))
class VASPGpuCheck(VASPCheck):
    def __init__(self, variant):
        super().__init__()
        self.descr = f'VASP GPU check (variant: {variant})'
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.executable = 'vasp_gpu'
        self.variables = {'CRAY_CUDA_MPS': '1'}
        self.num_gpus_per_node = 1
        if self.current_system.name == 'dom':
            self.num_tasks = 6
            self.num_tasks_per_node = 1
        else:
            self.num_tasks = 16
            self.num_tasks_per_node = 1

        references = {
            'maint': {
                'dom:gpu': {'time': (61.0, None, 0.10, 's')},
                'daint:gpu': {'time': (46.7, None, 0.20, 's')},
            },
            'prod': {
                'dom:gpu': {'time': (61.0, None, 0.10, 's')},
                'daint:gpu': {'time': (46.7, None, 0.20, 's')},
            }
        }
        self.reference = references[variant]
        self.tags |= {'maintenance' if variant == 'maint' else 'production'}
