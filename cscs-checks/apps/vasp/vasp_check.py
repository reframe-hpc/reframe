# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


class VASPCheck(rfm.RunOnlyRegressionTest):
    keep_files = ['OUTCAR']
    maintainers = ['LM']
    tags = {'scs'}
    strict_check = False
    modules = ['VASP']
    extra_resources = {
        'switches': {
            'num_switches': 1
        }
    }

    @run_after('init')
    def setup_by_system(self):
        if self.current_system.name in ['eiger', 'pilatus']:
            self.valid_prog_environs = ['cpeIntel']
        else:
            self.valid_prog_environs = ['builtin']

    @sanity_function
    def assert_reference(self):
        force = sn.extractsingle(r'1 F=\s+(?P<result>\S+)',
                                 self.stdout, 'result', float)
        return sn.assert_reference(force, -.85026214E+03, -1e-5, 1e-5)

    @performance_function('s')
    def time(self):
        return sn.extractsingle(r'Elapsed time \(sec\):'
                                r'\s+(?P<time>\S+)', 'OUTCAR',
                                'time', float)


@rfm.simple_test
class VASPCpuCheck(VASPCheck):
    variant = parameter(['maint', 'prod'])
    valid_systems = ['daint:mc', 'dom:mc', 'eiger:mc', 'pilatus:mc']
    executable = 'vasp_std'
    references_by_variant = {
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

    @run_after('init')
    def setup_by_variant(self):
        self.descr = f'VASP CPU check (variant: {self.variant})'

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

        self.reference = self.references_by_variant[self.variant]
        self.tags |= {'maintenance'
                      if self.variant == 'maint' else 'production'}

    @run_before('run')
    def set_task_distribution(self):
        self.job.options = ['--distribution=block:block']

    @run_before('run')
    def set_cpu_binding(self):
        self.job.launcher.options = ['--cpu-bind=cores']


@rfm.simple_test
class VASPGpuCheck(VASPCheck):
    variant = parameter(['maint', 'prod'])
    valid_systems = ['daint:gpu', 'dom:gpu']
    executable = 'vasp_std'
    num_gpus_per_node = 1
    references_by_variant = {
        'maint': {
            'dom:gpu': {'time': (61.0, None, 0.10, 's')},
            'daint:gpu': {'time': (46.7, None, 0.20, 's')},
        },
        'prod': {
            'dom:gpu': {'time': (61.0, None, 0.10, 's')},
            'daint:gpu': {'time': (46.7, None, 0.20, 's')},
        }
    }

    @run_after('init')
    def setup_by_variant(self):
        self.descr = f'VASP GPU check (variant: {self.variant})'
        if self.current_system.name == 'dom':
            self.num_tasks = 6
            self.num_tasks_per_node = 1
            self.num_cpus_per_task = 12
        else:
            self.num_tasks = 16
            self.num_tasks_per_node = 1

        self.reference = self.references_by_variant[self.variant]
        self.tags |= {'maintenance'
                      if self.variant == 'maint' else 'production'}
