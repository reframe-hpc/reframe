# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


class VASPCheck(rfm.RunOnlyRegressionTest):
    scale = parameter(['small', 'large'])
    variant = parameter(['maint', 'prod'])
    modules = ['VASP']
    executable = 'vasp_std'
    keep_files = ['OUTCAR']
    extra_resources = {
        'switches': {
            'num_switches': 1
        }
    }
    strict_check = False
    maintainers = ['LM']
    tags = {'scs'}

    @run_after('init')
    def setup_by_system(self):
        if self.current_system.name in ['eiger', 'pilatus']:
            self.valid_prog_environs = ['cpeIntel']
        else:
            self.valid_prog_environs = ['builtin']

        self.tags |= {
            'maintenance' if self.variant == 'maint' else 'production'
        }

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
    valid_systems = ['daint:mc', 'eiger:mc', 'pilatus:mc']
    use_multithreading = False

    @run_after('init')
    def setup_by_scale(self):
        self.descr = (f'VASP CPU check (version: {self.scale}, '
                      f'{self.variant})')
        if self.scale == 'small':
            self.valid_systems += ['dom:mc']
            if self.current_system.name in ['daint', 'dom']:
                self.num_tasks = 12
                self.num_tasks_per_node = 2
                self.num_cpus_per_task = 18
            elif self.current_system.name in ['eiger', 'pilatus']:
                self.num_tasks = 64
                self.num_tasks_per_node = 4
                self.num_cpus_per_task = 8
                self.num_tasks_per_core = 1
                self.variables = {
                    'MPICH_OFI_STARTUP_CONNECT': '1',
                    'OMP_NUM_THREADS': str(self.num_cpus_per_task),
                    'OMP_PLACES': 'cores',
                    'OMP_PROC_BIND': 'close'
                }
        else:
            if self.current_system.name in ['daint']:
                self.num_tasks = 32
                self.num_tasks_per_node = 2
                self.num_cpus_per_task = 18

    @run_before('performance')
    def set_reference(self):
        references = {
            'small': {
                'dom:mc': {'time': (65.0, None, 0.15, 's')},
                'daint:mc': {'time': (65.0, None, 0.15, 's')},
                'eiger:mc': {'time': (100.0, None, 0.10, 's')},
                'pilatus:mc': {'time': (100.0, None, 0.10, 's')}
            },
            'large': {
                'daint:mc': {'time': (55.0, None, 0.15, 's')},
                'eiger:mc': {'time': (100.0, None, 0.10, 's')},
                'pilatus:mc': {'time': (100.0, None, 0.10, 's')}
            }
        }
        self.reference = references[self.scale]

    @run_before('run')
    def set_task_distribution(self):
        self.job.options = ['--distribution=block:block']

    @run_before('run')
    def set_cpu_binding(self):
        self.job.launcher.options = ['--cpu-bind=cores']


@rfm.simple_test
class VASPGpuCheck(VASPCheck):
    valid_systems = ['daint:gpu']
    num_gpus_per_node = 1
    num_tasks_per_node = 1
    num_cpus_per_task = 12
    use_multithreading = False

    @run_after('init')
    def setup_test(self):
        self.descr = (f'VASP GPU check (version: {self.scale}, '
                      f'{self.variant})')
        if self.scale == 'small':
            self.valid_systems += ['dom:gpu']
            self.num_tasks = 6
        else:
            self.num_tasks = 16

    @run_before('performance')
    def set_reference(self):
        references = {
            'small': {
                'dom:gpu': {'time': (65.0, None, 0.15, 's')},
                'daint:gpu': {'time': (65.0, None, 0.15, 's')}
            },
            'large': {
                'daint:gpu': {'time': (55.0, None, 0.15, 's')}
            }
        }
        self.reference = references[self.scale]
