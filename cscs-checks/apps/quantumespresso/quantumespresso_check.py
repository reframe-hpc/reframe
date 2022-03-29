# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


class QuantumESPRESSOCheck(rfm.RunOnlyRegressionTest):
    scale = parameter(['small', 'large'])
    variant = parameter(['maint', 'prod'])
    modules = ['QuantumESPRESSO']
    executable = 'pw.x'
    executable_opts = ['-in', 'ausurf.in', '-pd', '.true.']
    extra_resources = {
        'switches': {
            'num_switches': 1
        }
    }
    strict_check = False
    maintainers = ['LM']
    tags = {'scs'}

    @run_after('init')
    def set_prog_envs_and_tags(self):
        if self.current_system.name in ['eiger', 'pilatus']:
            self.valid_prog_environs = ['cpeIntel']
        else:
            self.valid_prog_environs = ['builtin']

        self.tags |= {
            'maintenance' if self.variant == 'maint' else 'production'
        }

    @sanity_function
    def assert_simulation_success(self):
        energy = sn.extractsingle(r'!\s+total energy\s+=\s+(?P<energy>\S+) Ry',
                                  self.stdout, 'energy', float)
        energy_diff = sn.abs(energy-self.energy_reference)
        return sn.all([
            sn.assert_found(r'convergence has been achieved', self.stdout),
            sn.assert_lt(energy_diff, self.energy_tolerance)
        ])

    @performance_function('s')
    def time(self):
        return sn.extractsingle(r'electrons.+\s(?P<wtime>\S+)s WALL',
                                self.stdout, 'wtime', float)


@rfm.simple_test
class QuantumESPRESSOCpuCheck(QuantumESPRESSOCheck):
    valid_systems = ['daint:mc', 'eiger:mc', 'pilatus:mc']
    energy_tolerance = 1.0e-6

    @run_after('init')
    def setup_test(self):
        self.descr = (f'QuantumESPRESSO CPU check (version: {self.scale}, '
                      f'{self.variant})')
        if self.scale == 'small':
            self.valid_systems += ['dom:mc']
            self.energy_reference = -11427.09017218
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
            self.energy_reference = -11427.09017152
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

    @run_before('performance')
    def set_reference(self):
        references = {
            'small': {
                'dom:mc': {'time': (110.0, None, 0.05, 's')},
                'daint:mc': {'time': (110.0, None, 0.20, 's')},
                'eiger:mc': {'time': (66.0, None, 0.10, 's')},
                'pilatus:mc': {'time': (66.0, None, 0.10, 's')}
            },
            'large': {
                'daint:mc': {'time': (145.0, None, 0.30, 's')},
                'eiger:mc': {'time': (53.0, None, 0.10, 's')},
                'pilatus:mc': {'time': (53.0, None, 0.10, 's')}
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
class QuantumESPRESSOGpuCheck(QuantumESPRESSOCheck):
    valid_systems = ['daint:gpu']
    num_gpus_per_node = 1
    num_tasks_per_node = 1
    num_cpus_per_task = 12
    energy_tolerance = 1.0e-7

    @run_after('init')
    def setup_test(self):
        self.descr = (f'QuantumESPRESSO GPU check (version: {self.scale}, '
                      f'{self.variant})')
        if self.scale == 'small':
            self.valid_systems += ['dom:gpu']
            self.num_tasks = 6
            self.energy_reference = -11427.09017168
        else:
            self.num_tasks = 16
            self.energy_reference = -11427.09017179

    @run_before('performance')
    def set_reference(self):
        references = {
            'small': {
                'dom:gpu': {'time': (59.0, None, 0.05, 's')},
                'daint:gpu': {'time': (59.0, None, 0.05, 's')}
            },
            'large': {
                'daint:gpu': {'time': (39.0, None, 0.05, 's')}
            }
        }
        self.reference = references[self.scale]
