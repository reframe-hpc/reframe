# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class NamdCheck(rfm.RunOnlyRegressionTest):
    scale = parameter(['small', 'large'])
    variant = parameter(['maint', 'prod'])
    arch = parameter(['gpu', 'cpu'])

    valid_prog_environs = ['builtin', 'cpeGNU']
    modules = ['NAMD']
    executable = 'namd2'
    use_multithreading = True
    num_tasks_per_core = 2
    maintainers = ['CB', 'LM']
    tags = {'scs', 'external-resources'}
    extra_resources = {
        'switches': {
            'num_switches': 1
        }
    }

    @run_after('init')
    def adapt_description(self):
        self.descr = f'NAMD check ({self.arch}, {self.variant})'
        self.tags |= {
            'maintenance' if self.variant == 'maint' else 'production'
        }

    @run_after('init')
    def adapt_valid_systems(self):
        if self.arch == 'gpu':
            self.valid_systems = ['daint:gpu']
            if self.scale == 'small':
                self.valid_systems += ['dom:gpu']
        else:
            self.valid_systems = ['daint:mc', 'eiger:mc', 'pilatus:mc']
            if self.scale == 'small':
                self.valid_systems += ['dom:mc']

    @run_after('init')
    def adapt_valid_prog_environs(self):
        if self.current_system.name in ['eiger', 'pilatus']:
            self.valid_prog_environs.remove('builtin')

    @run_after('init')
    def setup_parallel_run(self):
        if self.arch == 'gpu':
            self.executable_opts = ['+idlepoll', '+ppn 23', 'stmv.namd']
            self.num_cpus_per_task = 24
            self.num_gpus_per_node = 1
        else:
            # On Eiger a no-smp NAMD version is the default
            if self.current_system.name in ['eiger', 'pilatus']:
                self.executable_opts = ['+idlepoll', 'stmv.namd']
            else:
                self.executable_opts = ['+idlepoll', '+ppn 71', 'stmv.namd']
                self.num_cpus_per_task = 72
        if self.scale == 'small':
            # On Eiger a no-smp NAMD version is the default
            if self.current_system.name in ['eiger', 'pilatus']:
                self.num_tasks = 768
                self.num_tasks_per_node = 128
            else:
                self.num_tasks = 6
                self.num_tasks_per_node = 1
        else:
            if self.current_system.name in ['eiger', 'pilatus']:
                self.num_tasks = 2048
                self.num_tasks_per_node = 128
            else:
                self.num_tasks = 16
                self.num_tasks_per_node = 1

    @run_before('compile')
    def prepare_build(self):
        # Reset sources dir relative to the SCS apps prefix
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'NAMD', 'prod')

    @sanity_function
    def validate_energy(self):
        energy = sn.avg(sn.extractall(
            r'ENERGY:([ \t]+\S+){10}[ \t]+(?P<energy>\S+)',
            self.stdout, 'energy', float)
        )
        energy_reference = -2451359.5
        energy_diff = sn.abs(energy - energy_reference)
        return sn.all([
            sn.assert_eq(sn.count(sn.extractall(
                         r'TIMING: (?P<step_num>\S+)  CPU:',
                         self.stdout, 'step_num')), 50),
            sn.assert_lt(energy_diff, 2720)
        ])

    @run_before('performance')
    def set_reference(self):
        if self.arch == 'gpu':
            if self.scale == 'small':
                self.reference = {
                    'dom:gpu': {'days_ns': (0.15, None, 0.05, 'days/ns')},
                    'daint:gpu': {'days_ns': (0.15, None, 0.05, 'days/ns')}
                }
            else:
                self.reference = {
                    'daint:gpu': {'days_ns': (0.07, None, 0.05, 'days/ns')}
                }
        else:
            if self.scale == 'small':
                self.reference = {
                    'dom:mc': {'days_ns': (0.51, None, 0.05, 'days/ns')},
                    'daint:mc': {'days_ns': (0.51, None, 0.05, 'days/ns')},
                    'eiger:mc': {'days_ns': (0.12, None, 0.05, 'days/ns')},
                    'pilatus:mc': {'days_ns': (0.12, None, 0.05, 'days/ns')},
                }
            else:
                self.reference = {
                    'daint:mc': {'days_ns': (0.28, None, 0.05, 'days/ns')},
                    'eiger:mc': {'days_ns': (0.05, None, 0.05, 'days/ns')},
                    'pilatus:mc': {'days_ns': (0.05, None, 0.05, 'days/ns')}
                }

    @performance_function('days/ns')
    def days_ns(self):
        return sn.avg(sn.extractall(
            r'Info: Benchmark time: \S+ CPUs \S+ '
            r's/step (?P<days_ns>\S+) days/ns \S+ MB memory',
            self.stdout, 'days_ns', float))
