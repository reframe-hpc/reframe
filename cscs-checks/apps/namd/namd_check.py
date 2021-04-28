# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn


class NamdBaseCheck(rfm.RunOnlyRegressionTest):
    scale = parameter(['small', 'large'])
    variant = parameter(['maint', 'prod'])

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

    def set_description(self, arch):
        self.descr = f'NAMD check ({arch}, {self.variant})'
        self.tags |= {'maintenance' if self.variant == 'maint'
                      else 'production'}

    @rfm.run_after('init')
    def set_job_configuration(self):
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

    @rfm.run_after('init')
    def set_valid_prgenv(self):
        if self.current_system.name == 'pilatus':
            self.valid_prog_environs = ['cpeIntel']
        else:
            self.valid_prog_environs = ['builtin']

    @rfm.run_before('compile')
    def set_sources_dir(self):
        # Reset sources dir relative to the SCS apps prefix
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'NAMD', 'prod')

    @rfm.run_before('sanity')
    def set_sanity_perf(self):
        energy = sn.avg(sn.extractall(
            r'ENERGY:([ \t]+\S+){10}[ \t]+(?P<energy>\S+)',
            self.stdout, 'energy', float)
        )
        energy_reference = -2451359.5
        energy_diff = sn.abs(energy - energy_reference)
        self.sanity_patterns = sn.all([
            sn.assert_eq(sn.count(sn.extractall(
                         r'TIMING: (?P<step_num>\S+)  CPU:',
                         self.stdout, 'step_num')), 50),
            sn.assert_lt(energy_diff, 2720)
        ])
        self.perf_patterns = {
            'days_ns': sn.avg(sn.extractall(
                r'Info: Benchmark time: \S+ CPUs \S+ '
                r's/step (?P<days_ns>\S+) days/ns \S+ MB memory',
                self.stdout, 'days_ns', float))
        }


@rfm.simple_test
class NamdGPUCheck(NamdBaseCheck):
    valid_systems = ['daint:gpu']
    executable_opts = ['+idlepoll', '+ppn 23', 'stmv.namd']
    num_cpus_per_task = 24
    num_gpus_per_node = 1

    @rfm.run_after('init')
    def set_arch(self):
        super().set_description('gpu')

    @rfm.run_after('init')
    def update_valid_systems(self):
        if self.scale == 'small':
            self.valid_systems += ['dom:gpu']

    @rfm.run_before('sanity')
    def set_performance_reference(self):
        if self.scale == 'small':
            self.reference = {
                'dom:gpu': {'days_ns': (0.15, None, 0.05, 'days/ns')},
                'daint:gpu': {'days_ns': (0.15, None, 0.05, 'days/ns')}
            }
        else:
            self.reference = {
                'daint:gpu': {'days_ns': (0.07, None, 0.05, 'days/ns')}
            }


@rfm.simple_test
class NamdCPUCheck(NamdBaseCheck):
    valid_systems = ['daint:mc', 'eiger:mc', 'pilatus:mc']

    @rfm.run_after('init')
    def set_arch(self):
        super().set_description('cpu')

    @rfm.run_after('init')
    def update_valid_systems(self):
        if self.scale == 'small':
            self.valid_systems += ['dom:mc']

    @rfm.run_before('run')
    def set_executable_options(self):
        # On Eiger a no-smp NAMD version is the default
        if self.current_system.name in ['eiger', 'pilatus']:
            self.executable_opts = ['+idlepoll', 'stmv.namd']
        else:
            self.executable_opts = ['+idlepoll', '+ppn 71', 'stmv.namd']
            self.num_cpus_per_task = 72

    @rfm.run_before('sanity')
    def set_performance_reference(self):
        if self.scale == 'small':
            self.reference = {
                'dom:mc': {'days_ns': (0.51, None, 0.05, 'days/ns')},
                'daint:mc': {'days_ns': (0.51, None, 0.05, 'days/ns')},
                'eiger:mc': {'days_ns': (0.12, None, 0.05, 'days/ns')},
                'pilatus:mc': {'days_ns': (0.15, None, 0.05, 'days/ns')},
            }
        else:
            self.reference = {
                'daint:mc': {'days_ns': (0.28, None, 0.05, 'days/ns')},
                'eiger:mc': {'days_ns': (0.05, None, 0.05, 'days/ns')},
                'pilatus:mc': {'days_ns': (0.06, None, 0.05, 'days/ns')}
            }
