# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn


class NamdBaseCheck(rfm.RunOnlyRegressionTest):
    def __init__(self, arch, scale, variant):
        self.descr = 'NAMD check (%s, %s)' % (arch, variant)
        self.valid_prog_environs = ['builtin']
        self.modules = ['NAMD']

        # Reset sources dir relative to the SCS apps prefix
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'NAMD', 'prod')
        self.executable = 'namd2'
        self.use_multithreading = True
        self.num_tasks_per_core = 2

        if scale == 'small':
            # On Eiger a no-smp NAMD version is the default
            if self.current_system.name == 'eiger':
                self.num_tasks = 768
                self.num_tasks_per_node = 128
            else:
                self.num_tasks = 6
                self.num_tasks_per_node = 1
        else:
            if self.current_system.name == 'eiger':
                self.num_tasks = 2048
                self.num_tasks_per_node = 128
            else:
                self.num_tasks = 16
                self.num_tasks_per_node = 1

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

        self.maintainers = ['CB', 'LM']
        self.tags = {'scs', 'external-resources'}
        self.extra_resources = {
            'switches': {
                'num_switches': 1
            }
        }


@rfm.required_version('>=2.16')
@rfm.parameterized_test(*([s, v]
                          for s in ['small', 'large']
                          for v in ['maint', 'prod']))
class NamdGPUCheck(NamdBaseCheck):
    def __init__(self, scale, variant):
        super().__init__('gpu', scale, variant)
        self.valid_systems = ['daint:gpu']
        self.executable_opts = ['+idlepoll', '+ppn 23', 'stmv.namd']
        self.num_cpus_per_task = 24
        self.num_gpus_per_node = 1
        self.tags |= {'maintenance' if variant == 'maint' else 'production'}
        if scale == 'small':
            self.valid_systems += ['dom:gpu']
            self.reference = {
                'dom:gpu': {'days_ns': (0.15, None, 0.05, 'days/ns')},
                'daint:gpu': {'days_ns': (0.15, None, 0.05, 'days/ns')}
            }
        else:
            self.reference = {
                'daint:gpu': {'days_ns': (0.07, None, 0.05, 'days/ns')}
            }


@rfm.required_version('>=2.16')
@rfm.parameterized_test(*([s, v]
                          for s in ['small', 'large']
                          for v in ['maint', 'prod']))
class NamdCPUCheck(NamdBaseCheck):
    def __init__(self, scale, variant):
        super().__init__('cpu', scale, variant)
        self.valid_systems = ['daint:mc', 'eiger:mc']
        # On Eiger a no-smp NAMD version is the default
        if self.current_system.name == 'eiger':
            self.executable_opts = ['+idlepoll', 'stmv.namd']
            self.num_tasks_per_core = 2
        else:
            self.executable_opts = ['+idlepoll', '+ppn 71', 'stmv.namd']
            self.num_cpus_per_task = 72
        if scale == 'small':
            self.valid_systems += ['dom:mc']
            self.reference = {
                'dom:mc': {'days_ns': (0.51, None, 0.05, 'days/ns')},
                'daint:mc': {'days_ns': (0.51, None, 0.05, 'days/ns')},
                'eiger:mc': {'days_ns': (0.12, None, 0.05, 'days/ns')}
            }
        else:
            self.reference = {
                'daint:mc': {'days_ns': (0.28, None, 0.05, 'days/ns')},
                'eiger:mc': {'days_ns': (0.05, None, 0.05, 'days/ns')}
            }

        self.tags |= {'maintenance' if variant == 'maint' else 'production'}
