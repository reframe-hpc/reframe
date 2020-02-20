# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn


class NamdBaseCheck(rfm.RunOnlyRegressionTest):
    def __init__(self, arch, scale, variant):
        super().__init__()
        self.descr = 'NAMD check (%s, %s)' % (arch, variant)
        self.valid_prog_environs = ['PrgEnv-intel']
        self.modules = ['NAMD']

        # Reset sources dir relative to the SCS apps prefix
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'NAMD', 'prod')
        self.executable = 'namd2'
        self.use_multithreading = True
        self.num_tasks_per_core = 2

        if scale == 'small':
            self.num_tasks = 6
            self.num_tasks_per_node = 1
        else:
            self.num_tasks = 16
            self.num_tasks_per_node = 1

        energy = sn.avg(sn.extractall(r'ENERGY:(\s+\S+){10}\s+(?P<energy>\S+)',
                                      self.stdout, 'energy', float))
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
                'Info: Benchmark time: \S+ CPUs \S+ '
                's/step (?P<days_ns>\S+) days/ns \S+ MB memory',
                self.stdout, 'days_ns', float))
        }

        self.maintainers = ['CB', 'LM']
        self.tags = {'scs', 'external-resources'}
        self.strict_check = False
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
                'dom:gpu': {'days_ns': (0.18, None, 0.05, 'days/ns')},
                'daint:gpu': {'days_ns': (0.18, None, 0.05, 'days/ns')}
            }
        else:
            self.reference = {
                'daint:gpu': {'days_ns': (0.11, None, 0.05, 'days/ns')}
            }


@rfm.required_version('>=2.16')
@rfm.parameterized_test(*([s, v]
                          for s in ['small', 'large']
                          for v in ['maint', 'prod']))
class NamdCPUCheck(NamdBaseCheck):
    def __init__(self, scale, variant):
        super().__init__('cpu', scale, variant)
        self.valid_systems = ['daint:mc']
        self.executable_opts = ['+idlepoll', '+ppn 71', 'stmv.namd']
        self.num_cpus_per_task = 72
        if scale == 'small':
            self.valid_systems += ['dom:mc']
            self.reference = {
                'dom:mc': {'days_ns': (0.57, None, 0.05, 'days/ns')},
                'daint:mc': {'days_ns': (0.56, None, 0.05, 'days/ns')}
            }
        else:
            self.reference = {
                'daint:mc': {'days_ns': (0.38, None, 0.05, 'days/ns')}
            }

        self.tags |= {'maintenance' if variant == 'maint' else 'production'}
