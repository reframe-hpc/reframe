# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
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
        self.valid_systems = ['daint:mc']
        self.modules = ['QuantumESPRESSO/6.5-CrayIntel-19.10']
        if scale == 'small':
            self.valid_systems += ['dom:mc']
            self.num_tasks = 216
            energy_reference = -11427.09017162
        else:
            self.num_tasks = 576
            energy_reference = -11427.09017152

        self.num_tasks_per_node = 36

        energy = sn.extractsingle(r'!\s+total energy\s+=\s+(?P<energy>\S+) Ry',
                                  self.stdout, 'energy', float)
        energy_diff = sn.abs(energy-energy_reference)
        self.sanity_patterns = sn.all([
            self.sanity_patterns,
            sn.assert_lt(energy_diff, 1e-8)
        ])

        references = {
            'maint': {
                'small': {
                    'dom:mc': {'time': (115.0, None, 0.05, 's')},
                    'daint:mc': {'time': (115.0, None, 0.10, 's')}
                },
                'large': {
                    'daint:mc': {'time': (115.0, None, 0.10, 's')}
                }
            },
            'prod': {
                'small': {
                    'dom:mc': {'time': (115.0, None, 0.05, 's')},
                    'daint:mc': {'time': (115.0, None, 0.10, 's')}
                },
                'large': {
                    'daint:mc': {'time': (115.0, None, 0.10, 's')}
                }
            }
        }

        self.reference = references[variant][scale]
        self.tags |= {'maintenance' if variant == 'maint' else 'production'}


@rfm.parameterized_test(*([s, v]
                          for s in ['small', 'large']
                          for v in ['maint', 'prod']))
class QuantumESPRESSOGpuCheck(QuantumESPRESSOCheck):
    def __init__(self, scale, variant):
        super().__init__()
        self.descr = f'QuantumESPRESSO GPU check (version: {scale}, {variant})'
        self.valid_systems = ['daint:gpu']
        self.modules = ['QuantumESPRESSO/6.5a1-CrayPGI-19.10-cuda-10.1']
        self.num_gpus_per_node = 1
        if scale == 'small':
            self.valid_systems += ['dom:gpu']
            self.num_tasks = 6
            energy_reference = -11427.09017176
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
            sn.assert_lt(energy_diff, 1e-8)
        ])

        references = {
            'maint': {
                'small': {
                    'dom:gpu': {'time': (60.0, None, 0.05, 's')},
                    'daint:gpu': {'time': (60.0, None, 0.10, 's')}
                },
                'large': {
                    'daint:gpu': {'time': (60.0, None, 0.10, 's')}
                }
            },
            'prod': {
                'small': {
                    'dom:gpu': {'time': (60.0, None, 0.05, 's')},
                    'daint:gpu': {'time': (60.0, None, 0.10, 's')}
                },
                'large': {
                    'daint:gpu': {'time': (60.0, None, 0.10, 's')}
                }
            }
        }

        self.reference = references[variant][scale]
        self.tags |= {'maintenance' if variant == 'maint' else 'production'}
