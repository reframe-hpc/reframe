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

        energy = sn.extractsingle(r'!\s+total energy\s+=\s+(?P<energy>\S+) Ry',
                                  self.stdout, 'energy', float)
        energy_reference = -11427.09017179
        energy_diff = sn.abs(energy-energy_reference)
        self.sanity_patterns = sn.all([
            sn.assert_found(r'convergence has been achieved', self.stdout),
            sn.assert_lt(energy_diff, 1e-10)
        ])

        self.perf_patterns = {
            'time': sn.extractsingle(r'electrons    :\s+(?P<sec>\S+)s CPU ',
                                     self.stdout, 'sec', float)
        }

        self.maintainers = ['LM']
        self.tags = {'scs'}
        self.strict_check = False
        self.modules = ['QuantumESPRESSO']
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
        self.descr = 'QuantumESPRESSO CPU check (version: %s, %s)' % (scale, variant)
        self.valid_systems = ['daint:mc']
        if scale == 'small':
            self.valid_systems += ['dom:mc']
            self.num_tasks = 216
        else:
            self.num_tasks = 576

        self.num_tasks_per_node = 36
        references = {
            'maint': {
                'small': {
                    'dom:mc': {'time': (159.0, None, 0.05, 's')},
                    'daint:mc': {'time': (147.3, None, 0.41, 's')}
                },
                'large': {
                    'daint:mc': {'time': (149.7, None, 0.52, 's')}
                }
            },
            'prod': {
                'small': {
                    'dom:mc': {'time': (159.0, None, 0.05, 's')},
                    'daint:mc': {'time': (147.3, None, 0.41, 's')}
                },
                'large': {
                    'daint:mc': {'time': (149.7, None, 0.52, 's')}
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
        self.descr = 'QuantumESPRESSO GPU check (version: %s, %s)' % (scale, variant)
        self.valid_systems = ['daint:gpu']
        self.variables = {'CRAY_CUDA_MPS': '1'}
        self.num_gpus_per_node = 1
        if scale == 'small':
            self.valid_systems += ['dom:gpu']
            self.num_tasks = 72
        else:
            self.num_tasks = 192

        self.num_tasks_per_node = 12
        references = {
            'maint': {
                'small': {
                    'dom:mc': {'time': (159.0, None, 0.05, 's')},
                    'daint:mc': {'time': (147.3, None, 0.41, 's')}
                },
                'large': {
                    'daint:mc': {'time': (149.7, None, 0.52, 's')}
                }
            },
            'prod': {
                'small': {
                    'dom:mc': {'time': (159.0, None, 0.05, 's')},
                    'daint:mc': {'time': (147.3, None, 0.41, 's')}
                },
                'large': {
                    'daint:mc': {'time': (149.7, None, 0.52, 's')}
                }
            }
        }

        self.reference = references[variant][scale]
        self.tags |= {'maintenance' if variant == 'maint' else 'production'}
