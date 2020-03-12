# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import reframe as rfm
import reframe.utility.sanity as sn


class Cp2kCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.valid_prog_environs = ['builtin']
        self.executable = 'cp2k.psmp'
        self.executable_opts = ['H2O-256.inp']

        energy = sn.extractsingle(r'\s+ENERGY\| Total FORCE_EVAL \( QS \) '
                                  r'energy \(a\.u\.\):\s+(?P<energy>\S+)',
                                  self.stdout, 'energy', float, item=-1)
        energy_reference = -4404.2323
        energy_diff = sn.abs(energy-energy_reference)
        self.sanity_patterns = sn.all([
            sn.assert_found(r'PROGRAM STOPPED IN', self.stdout),
            sn.assert_eq(sn.count(sn.extractall(
                r'(?P<step_count>STEP NUM)',
                self.stdout, 'step_count')), 10),
            sn.assert_lt(energy_diff, 1e-4)
        ])

        self.perf_patterns = {
            'time': sn.extractsingle(r'^ CP2K(\s+[\d\.]+){4}\s+(?P<perf>\S+)',
                                     self.stdout, 'perf', float)
        }

        self.maintainers = ['LM']
        self.tags = {'scs'}
        self.strict_check = False
        self.modules = ['CP2K']
        self.extra_resources = {
            'switches': {
                'num_switches': 1
            }
        }


@rfm.parameterized_test(*([s, v]
                          for s in ['small', 'large']
                          for v in ['maint', 'prod']))
class Cp2kCpuCheck(Cp2kCheck):
    def __init__(self, scale, variant):
        super().__init__()
        self.descr = 'CP2K CPU check (version: %s, %s)' % (scale, variant)
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
                    'dom:mc': {'time': (202.2, None, 0.05, 's')},
                    'daint:mc': {'time': (214.5, None, 0.15, 's')}
                },
                'large': {
                    'daint:mc': {'time': (141.0, None, 0.05, 's')}
                }
            },
            'prod': {
                'small': {
                    'dom:mc': {'time': (202.2, None, 0.05, 's')},
                    'daint:mc': {'time': (214.5, None, 0.15, 's')}
                },
                'large': {
                    'daint:mc': {'time': (113.0, None, 0.05, 's')}
                }
            }
        }

        self.reference = references[variant][scale]
        self.tags |= {'maintenance' if variant == 'maint' else 'production'}


@rfm.parameterized_test(*([s, v]
                          for s in ['small', 'large']
                          for v in ['maint', 'prod']))
class Cp2kGpuCheck(Cp2kCheck):
    def __init__(self, scale, variant):
        super().__init__()
        self.descr = 'CP2K GPU check (version: %s, %s)' % (scale, variant)
        self.valid_systems = ['daint:gpu']
        self.variables = {'CRAY_CUDA_MPS': '1'}
        self.modules = ['CP2K']
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
                    'dom:gpu': {'time': (251.8, None, 0.15, 's')},
                    'daint:gpu': {'time': (262.6, None, 0.10, 's')}
                },
                'large': {
                    'daint:gpu': {'time': (222.6, None, 0.05, 's')}
                }
            },
            'prod': {
                'small': {
                    'dom:gpu': {'time': (240.0, None, 0.05, 's')},
                    'daint:gpu': {'time': (262.6, None, 0.10, 's')}
                },
                'large': {
                    'daint:gpu': {'time': (222.6, None, 0.05, 's')}
                }
            }
        }
        self.reference = references[variant][scale]
        self.tags |= {'maintenance' if variant == 'maint' else 'production'}
