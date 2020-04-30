# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn


class LAMMPSBaseCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.valid_prog_environs = ['builtin']
        self.modules = ['LAMMPS']

        # Reset sources dir relative to the SCS apps prefix
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'LAMMPS')
        energy_reference = -4.6195
        energy = sn.extractsingle(
            r'\s+500000(\s+\S+){3}\s+(?P<energy>\S+)\s+\S+\s\n',
            self.stdout, 'energy', float)
        self.perf_patterns = {
            'perf': sn.extractsingle(r'\s+(?P<perf>\S+) timesteps/s',
                                     self.stdout, 'perf', float),
        }
        energy_diff = sn.abs(energy-energy_reference)
        self.sanity_patterns = sn.all([
            sn.assert_found(r'Total wall time:', self.stdout),
            sn.assert_lt(energy_diff, 6e-4)
        ])
        self.strict_check = False
        self.extra_resources = {
            'switches': {
                'num_switches': 1
            }
        }

        self.tags = {'scs', 'external-resources'}
        self.maintainers = ['TR', 'VH']


@rfm.required_version('>=2.16')
@rfm.parameterized_test(*([s, v]
                          for s in ['small', 'large']
                          for v in ['prod', 'maint']))
class LAMMPSGPUCheck(LAMMPSBaseCheck):
    def __init__(self, scale, variant):
        super().__init__()
        self.valid_systems = ['daint:gpu']
        self.executable = 'lmp_mpi'
        self.executable_opts = ['-sf gpu', '-pk gpu 1', '-in in.lj.gpu']
        self.variables = {'CRAY_CUDA_MPS': '1'}
        self.num_gpus_per_node = 1
        if scale == 'small':
            self.valid_systems += ['dom:gpu']
            self.num_tasks = 12
            self.num_tasks_per_node = 2
        else:
            self.num_tasks = 32
            self.num_tasks_per_node = 2

        references = {
            'maint': {
                'small': {
                    'dom:gpu': {'perf': (3457, -0.10, None, 'timesteps/s')},
                    'daint:gpu': {'perf': (2524, -0.10, None, 'timesteps/s')}
                },
                'large': {
                    'daint:gpu': {'perf': (3832, -0.05, None, 'timesteps/s')}
                }
            },
            'prod': {
                'small': {
                    'dom:gpu': {'perf': (3132, -0.05, None, 'timesteps/s')},
                    'daint:gpu': {'perf': (2400, -0.40, None, 'timesteps/s')}
                },
                'large': {
                    'daint:gpu': {'perf': (3260, -0.50, None, 'timesteps/s')}
                }
            },
        }
        self.reference = references[variant][scale]
        self.tags |= {'maintenance' if variant == 'maint' else 'production'}


@rfm.required_version('>=2.16')
@rfm.parameterized_test(*([s, v]
                          for s in ['small', 'large']
                          for v in ['prod']))
class LAMMPSCPUCheck(LAMMPSBaseCheck):
    def __init__(self, scale, variant):
        super().__init__()
        self.valid_systems = ['daint:mc']
        self.executable = 'lmp_omp'
        self.executable_opts = ['-sf omp', '-pk omp 1', '-in in.lj.cpu']
        if scale == 'small':
            self.valid_systems += ['dom:mc']
            self.num_tasks = 216
            self.num_tasks_per_node = 36
        else:
            self.num_tasks_per_node = 36
            self.num_tasks = 576

        references = {
            'prod': {
                'small': {
                    'dom:mc': {'perf': (4394, -0.05, None, 'timesteps/s')},
                    'daint:mc': {'perf': (3824, -0.10, None, 'timesteps/s')}
                },
                'large': {
                    'daint:mc': {'perf': (5310, -0.65, None, 'timesteps/s')}
                }
            },
        }
        self.reference = references[variant][scale]
        self.tags |= {'maintenance' if variant == 'maint' else 'production'}
