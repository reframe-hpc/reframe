# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import itertools
import os

import reframe as rfm
import reframe.utility.sanity as sn


class GromacsBaseCheck(rfm.RunOnlyRegressionTest):
    def __init__(self, output_file):
        self.valid_prog_environs = ['builtin']
        self.executable = 'gmx_mpi'

        # Reset sources dir relative to the SCS apps prefix
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'Gromacs', 'herflat')
        self.keep_files = [output_file]

        energy = sn.extractsingle(r'\s+Potential\s+Kinetic En\.\s+Total Energy'
                                  r'\s+Conserved En\.\s+Temperature\n'
                                  r'(\s+\S+){2}\s+(?P<energy>\S+)(\s+\S+){2}\n'
                                  r'\s+Pressure \(bar\)\s+Constr\. rmsd',
                                  output_file, 'energy', float, item=-1)
        energy_reference = -3270799.9

        self.sanity_patterns = sn.all([
            sn.assert_found('Finished mdrun', output_file),
            sn.assert_reference(energy, energy_reference, -0.001, 0.001)
        ])

        self.perf_patterns = {
            'perf': sn.extractsingle(r'Performance:\s+(?P<perf>\S+)',
                                     output_file, 'perf', float)
        }

        self.modules = ['GROMACS']
        self.maintainers = ['VH', 'SO']
        self.strict_check = False
        self.use_multithreading = False
        self.extra_resources = {
            'switches': {
                'num_switches': 1
            }
        }
        self.tags = {'scs', 'external-resources'}


@rfm.required_version('>=2.19')
@rfm.parameterized_test(*([s, v]
                          for s in ['small', 'large']
                          for v in ['prod', 'maint']))
class GromacsGPUCheck(GromacsBaseCheck):
    def __init__(self, scale, variant):
        super().__init__('md.log')
        self.valid_systems = ['daint:gpu', 'tiger:gpu']
        self.descr = 'GROMACS GPU check'
        self.executable_opts = ['mdrun', '-dlb yes', '-ntomp 1', '-npme 0',
                                '-s herflat.tpr']
        self.variables = {'CRAY_CUDA_MPS': '1'}
        self.num_gpus_per_node = 1
        if scale == 'small':
            self.valid_systems += ['dom:gpu']
            self.num_tasks = 72
            self.num_tasks_per_node = 12
        else:
            self.num_tasks = 192
            self.num_tasks_per_node = 12

        references = {
            'maint': {
                'large': {
                    'daint:gpu': {'perf': (73.4, -0.10, None, 'ns/day')}
                }
            },
            'prod': {
                'small': {
                    'dom:gpu': {'perf': (37.0, -0.05, None, 'ns/day')},
                    'daint:gpu': {'perf': (35.0, -0.10, None, 'ns/day')}
                },
                'large': {
                    'daint:gpu': {'perf': (63.0, -0.20, None, 'ns/day')}
                }
            },
        }
        with contextlib.suppress(KeyError):
            self.reference = references[variant][scale]

        self.tags |= {'maintenance' if variant == 'maint' else 'production'}


@rfm.required_version('>=2.19')
@rfm.parameterized_test(*([s, v]
                          for s in ['small', 'large']
                          for v in ['prod']))
class GromacsCPUCheck(GromacsBaseCheck):
    def __init__(self, scale, variant):
        super().__init__('md.log')
        self.valid_systems = ['daint:mc']
        self.descr = 'GROMACS CPU check'
        self.executable_opts = ['mdrun', '-dlb yes', '-ntomp 1', '-npme -1',
                                '-nb cpu', '-s herflat.tpr']

        if scale == 'small':
            self.valid_systems += ['dom:mc']
            self.num_tasks = 216
            self.num_tasks_per_node = 36
        else:
            self.num_tasks = 576
            self.num_tasks_per_node = 36

        references = {
            'prod': {
                'small': {
                    'dom:mc': {'perf': (40.0, -0.05, None, 'ns/day')},
                    'daint:mc': {'perf': (38.8, -0.10, None, 'ns/day')}
                },
                'large': {
                    'daint:mc': {'perf': (68.0, -0.20, None, 'ns/day')}
                }
            },
        }
        self.reference = references[variant][scale]
        self.tags |= {'maintenance' if variant == 'maint' else 'production'}
