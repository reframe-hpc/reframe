# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn


class FieldextraTestBase(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.maintainers = ['MKr']
        self.tags = {'mch', 'external-resources'}

        self.valid_systems = ['kesch:cn']
        self.valid_prog_environs = ['PrgEnv-gnu-nompi']
        self.executable = 'cookbook/run.bash'

        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.num_cpus_per_task = 24
        self.num_task_per_core = 1
        self.use_multithreading = True
        self.strict_check = False


@rfm.parameterized_test(['opt'], ['opt_omp'])
class FieldextraAccuracyTest(FieldextraTestBase):
    def __init__(self, variant):
        super().__init__()
        self.descr = 'Fieldextra validation test (accuracy - "cookbook")'
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'Fieldextra', 'cookbook_tests')
        self.readonly_files = [
            'cookbook/support/input',
            'reference_cookbook'
        ]
        self.pre_run = [
            'ln -s ${EBROOTFIELDEXTRA}/bin bin',
            'ln -s ${EBROOTFIELDEXTRA}/tools tools',
            'ln -s ${EBROOTFIELDEXTRA}/resources resources',
        ]

        if variant == 'opt':
            self.modules = ['fieldextra/12.7.5-gmvolf-17.02']
            self.executable_opts = ['-c gnu -m opt']
        else:
            self.modules = ['fieldextra/12.7.5-gmvolf-17.02-openmp']
            self.executable_opts = ['-c', 'gnu', '-m', 'opt_omp']
            self.variables = {
                'OMP_STACKSIZE': '500M',
                'OMP_NUM_THREADS': '$SLURM_CPUS_PER_TASK',
            }

        self.sanity_patterns = sn.assert_found(r'All tests successful',
                                               self.stdout)


@rfm.simple_test
class FieldextraPerformanceTest(FieldextraTestBase):
    def __init__(self):
        super().__init__()
        self.descr = 'Fieldextra validation test (performance)'
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'Fieldextra', 'performance')
        self.modules = ['fieldextra/12.7.5-gmvolf-17.02-openmp']
        self.executable = 'fieldextra_gnu_opt_omp'
        self.executable_opts = ['nl.TC']
        self.readonly_files = ['18112900_204']
        self.pre_run = ['source create_nl_6h.template']
        self.variables = {
            'OMP_STACKSIZE': '500M',
            'OMP_NUM_THREADS': '$SLURM_CPUS_PER_TASK',
        }

        self.sanity_patterns = sn.assert_found(
            r'%INFO fieldextra: Program successfully completed', self.stdout
        )
        self.perf_patterns = {
            'time': sn.extractsingle(r'WALL CLOCK\s*SPEEDUP\D*(?P<time>\S+)',
                                     'fieldextra.diagnostic', 'time', float)
        }
        self.reference = {
            'kesch': {
                'time': (420., None, 0.10, 's')
            }
        }
