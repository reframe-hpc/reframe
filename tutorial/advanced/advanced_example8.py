# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.parameterized_test(['MPI'], ['OpenMP'])
class MatrixVectorTest(rfm.RegressionTest):
    def __init__(self, variant):
        self.descr = 'Matrix-vector multiplication test (%s)' % variant
        self.valid_systems = ['daint:gpu', 'daint:mc']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi']
        self.build_system = 'SingleSource'
        self.prgenv_flags = {
            'PrgEnv-cray':  ['-homp'],
            'PrgEnv-gnu':   ['-fopenmp'],
            'PrgEnv-intel': ['-openmp'],
            'PrgEnv-pgi':   ['-mp']
        }

        if variant == 'MPI':
            self.num_tasks = 8
            self.num_tasks_per_node = 2
            self.num_cpus_per_task = 4
            self.sourcepath = 'example_matrix_vector_multiplication_mpi_openmp.c'
        elif variant == 'OpenMP':
            self.sourcepath = 'example_matrix_vector_multiplication_openmp.c'
            self.num_cpus_per_task = 4

        self.variables = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task)
        }
        matrix_dim = 1024
        iterations = 100
        self.executable_opts = [str(matrix_dim), str(iterations)]

        expected_norm = matrix_dim
        found_norm = sn.extractsingle(
            r'The L2 norm of the resulting vector is:\s+(?P<norm>\S+)',
            self.stdout, 'norm', float)
        self.sanity_patterns = sn.all([
            sn.assert_found(
                r'time for single matrix vector multiplication', self.stdout),
            sn.assert_lt(sn.abs(expected_norm - found_norm), 1.0e-6)
        ])
        self.maintainers = ['you-can-type-your-email-here']
        self.tags = {'tutorial'}

    @rfm.run_before('compile')
    def setflags(self):
        if self.prgenv_flags is not None:
            env = self.current_environ.name
            self.build_system.cflags = self.prgenv_flags[env]
