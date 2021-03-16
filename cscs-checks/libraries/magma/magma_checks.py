# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.parameterized_test(['cblas_z'], ['zgemm'],
                        ['zsymmetrize'], ['ztranspose'])
class MagmaCheck(rfm.RegressionTest):
    def __init__(self, subtest):
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.num_gpus_per_node = 1
        self.sanity_patterns = sn.assert_found(r'Result = PASS', self.stdout)

        self.prebuild_cmds = ['patch < patch.txt']
        self.build_system = 'Make'
        self.valid_prog_environs = ['builtin']
        self.build_system.makefile = 'Makefile_%s' % subtest
        # Compile with -O0 since with a higher level the compiler seems to
        # optimise something away
        self.build_system.cflags = ['-O0']
        self.build_system.cxxflags = ['-O0', '-std=c++11']
        self.build_system.ldflags = ['-lcusparse', '-lcublas', '-lmagma',
                                     '-lmagma_sparse']
        self.executable = './testing_' + subtest
        self.modules = ['magma']
        self.maintainers = ['AJ', 'SK']
        self.tags = {'scs', 'production', 'maintenance'}
        if subtest == 'cblas_z':
            self.perf_patterns = {
                'duration': sn.extractsingle(r'Duration: (\S+)',
                                             self.stdout, 1, float)
            }
            self.reference = {
                'daint:gpu': {
                    'duration': (0.10, None, 1.05, 's'),
                },
                'dom:gpu': {
                    'duration': (0.10, None, 1.05, 's'),
                },
            }
        elif subtest == 'zgemm':
            self.executable_opts = ['--range 1088:3136:1024']
            self.perf_patterns = {
                'magma': sn.extractsingle(
                    r'MAGMA GFlops: (?P<magma_gflops>\S+)',
                    self.stdout, 'magma_gflops', float, 2
                ),
                'cublas': sn.extractsingle(
                    r'cuBLAS GFlops: (?P<cublas_gflops>\S+)', self.stdout,
                    'cublas_gflops', float, 2)
            }
            self.reference = {
                'daint:gpu': {
                    'magma':  (3692.65, -0.05, None, 'Gflop/s'),
                    'cublas': (4269.31, -0.09, None, 'Gflop/s'),
                },
                'dom:gpu': {
                    'magma':  (3692.65, -0.05, None, 'Gflop/s'),
                    'cublas': (4269.31, -0.09, None, 'Gflop/s'),
                }
            }
        elif subtest == 'zsymmetrize':
            self.perf_patterns = {
                'gpu_perf': sn.extractsingle(r'GPU performance: (\S+)',
                                             self.stdout, 1, float),
            }
            self.reference = {
                'daint:gpu': {
                    'gpu_perf': (158.3, -0.05, None, 'GB/s'),
                },
                'dom:gpu': {
                    'gpu_perf': (158.3, -0.05, None, 'GB/s'),
                }
            }
        elif subtest == 'ztranspose':
            self.perf_patterns = {
                'gpu_perf':
                    sn.extractsingle(
                        r'GPU performance: (?P<gpu_performance>\S+)',
                        self.stdout, 'gpu_performance', float
                    )
            }
            self.reference = {
                'daint:gpu': {
                    'gpu_perf': (498.2, -0.05, None, 'GB/s'),
                },
                'dom:gpu': {
                    'gpu_perf': (498.2, -0.05, None, 'GB/s'),
                }
            }
        elif subtest == 'zunmbr':
            # This test fails to compile with Magma 2.4
            self.perf_patterns = {
                'gpu_perf':
                    sn.extractsingle(
                        r'GPU performance: (?P<gpu_performance>\S+)',
                        self.stdout, 'gpu_performance', float
                    )
            }
            self.reference = {
                'daint:gpu': {
                    'gpu_perf': (254.7, -0.05, None, 'Gflop/s'),
                },
                'dom:gpu': {
                    'gpu_perf': (254.7, -0.05, None, 'Gflop/s'),
                }
            }
