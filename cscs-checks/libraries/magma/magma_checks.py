# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class MagmaCheck(rfm.RegressionTest):
    subtest = parameter(['cblas_z', 'zgemm', 'zsymmetrize', 'ztranspose'])
    valid_systems = ['daint:gpu', 'dom:gpu']
    valid_prog_environs = ['builtin']
    num_gpus_per_node = 1
    prebuild_cmds = ['patch < patch.txt']
    modules = ['magma']
    maintainers = ['AJ', 'SK']
    tags = {'scs', 'production', 'maintenance'}

    @run_after('init')
    def set_build_system_opts(self):
        self.build_system = 'Make'
        self.build_system.makefile = 'Makefile_%s' % self.subtest
        # Compile with -O0 since with a higher level the compiler seems to
        # optimise stuff away
        self.build_system.cflags = ['-O0']
        self.build_system.cxxflags = ['-O0', '-std=c++11']
        self.build_system.ldflags = ['-lcusparse', '-lcublas', '-lmagma',
                                     '-lmagma_sparse']
        self.executable = './testing_' + self.subtest

    @run_after('init')
    def set_sanity_patterns(self):
        self.sanity_patterns = sn.assert_found(r'Result = PASS', self.stdout)

    @run_after('init')
    def set_performance_patterns(self):
        if self.subtest == 'cblas_z':
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
        elif self.subtest == 'zgemm':
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
        elif self.subtest == 'zsymmetrize':
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
        elif self.subtest == 'ztranspose':
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
        elif self.subtest == 'zunmbr':
            # FIXME: update the test, because it fails to compile with Magma 2.4
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
