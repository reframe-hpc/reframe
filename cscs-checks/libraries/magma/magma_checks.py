# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class MagmaCheck(rfm.RegressionTest):
    subtest = parameter(['cblas_z', 'zgemm', 'zsymmetrize', 'ztranspose',
                         'zunmbr'])
    valid_systems = ['daint:gpu', 'dom:gpu']
    valid_prog_environs = ['builtin']
    num_gpus_per_node = 1
    prebuild_cmds = ['patch < patch.txt']
    modules = ['magma']
    maintainers = ['AJ', 'SK']
    tags = {'scs', 'production', 'maintenance'}

    @run_before('compile')
    def set_build_system_opts(self):
        self.build_system = 'Make'
        self.build_system.makefile = f'Makefile_{self.subtest}'
        self.build_system.cxxflags = ['-std=c++11']
        self.build_system.ldflags = ['-lcusparse', '-lcublas', '-lmagma',
                                     '-lmagma_sparse']
        self.executable = f'./testing_{self.subtest}'
        # FIXME: Compile cblas_z  with -O0 since with a higher level a
        # segmentation fault is thrown
        if self.subtest == 'cblas_z':
            self.build_system.cxxflags += ['-O0']

    @run_before('run')
    def set_exec_opts(self):
        if self.subtest == 'zgemm':
            self.executable_opts = ['--range 1088:3136:1024']

    @sanity_function
    def assert_success(self):
        return sn.assert_found(r'Result = PASS', self.stdout)

    @run_before('performance')
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
