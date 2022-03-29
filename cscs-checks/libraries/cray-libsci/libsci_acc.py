# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


# @rfm.parameterized_test(['dgemm'])
class LibsciAccBaseCheck(rfm.RegressionTest):
    def __init__(self):
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu']
        self.prebuild_cmds = ['module list']
        # FIXME: https://jira.cscs.ch/browse/PROGENV-24
        self.modules = ['craype-accel-nvidia60', 'cray-libsci_acc']
        self.maintainers = ['JG']
        self.tags = {'scs', 'production', 'maintenance'}


@rfm.simple_test
class LibsciAccDgemmCublasCheck(LibsciAccBaseCheck):
    def __init__(self):
        super().__init__()
        self.descr = 'Test Cray LibSci on the GPU (dgemm with cublas alloc)'
        self.build_system = 'SingleSource'
        self.sourcesdir = None
        self.sourcepath = ('$CRAY_LIBSCI_ACC_DIR/examples/examples/gnu_cuda/'
                           'dgemm_cuda.c')
        self.build_system.ldflags = ['-lcublas']
        self.sanity_patterns = sn.assert_found(r'(4096\s+){3}', self.stdout)
        regex = r'(\s+\d+){3}\s+(?P<gpu_flops>\S+)\s+(?P<cpu_flops>\S+)\s+'
        self.perf_patterns = {
            'dgemm_gpu': sn.max(sn.extractall(regex, self.stdout, 'gpu_flops',
                                              float)),
            'dgemm_cpu': sn.max(sn.extractall(regex, self.stdout, 'cpu_flops',
                                              float)),
        }
        self.reference = {
            'daint:gpu': {
                'dgemm_gpu': (4127.0, -0.05, None, 'GFLop/s'),
                'dgemm_cpu': (45.0, -0.05, None, 'GFLop/s'),
            },
            'dom:gpu': {
                'dgemm_gpu': (4127.0, -0.05, None, 'GFLop/s'),
                'dgemm_cpu': (45.0, -0.05, None, 'GFLop/s'),
            },
        }


@rfm.simple_test
class LibsciAccDgemmCheck(LibsciAccBaseCheck):
    def __init__(self):
        super().__init__()
        self.descr = 'Test Cray LibSci on the GPU (dgemm with libsci alloc)'
        self.build_system = 'SingleSource'
        self.sourcesdir = None
        self.sourcepath = ('$CRAY_LIBSCI_ACC_DIR/examples/examples/c_simple/'
                           'dgemm_simple.c')
        self.sanity_patterns = sn.assert_found(r'(4096\s+){3}', self.stdout)
        regex = r'(\s+\d+){3}\s+(?P<gpu_flops>\S+)\s+(?P<cpu_flops>\S+)\s+'
        self.perf_patterns = {
            'dgemm_gpu': sn.max(sn.extractall(regex, self.stdout, 'gpu_flops',
                                              float)),
            'dgemm_cpu': sn.max(sn.extractall(regex, self.stdout, 'cpu_flops',
                                              float)),
        }
        self.reference = {
            'daint:gpu': {
                'dgemm_gpu': (2264.0, -0.05, None, 'GFLop/s'),
                'dgemm_cpu': (45.0, -0.05, None, 'GFLop/s'),
            },
            'dom:gpu': {
                'dgemm_gpu': (2264.0, -0.05, None, 'GFLop/s'),
                'dgemm_cpu': (45.0, -0.05, None, 'GFLop/s'),
            },
        }
