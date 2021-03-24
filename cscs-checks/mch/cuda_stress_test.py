# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class CudaStressTest(rfm.RegressionTest):
    def __init__(self):
        self.descr = 'MCH CUDA stress test'
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'arolla:cn', 'tsa:cn']
        if self.current_system.name in ['arolla', 'tsa']:
            self.exclusive_access = True
            self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-gnu-nompi',
                                        'PrgEnv-pgi', 'PrgEnv-pgi-nompi']
            self.modules = ['cuda/10.1.243']
        else:
            self.valid_prog_environs = ['PrgEnv-gnu']
            self.modules = ['craype-accel-nvidia60', 'cdt-cuda']

        self.sourcepath = 'cuda_stencil_test.cu'
        self.build_system = 'SingleSource'
        self.build_system.cxxflags = ['-std=c++11']
        self.num_tasks = 1
        self.num_gpus_per_node = 1
        self.sanity_patterns = sn.assert_found(r'Result: OK', self.stdout)
        self.perf_patterns = {
            'time': sn.extractsingle(r'Timing: (\S+)', self.stdout, 1, float)
        }
        self.reference = {
            'daint:gpu': {
                'time': (1.41184, None, 0.05, 's')
            },
            'dom:gpu': {
                'time': (1.39758, None, 0.05, 's')
            },
        }
        self.tags = {'production', 'mch', 'craype'}
        self.maintainers = ['MKr', 'AJ']
