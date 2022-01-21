# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class cuda_stress_test(rfm.RegressionTest):
    descr = 'MCH CUDA stress test'
    valid_systems = ['daint:gpu', 'dom:gpu', 'arolla:cn', 'tsa:cn']
    valid_prog_environs = ['*']
    sourcepath = 'cuda_stencil_test.cu'
    build_system = 'SingleSource'
    num_tasks = 1
    num_gpus_per_node = 1
    tags = {'production', 'mch', 'craype', 'health'}
    maintainers = ['MKr', 'AJ']

    @run_after('init')
    def set_environment(self):
        if self.current_system.name in ['arolla', 'tsa']:
            self.exclusive_access = True
            self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-gnu-nompi',
                                        'PrgEnv-pgi', 'PrgEnv-pgi-nompi']
        else:
            self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-nvidia']

    @run_after('setup')
    def set_modules(self):
        if self.current_system.name in {'arolla', 'tsa'}:
            self.modules = ['cuda/10.1.243']
        elif self.current_environ.name != 'PrgEnv-nvidia':
            self.modules = ['craype-accel-nvidia60', 'cdt-cuda']

    @run_before('compile')
    def set_compile_flags(self):
        self.build_system.cxxflags = ['-std=c++11']

    @run_before('sanity')
    def set_sanity_and_perf(self):
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
