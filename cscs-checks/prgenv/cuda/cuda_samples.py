# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class CudaSamplesTest(rfm.RegressionTest):
    sample = parameter([
        'deviceQuery', 'concurrentKernels', 'simpleCUBLAS', 'bandwidthTest',
        'conjugateGradientCudaGraphs'
    ])
    valid_systems = [
        'daint:gpu', 'dom:gpu', 'arolla:cn', 'tsa:cn',
        'ault:amdv100', 'ault:intelv100'
    ]
    sourcesdir = 'https://github.com/NVIDIA/cuda-samples.git'
    build_system = 'Make'
    maintainers = ['JO']
    tags = {'production'}

    # Required variables
    nvidia_sm = variable(str)

    @run_after('init')
    def set_descr(self):
        self.descr = f'CUDA {self.sample} test'

    @run_after('init')
    def set_system_configs(self):
        if self.current_system.name in ['arolla', 'tsa']:
            self.valid_prog_environs = ['PrgEnv-pgi',
                                        'PrgEnv-gnu-nompi',
                                        'PrgEnv-pgi-nompi']
        elif self.current_system.name in ['ault']:
            self.valid_prog_environs = ['PrgEnv-gnu']
        else:
            self.valid_prog_environs = ['PrgEnv-cray',
                                        'PrgEnv-gnu',
                                        'PrgEnv-pgi']

        if self.current_system.name in {'dom'}:
            self.valid_prog_environs += ['PrgEnv-nvidia']

    @run_after('setup')
    def set_gpu_arch(self):
        if self.current_system.name in ['arolla', 'tsa', 'ault']:
            self.exclusive_access = True
            self.nvidia_sm = '70'
        elif self.current_partition.fullname in {'ault:amda100'}:
            self.nvidia_sm = '80'
        else:
            self.nvidia_sm = '60'

    @run_after('setup')
    def set_modules(self):
        if self.current_system.name in {'arolla', 'tsa'}:
            self.modules = ['cuda/10.1.243']
        elif self.current_system.name in {'ault'}:
            self.modules = ['cuda/11.0']
        elif self.current_system.name in {'dom', 'daint'}:
            self.modules = ['craype-accel-nvidia60']
            if self.current_environ.name not in {'PrgEnv-nvidia'}:
                self.modules += ['cdt-cuda']

    @run_before('compile')
    def set_build_options(self):
        self.build_system.options = [
            f'SMS="{self.nvidia_sm}"', f'CUDA_PATH=$CUDA_HOME'
        ]
        self.prebuild_cmds = [
            f'git checkout v11.0', f'cd Samples/{self.sample}'
        ]

    @run_before('run')
    def set_executable(self):
        self.executable = f'Samples/{self.sample}/{self.sample}'

    @run_before('sanity')
    def set_sanity_patterns(self):
        output_patterns = {
            'deviceQuery': r'Result = PASS',
            'concurrentKernels': r'Test passed',
            'simpleCUBLAS': r'test passed',
            'bandwidthTest': r'Result = PASS',
            'conjugateGradientCudaGraphs':
                r'Test Summary:  Error amount = 0.00000'
        }
        self.sanity_patterns = sn.assert_found(
            output_patterns[self.sample], self.stdout
        )
