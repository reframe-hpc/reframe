# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class CudaSamplesBuildCheck(rfm.CompileOnlyRegressionTest):
    def __init__(self):
        super().__init__()
        self.descr = 'Downloads and compiles the entire CUDA Samples repo from NVIDIA.'
        self.sourcesdir = 'https://github.com/NVIDIA/cuda-samples.git'
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'kesch:cn', 'tiger:gpu',
                              'arolla:cn', 'tsa:cn', 'ault:amdv100', 'ault:intelv100']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu']
        if self.current_system.name == 'kesch':
            self.valid_prog_environs += ['PrgEnv-cray-nompi',
                                         'PrgEnv-gnu-nompi']
        elif self.current_system.name in ['arolla', 'tsa']:
            self.valid_prog_environs += ['PrgEnv-pgi',
                                         'PrgEnv-gnu-nompi',
                                         'PrgEnv-pgi-nompi']
        elif self.current_system.name in ['ault']:
            self.valid_prog_environs = ['PrgEnv-gnu']

        # Remove the tensorcore cases, which don't even compile
        self.prebuild_cmds = ['rm Samples/*TensorCoreGemm/Makefile']
        self.sanity_patterns = sn.assert_found(r'Finished building CUDA samples', self.stdout)
        if self.current_system.name == 'kesch':
            self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu']
        elif self.current_system.name in ['arolla', 'tsa','ault']:
            self.valid_prog_environs = ['PrgEnv-gnu']

        self.nvidia_sm = '60'
        self.num_tasks = 8
        if self.current_system.name == 'kesch':
            self.exclusive_access = True
            self.nvidia_sm = '37'
        elif self.current_system.name in ['arolla', 'tsa', 'ault']:
            self.exclusive_access = True
            self.nvidia_sm = '70'

        self.build_system = 'Make'
        self.build_system.options = ['SMS="%s"' % self.nvidia_sm]


class CudaCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'kesch:cn', 'tiger:gpu',
                              'arolla:cn', 'tsa:cn', 'ault:amdv100', 'ault:intelv100']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu']
        if self.current_system.name == 'kesch':
            self.valid_prog_environs += ['PrgEnv-cray-nompi',
                                         'PrgEnv-gnu-nompi']
        elif self.current_system.name in ['arolla', 'tsa']:
            self.valid_prog_environs += ['PrgEnv-pgi',
                                         'PrgEnv-gnu-nompi',
                                         'PrgEnv-pgi-nompi']
        elif self.current_system.name in ['ault']:
            self.valid_prog_environs = ['PrgEnv-gnu']

        self.sourcesdir = None
        self.depends_on('CudaSamplesBuildCheck')
        if self.current_system.name == 'kesch':
            self.modules = ['cudatoolkit/8.0.61']
        elif self.current_system.name in ['arolla', 'tsa']:
            self.modules = ['cuda/10.1.243']
        elif self.current_system.name in ['ault']:
            self.modules = ['cuda/11.0']
        else:
            self.modules = ['craype-accel-nvidia60']

        self.num_tasks = 1
        self.num_gpus_per_node = 1
        self.nvidia_sm = '60'
        if self.current_system.name == 'kesch':
            self.exclusive_access = True
            self.nvidia_sm = '37'
        elif self.current_system.name in ['arolla', 'tsa', 'ault']:
            self.exclusive_access = True
            self.nvidia_sm = '70'

        self.maintainers = ['AJ', 'SK']
        self.tags = {'production', 'craype', 'external-resources'}


@rfm.required_version('>=2.14')
@rfm.simple_test
class CudaMatrixMultCublasCheck(CudaCheck):
    def __init__(self):
        super().__init__()
        self.descr = 'CUDA simpleCUBLAS test'
        self.sanity_patterns = sn.assert_found(
            r'test passed',
            self.stdout)

    @rfm.require_deps
    def set_executable(self, CudaSamplesBuildCheck):
        self.executable = os.path.join(
            CudaSamplesBuildCheck().stagedir, 
            'Samples', 'simpleCUBLAS', 'simpleCUBLAS') 


@rfm.required_version('>=2.14')
@rfm.simple_test
class CudaDeviceQueryCheck(CudaCheck):
    def __init__(self):
        super().__init__()
        self.descr = 'CUDA deviceQuery test.'
        self.sanity_patterns = sn.assert_found(
            r'Result = PASS',
            self.stdout)

    @rfm.require_deps
    def set_executable(self, CudaSamplesBuildCheck):
        self.executable = os.path.join(
            CudaSamplesBuildCheck().stagedir, 
            'Samples', 'deviceQuery', 'deviceQuery') 


@rfm.required_version('>=2.14')
@rfm.simple_test
class CudaConcurrentKernelsCheck(CudaCheck):
    def __init__(self):
        super().__init__()
        self.descr = 'CUDA concurrentKernels test'
        self.sanity_patterns = sn.assert_found(
            r'Test passed',
            self.stdout)

    @rfm.require_deps
    def set_executable(self, CudaSamplesBuildCheck):
        self.executable = os.path.join(
            CudaSamplesBuildCheck().stagedir, 
            'Samples', 'concurrentKernels', 'concurrentKernels') 



