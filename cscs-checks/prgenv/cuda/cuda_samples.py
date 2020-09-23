# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import reframe as rfm
import reframe.utility.sanity as sn

class CudaSamples(rfm.RegressionTest):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'kesch:cn', 'tiger:gpu',
                              'arolla:cn', 'tsa:cn', 'ault:amdv100', 'ault:intelv100']
        if self.current_system.name == 'kesch':
            self.valid_prog_environs += ['PrgEnv-cray-nompi',
                                         'PrgEnv-gnu-nompi']
            self.modules = ['cudatoolkit/8.0.61']
        elif self.current_system.name in ['arolla', 'tsa']:
            self.valid_prog_environs += ['PrgEnv-pgi',
                                         'PrgEnv-gnu-nompi',
                                         'PrgEnv-pgi-nompi']
            self.modules = ['cuda/10.1.243']
        elif self.current_system.name in ['ault']:
            self.valid_prog_environs = ['PrgEnv-gnu']
            self.modules = ['cuda/11.0']
        else:
            self.valid_prog_environs = ['PrgEnv-cray', 
                                        'PrgEnv-gnu', 
                                        'PrgEnv-pgi']
            self.modules = ['craype-accel-nvidia60']

        if self.current_system.name == 'kesch':
            self.exclusive_access = True
            self.nvidia_sm = '37'
        elif self.current_system.name in ['arolla', 'tsa', 'ault']:
            self.exclusive_access = True
            self.nvidia_sm = '70'
        else:
            self.nvidia_sm = '60'
            self.modules = ['cudatoolkit']

        self.sourcesdir = None
        self.build_system = 'Make'
        if self.current_system.name in ['daint']:
            self.prebuild_cmds = ['export CUDA_HOME=$CUDATOOLKIT_HOME']
        else:
            self.prebuild_cmds = []

        self.build_system.options = ['SMS="%s"' % self.nvidia_sm, 'CUDA_PATH=$CUDA_HOME']
        self.maintainers = ['JO']
        self.tags = {'production', 'external_resosurces'} 


@rfm.required_version('>=2.14')
@rfm.simple_test
class CudaDeviceQueryCheck(CudaSamples):
    def __init__(self):
        super().__init__()
        self.descr = 'CUDA deviceQuery test.'
        self.sourcesdir = 'https://github.com/NVIDIA/cuda-samples.git'
        self.num_tasks = 1
        self.prebuild_cmds += ['git checkout v11.0',
                               'cd Samples/deviceQuery'
                              ]
        self.executable = 'Samples/deviceQuery/deviceQuery'
        self.sanity_patterns = sn.assert_found(
            r'Result = PASS',
            self.stdout)


class DependentCudaSamples(CudaSamples):
    def __init__(self):
        super().__init__()
        self.depends_on('CudaDeviceQueryCheck')


@rfm.required_version('>=2.14')
@rfm.simple_test
class CudaConcurrentKernelsCheck(DependentCudaSamples):
    def __init__(self):
        super().__init__()
        self.descr = 'CUDA concurrentKernels test'
        self.sanity_patterns = sn.assert_found(
            r'Test passed',
            self.stdout)

    @rfm.require_deps
    def set_prebuild_cmds(self, CudaDeviceQueryCheck):
        self.prebuild_cmds += ['cd %s' % os.path.join(
            CudaDeviceQueryCheck().stagedir,
            'Samples', 'concurrentKernels')] 

    @rfm.require_deps
    def set_executable(self,CudaDeviceQueryCheck):
        self.executable = os.path.join(
            CudaDeviceQueryCheck().stagedir, 
            'Samples', 'concurrentKernels', 'concurrentKernels') 


@rfm.required_version('>=2.14')
@rfm.simple_test
class CudaMatrixMultCublasCheck(DependentCudaSamples):
    def __init__(self):
        super().__init__()
        self.descr = 'CUDA simpleCUBLAS test'
        self.sanity_patterns = sn.assert_found(
            r'test passed',
            self.stdout)

    @rfm.require_deps
    def set_prebuild_cmds(self, CudaDeviceQueryCheck):
        self.prebuild_cmds += ['cd %s' % os.path.join(
            CudaDeviceQueryCheck().stagedir, 
            'Samples', 'simpleCUBLAS')] 

    @rfm.require_deps
    def set_executable(self, CudaDeviceQueryCheck):
        self.executable = os.path.join(
            CudaDeviceQueryCheck().stagedir, 
            'Samples', 'simpleCUBLAS', 'simpleCUBLAS') 


@rfm.required_version('>=2.14')
@rfm.simple_test
class CudaBandwidthCheck(DependentCudaSamples):
    def __init__(self):
        super().__init__()
        self.descr = 'CUDA simpleCUBLAS test'
        self.sanity_patterns = sn.assert_found(
            r'Result = PASS',
            self.stdout)

    @rfm.require_deps
    def set_prebuild_cmds(self, CudaDeviceQueryCheck):
        self.prebuild_cmds += ['cd %s' % os.path.join(
            CudaDeviceQueryCheck().stagedir, 
            'Samples', 'bandwidthTest')] 

    @rfm.require_deps
    def set_executable(self, CudaDeviceQueryCheck):
        self.executable = os.path.join(
            CudaDeviceQueryCheck().stagedir, 
            'Samples', 'bandwidthTest', 'bandwidthTest') 


@rfm.required_version('>=2.14')
@rfm.simple_test
class CudaGraphsCGCheck(DependentCudaSamples):
    def __init__(self):
        super().__init__()
        self.descr = 'CUDA simpleCUBLAS test'
        self.sanity_patterns = sn.assert_found(
            r'Test Summary:  Error amount = 0.00000',
            self.stdout)

    @rfm.require_deps
    def set_prebuild_cmds(self, CudaDeviceQueryCheck):
        self.prebuild_cmds += ['cd %s' % os.path.join(
            CudaDeviceQueryCheck().stagedir, 
            'Samples', 'conjugateGradientCudaGraphs')] 

    @rfm.require_deps
    def set_executable(self, CudaDeviceQueryCheck):
        self.executable = os.path.join(
            CudaDeviceQueryCheck().stagedir, 
            'Samples', 'conjugateGradientCudaGraphs', 'conjugateGradientCudaGraphs') 

