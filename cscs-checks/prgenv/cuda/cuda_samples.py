# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.os_ext as osx


class CudaSamples(rfm.RegressionTest):
    def __init__(self):
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'tiger:gpu',
                              'arolla:cn', 'tsa:cn',
                              'ault:amdv100', 'ault:intelv100']
        if self.current_system.name in ['arolla', 'tsa']:
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

        if self.current_system.name in ['arolla', 'tsa', 'ault']:
            self.exclusive_access = True
            self.nvidia_sm = '70'
        else:
            self.nvidia_sm = '60'
            self.modules = ['cudatoolkit']

        self.sourcesdir = 'https://github.com/NVIDIA/cuda-samples.git'
        self.build_system = 'Make'
        self.build_system.options = [f'SMS="{self.nvidia_sm}"',
                                     f'CUDA_PATH=$CUDA_HOME']
        self.prebuild_cmds = ['git checkout v11.0']
        self.maintainers = ['JO']
        self.tags = {'production', 'external_resosurces'}

    @rfm.run_before('compile')
    def cdt2008_pgi_workaround(self):
        if (self.current_environ.name == 'PrgEnv-pgi' and
            osx.cray_cdt_version() == '20.08' and
            self.current_system.name in ['daint', 'dom']):
            self.variables['CUDA_HOME'] = '$CUDATOOLKIT_HOME'


@rfm.simple_test
class CudaDeviceQueryCheck(CudaSamples):
    def __init__(self):
        super().__init__()
        self.descr = 'CUDA deviceQuery test.'
        self.num_tasks = 1
        self.prebuild_cmds += ['cd Samples/deviceQuery']
        self.executable = 'Samples/{0}/{0}'.format('deviceQuery')
        self.sanity_patterns = sn.assert_found(
            r'Result = PASS',
            self.stdout)


@rfm.simple_test
class CudaConcurrentKernelsCheck(CudaSamples):
    def __init__(self):
        super().__init__()
        self.descr = 'CUDA concurrentKernels test'
        self.executable = 'Samples/{0}/{0}'.format('concurrentKernels')
        self.prebuild_cmds += ['cd Samples/concurrentKernels']
        self.sanity_patterns = sn.assert_found(r'Test passed', self.stdout)


@rfm.simple_test
class CudaMatrixMultCublasCheck(CudaSamples):
    def __init__(self):
        super().__init__()
        self.descr = 'CUDA simpleCUBLAS test'
        self.executable = 'Samples/{0}/{0}'.format('simpleCUBLAS')
        self.prebuild_cmds += ['cd Samples/simpleCUBLAS']
        self.sanity_patterns = sn.assert_found(r'test passed', self.stdout)


@rfm.simple_test
class CudaBandwidthCheck(CudaSamples):
    def __init__(self):
        super().__init__()
        self.descr = 'CUDA bandwidthTest test'
        self.executable = 'Samples/{0}/{0}'.format('bandwidthTest')
        self.prebuild_cmds += ['cd Samples/bandwidthTest']
        self.sanity_patterns = sn.assert_found(r'Result = PASS', self.stdout)


@rfm.simple_test
class CudaGraphsCGCheck(CudaSamples):
    def __init__(self):
        super().__init__()
        self.descr = 'CUDA conjugateGradientCudaGraphs test'
        self.executable = 'Samples/{0}/{0}'.format(
            'conjugateGradientCudaGraphs'
        )
        self.prebuild_cmds += ['cd Samples/conjugateGradientCudaGraphs']
        self.sanity_patterns = sn.assert_found(
            r'Test Summary:  Error amount = 0.00000', self.stdout)
