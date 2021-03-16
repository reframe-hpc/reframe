# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.osext as osext


@rfm.parameterized_test(['deviceQuery'], ['concurrentKernels'],
                        ['simpleCUBLAS'], ['bandwidthTest'],
                        ['conjugateGradientCudaGraphs'])
class CudaSamplesTest(rfm.RegressionTest):
    def __init__(self, sample):
        self.descr = f'CUDA {sample} test'
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'arolla:cn', 'tsa:cn',
                              'ault:amdv100', 'ault:intelv100']
        if self.current_system.name in ['arolla', 'tsa']:
            self.valid_prog_environs = ['PrgEnv-pgi',
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

        output_patterns = {
            'deviceQuery': r'Result = PASS',
            'concurrentKernels': r'Test passed',
            'simpleCUBLAS': r'test passed',
            'bandwidthTest': r'Result = PASS',
            'conjugateGradientCudaGraphs':
                r'Test Summary:  Error amount = 0.00000'
        }

        self.sourcesdir = 'https://github.com/NVIDIA/cuda-samples.git'
        self.build_system = 'Make'
        self.build_system.options = [f'SMS="{self.nvidia_sm}"',
                                     f'CUDA_PATH=$CUDA_HOME']
        self.prebuild_cmds = [f'git checkout v11.0', f'cd Samples/{sample}']
        self.executable = f'Samples/{sample}/{sample}'
        self.sanity_patterns = sn.assert_found(
            output_patterns[sample], self.stdout
        )
        self.maintainers = ['JO']
        self.tags = {'production'}

    @rfm.run_before('compile')
    def cdt2008_pgi_workaround(self):
        if (self.current_environ.name == 'PrgEnv-pgi' and
            osext.cray_cdt_version() == '20.08' and
            self.current_system.name in ['daint', 'dom']):
            self.variables['CUDA_HOME'] = '$CUDATOOLKIT_HOME'

    @rfm.run_before('compile')
    def dom_set_cuda_cdt(self):
        if self.current_system.name == 'dom':
            self.modules += ['cdt-cuda']
