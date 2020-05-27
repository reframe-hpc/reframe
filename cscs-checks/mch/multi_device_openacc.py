# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class MultiDeviceOpenaccTest(rfm.RegressionTest):
    def __init__(self):
        self.descr = (
            'Allocate one accelerator per MPI task using OpenACC on '
            'multi-device nodes with additional CUDA, MPI, and C++ calls'
        )
        self.valid_systems = ['arolla:cn', 'tsa:cn', 'kesch:cn']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-pgi']
        self.build_system = 'Make'
        self.build_system.makefile = 'Makefile.multi_device_openacc'
        self.build_system.fflags = ['-O2']
        if self.current_system.name == 'kesch':
            self.exclusive_access = True
            self.modules = ['cudatoolkit/8.0.61']
            self.num_tasks = 9
            self.num_tasks_per_node = 9
            self.num_gpus_per_node = 8
            self.build_system.options = ['NVCC_FLAGS="-arch=compute_37"']
        elif self.current_system.name in ['arolla', 'tsa']:
            self.exclusive_access = True
            self.modules = ['cuda/10.1.243']
            self.num_tasks = 9
            self.num_tasks_per_node = 9
            self.num_gpus_per_node = 8
            self.build_system.options = ['NVCC_FLAGS="-arch=compute_70"']

        self.executable = 'multi_device_openacc'
        self.sanity_patterns = sn.assert_found(r'Test\sResult\s*:\s+OK',
                                               self.stdout)
        self.maintainers = ['LM', 'AJ']
        self.tags = {'production', 'mch'}

    @rfm.run_before('compile')
    def setflags(self):
        if self.current_environ.name.startswith('PrgEnv-pgi'):
            self.build_system.fflags += ['-acc']
            if self.current_system.name == 'kesch':
                self.build_system.fflags += ['-ta=tesla,cc35,cuda8.0']
                self.build_system.ldflags = [
                    '-acc', '-ta:tesla:cc35,cuda8.0', '-lstdc++',
                    '-L/global/opt/nvidia/cudatoolkit/8.0.61/lib64',
                    '-lcublas', '-lcudart'
                ]
            elif self.current_system.name in ['arolla', 'tsa']:
                self.build_system.fflags += ['-ta=tesla,cc70,cuda10.1']
                self.build_system.ldflags = [
                    '-acc', '-ta:tesla:cc70,cuda10.1', '-lstdc++',
                    '-L$EBROOTCUDA/lib64', '-lcublas', '-lcudart'
                ]
        elif self.current_environ.name.startswith('PrgEnv-cray'):
            self.build_system.fflags += ['-DCRAY', '-hacc', '-hnoomp']
            self.variables = {
                'CRAY_ACCEL_TARGET': 'nvidia35',
                'MV2_USE_CUDA': '1'
            }
