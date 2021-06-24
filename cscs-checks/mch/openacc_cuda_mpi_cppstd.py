# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.osext as osext
import reframe.utility.sanity as sn


@rfm.simple_test
class OpenaccCudaCpp(rfm.RegressionTest):
    def __init__(self):
        self.descr = 'test for OpenACC, CUDA, MPI, and C++'
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'arolla:cn', 'tsa:cn']
        self.valid_prog_environs = ['PrgEnv-cce', 'PrgEnv-cray',
                                    'PrgEnv-pgi', 'PrgEnv-nvidia']
        self.build_system = 'Make'
        self.build_system.fflags = ['-O2']

        if self.current_system.name in ['daint', 'dom']:
            self.num_tasks = 12
            self.num_tasks_per_node = 12
            self.num_gpus_per_node = 1
            self.build_system.options = ['NVCC_FLAGS="-arch=compute_60"']
            self.variables = {
                'MPICH_RDMA_ENABLED_CUDA': '1',
                'CRAY_CUDA_MPS': '1'
            }
        elif self.current_system.name in ['arolla', 'tsa']:
            self.exclusive_access = True
            self.modules = ['cuda/10.1.243']
            self.num_tasks = 8
            self.num_tasks_per_node = 8
            self.num_gpus_per_node = 8
            self.build_system.options = ['NVCC_FLAGS="-arch=compute_70"']
            self.variables = {
                'G2G': '1'
            }

        self.executable = 'openacc_cuda_mpi_cppstd'
        self.sanity_patterns = sn.assert_found(r'Result:\s+OK', self.stdout)
        self.maintainers = ['AJ', 'MKr']
        self.tags = {'production', 'mch', 'craype'}

    @run_before('compile')
    def setflags(self):
        if (not self.current_environ.name.startswith('PrgEnv-nvidia')):
            self.modules = ['craype-accel-nvidia60']
        if self.current_environ.name.startswith('PrgEnv-cray'):
            self.build_system.fflags += ['-hacc', '-hnoomp']

        elif self.current_environ.name.startswith('PrgEnv-nvidia'):
            self.build_system.fflags += ['-acc']
            self.build_system.fflags += ['-ta:tesla:cc60']
            self.build_system.ldflags = [
                '-acc', '-ta:tesla:cc60', '-Mnorpath', '-lstdc++',
                '-Mcuda'
            ]

        elif self.current_environ.name.startswith('PrgEnv-pgi'):
            self.build_system.fflags += ['-acc']
            if self.current_system.name in ['daint', 'dom']:
                self.build_system.fflags += ['-ta:tesla:cc60']
                self.build_system.ldflags = ['-acc', '-ta:tesla:cc60',
                                             '-Mnorpath', '-lstdc++']
            elif self.current_system.name in ['arolla', 'tsa']:
                self.build_system.fflags += ['-ta=tesla,cc70,cuda10.1']
                self.build_system.ldflags = [
                    '-acc', '-ta:tesla:cc70,cuda10.1', '-lstdc++',
                    '-L$EBROOTCUDA/lib64', '-lcublas', '-lcudart'
                ]

        elif self.current_environ.name.startswith('PrgEnv-gnu'):
            self.build_system.ldflags = ['-lstdc++']
            if self.current_system.name in ['arolla', 'tsa']:
                self.build_system.ldflags += [
                    '-L$EBROOTCUDA/lib64', '-lcublas', '-lcudart'
                ]

    @run_before('compile')
    def cdt2006_pgi_workaround(self):
        cdt = osext.cray_cdt_version()
        if not cdt:
            return

        if (self.current_environ.name == 'PrgEnv-pgi' and cdt >= '20.06'):
            self.variables.update({'CUDA_HOME': '$CUDATOOLKIT_HOME'})
