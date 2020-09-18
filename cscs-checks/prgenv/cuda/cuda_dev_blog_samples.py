# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class CudaAwareMPICheck(rfm.CompileOnlyRegressionTest):
    def __init__(self):
        super().__init__()
        self.descr = 'Cuda-aware MPI test from the Parallel Forall repo by NVIDIA.'
        self.sourcesdir = 'https://github.com/NVIDIA-developer-blog/code-samples.git'
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

        if self.current_system.name == 'kesch':
            self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu']
        elif self.current_system.name in ['arolla', 'tsa','ault']:
            self.valid_prog_environs = ['PrgEnv-gnu']

        self.sanity_patterns = sn.assert_found(r'Finished building CUDA samples', self.stdout)
        self.nvidia_sm = '60'
        self.num_tasks = 2
        if self.current_system.name == 'kesch':
            self.exclusive_access = True
            self.nvidia_sm = '37'
        elif self.current_system.name in ['arolla', 'tsa', 'ault']:
            self.exclusive_access = True
            self.nvidia_sm = '70'

        self.prebuild_cmds = ['cd posts/cuda-aware-mpi-example/src']
        self.build_system = 'Make'
        self.build_system.options = ['GENCODE_FLAGS="-gencode arch=compute_%s,code=sm_%s"'  % (self.nvidia_sm, self.nvidia_sm)]
        self.postbuild_cmds = ['ls ../bin']
        self.sanity_patterns = sn.assert_found(r'jacobi_cuda_aware_mpi', self.stdout)


@rfm.simple_test
class CudaAwareMPIOneNodeCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        super().__init__() 
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

        if self.current_system.name == 'kesch':
            self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu']
        elif self.current_system.name in ['arolla', 'tsa','ault']:
            self.valid_prog_environs = ['PrgEnv-gnu']

        self.partition_num_gpus_per_node = {
            'daint:gpu':      1, 
            'dom:gpu':        1, 
            'kesh:cn':        2, 
            'tiger:gpu':      2,
            'arolla:cn':      2,
            'tsa:cn':         2,
            'ault:amdv100':   2, 
            'ault:intelv100': 4
        }

        # Define a minimum of 2 tasks (see below set_num_gpu_per_node) 
        self.min_num_tasks = 2
        self.executable = '../bin/jacobi_cuda_aware_mpi'
        self.depends_on('CudaAwareMPICheck')

    @rfm.run_before('run')
    def set_num_gpus_per_node(self):
        if self.current_partition.fullname in self.partition_num_gpus_per_node:
            self.num_gpus_per_node = self.partition_num_gpus_per_node.get(self.current_partition.fullname)
        else:
            self.num_gpus_per_node = 1

        if self.num_gpus_per_node < self.num_tasks:
            self.variables = {'CRAY_CUDA_MPS': '1'}
 
        if self.num_gpus_per_node < self.min_num_tasks:
            self.num_tasks = self.min_num_tasks
            self.num_tasks_per_node = self.min_num_tasks
        else:
            self.num_tasks = self.num_gpus_per_node
            self.num_tasks_per_node = self.num_tasks

    @rfm.require_deps
    def set_executable(self, CudaAwareMPICheck):
        self.executable = os.path.join(
            CudaAwareMPICheck().stagedir,
            'posts','cuda-aware-mpi-example',
            'bin', 'jacobi_cuda_aware_mpi'
        )


@rfm.simple_test
class CudaAwareMPITwoNodesCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        super().__init__() 
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

        if self.current_system.name == 'kesch':
            self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu']
        elif self.current_system.name in ['arolla', 'tsa','ault']:
            self.valid_prog_environs = ['PrgEnv-gnu']

        # Run the case across two nodes
        self.num_tasks = 2
        self.num_tasks_per_node = 1
        self.num_gpus_per_node = 1
        self.executable = '../bin/jacobi_cuda_aware_mpi'
        self.depends_on('CudaAwareMPICheck')

    @rfm.require_deps
    def set_executable(self, CudaAwareMPICheck):
        self.executable = os.path.join(
            CudaAwareMPICheck().stagedir,
            'posts','cuda-aware-mpi-example',
            'bin', 'jacobi_cuda_aware_mpi'
        )
 
##@rfm.required_version('>=2.14')
##@rfm.simple_test
##class CudaSimpleMPICheck(CudaParForallCheck):
##    def __init__(self):
##        super().__init__()
##        self.descr = 'Simple example demonstrating how to use MPI with CUDA'
##        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
##                                       'CUDA', 'simplempi')
##        self.executable = './simplempi'
##        self.num_tasks = 2
##        self.num_tasks_per_node = 2
##        self.sanity_patterns = sn.assert_found(r'Result = PASS', self.stdout)
##        if self.current_system.name == 'kesch':
##            self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu']
##            self.variables = {'G2G': '0'}
##            self.num_gpus_per_node = 2
##        elif self.current_system.name in ['arolla', 'tsa','ault']:
##            self.valid_prog_environs = ['PrgEnv-gnu']
##            self.num_gpus_per_node = 2
##        else:
##            self.variables = {'CRAY_CUDA_MPS': '1'}
##
##        self.build_system = 'Make'
##        self.build_system.cxxflags = ['-I.', '-ccbin g++', '-m64',
##                                      '-arch=sm_%s' % self.nvidia_sm]
##        self.build_system.ldflags = ['-lcublas']
##

