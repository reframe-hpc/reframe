# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class CudaAwareMPICheck(rfm.CompileOnlyRegressionTest):
    def __init__(self):
        self.descr = 'Cuda-aware MPI test from the NVIDIA repo.'
        self.sourcesdir = ('https://github.com/NVIDIA-developer-blog/'
                           'code-samples.git')
        self.valid_systems = ['daint:gpu', 'dom:gpu',
                              'arolla:cn', 'tsa:cn',
                              'ault:amdv100', 'ault:intelv100']
        if self.current_system.name in ['arolla', 'tsa']:
            self.valid_prog_environs = ['PrgEnv-gnu']
            self.modules = ['cuda/10.1.243']
        elif self.current_system.name in ['ault']:
            self.valid_prog_environs = ['PrgEnv-gnu']
            self.modules = ['cuda/11.0']
        else:
            self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                        'PrgEnv-pgi']
            self.modules = ['craype-accel-nvidia60', 'cdt-cuda']

        self.sanity_patterns = sn.assert_found(r'Finished building'
                                               r' CUDA samples', self.stdout)
        if self.current_system.name in ['arolla', 'tsa', 'ault']:
            self.exclusive_access = True
            nvidia_sm = '70'
        else:
            nvidia_sm = '60'

        self.prebuild_cmds += ['cd posts/cuda-aware-mpi-example/src']
        gcd_flgs = '-gencode arch=compute_{0},code=sm_{0}'.format(nvidia_sm)
        self.build_system = 'Make'
        self.build_system.options = ['CUDA_INSTALL_PATH=$CUDA_HOME',
                                     'MPI_HOME=$CRAY_MPICH_PREFIX',
                                     'GENCODE_FLAGS="%s"' % (gcd_flgs)]

        self.postbuild_cmds = ['ls ../bin']
        self.sanity_patterns = sn.assert_found(r'jacobi_cuda_aware_mpi',
                                               self.stdout)
        self.maintainers = ['JO']
        self.tags = {'production', 'scs'}

    @run_before('compile')
    def set_compilers(self):
        if self.current_environ.name == 'PrgEnv-pgi':
            self.build_system.cflags = ['-std=c99', ' -O3']

        self.build_system.options += [
            'MPICC="%s"' % self.current_environ.cc,
            'MPILD="%s"' % self.current_environ.cxx
        ]


class CudaAwareMPIRuns(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.depends_on('CudaAwareMPICheck')
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'arolla:cn', 'tsa:cn',
                              'ault:amdv100', 'ault:intelv100']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu']
        if self.current_system.name in ['arolla', 'tsa', 'daint', 'dom']:
            self.valid_prog_environs += ['PrgEnv-pgi']
        elif self.current_system.name in ['ault']:
            self.valid_prog_environs = ['PrgEnv-gnu']

        if self.current_system.name in ['arolla', 'tsa', 'ault']:
            self.valid_prog_environs = ['PrgEnv-gnu']
        else:
            self.modules = ['craype-accel-nvidia60', 'cdt-cuda']

        self.prerun_cmds = ['export MPICH_RDMA_ENABLED_CUDA=1']
        self.sanity_patterns = sn.assert_found(r'Stopped after 1000 iterations'
                                               r' with residue 0.00024',
                                               self.stdout)

    @require_deps
    def set_executable(self, CudaAwareMPICheck):
        self.executable = os.path.join(
            CudaAwareMPICheck().stagedir,
            'posts', 'cuda-aware-mpi-example',
            'bin', 'jacobi_cuda_aware_mpi'
        )


@rfm.simple_test
class CudaAwareMPIOneNodeCheck(CudaAwareMPIRuns):
    def __init__(self):
        super().__init__()
        self.partition_num_gpus_per_node = {
            'daint:gpu':      1,
            'dom:gpu':        1,
            'arolla:cn':      2,
            'tsa:cn':         2,
            'ault:amdv100':   2,
            'ault:intelv100': 4
        }

        self.prerun_cmds += ['export CRAY_CUDA_MPS=1']

    @run_before('run')
    def set_num_gpus_per_node(self):
        cp = self.current_partition.fullname
        self.num_gpus_per_node = self.partition_num_gpus_per_node.get(cp, 1)
        self.num_tasks = 2 * self.num_gpus_per_node
        self.num_tasks_per_node = self.num_tasks
        self.executable_opts = [f'-t {self.num_tasks/2} 2']


@rfm.simple_test
class CudaAwareMPITwoNodesCheck(CudaAwareMPIRuns):
    def __init__(self):
        super().__init__()
        # Run the case across two nodes
        self.num_tasks = 2
        self.num_tasks_per_node = 1
        self.num_gpus_per_node = 1
        self.executable_opts = ['-t %d 1' % self.num_tasks]
