# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class CudaGdbCheck(rfm.RegressionTest):
    def __init__(self):
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'arolla:cn', 'tsa:cn']
        self.num_gpus_per_node = 1
        self.num_tasks_per_node = 1
        self.sourcesdir = 'src/Cuda'
        self.executable = 'cuda-gdb'
        self.executable_opts = ['-x .in.cudagdb ./cuda_gdb_check']
        # unload xalt to avoid runtime error:
        self.prerun_cmds = ['unset LD_PRELOAD']
        if self.current_system.name in ['arolla', 'tsa']:
            self.exclusive_access = True
            self.modules = ['cuda/10.1.243']
            nvidia_sm = '70'
        else:
            self.modules = ['craype-accel-nvidia60', 'cdt-cuda']
            nvidia_sm = '60'

        self.build_system = 'Make'
        self.build_system.makefile = 'Makefile_cuda_gdb'
        self.build_system.cflags = ['-g', '-D_CSCS_ITMAX=1', '-DUSE_MPI',
                                    '-fopenmp']
        self.build_system.cxxflags = ['-g', '-G', '-arch=sm_%s' % nvidia_sm]
        self.build_system.ldflags = ['-g', '-fopenmp', '-lstdc++']

        if self.current_system.name in ['arolla', 'tsa']:
            self.build_system.ldflags += ['-L$EBROOTCUDA/lib64',
                                          '-lcudart', '-lm']

        self.sanity_patterns = sn.all([
            sn.assert_found(r'^Breakpoint 1 at .*: file ', self.stdout),
            sn.assert_found(r'_jacobi-cuda-kernel.cu, line 59\.', self.stdout),
            sn.assert_found(r'^\(cuda-gdb\) quit', self.stdout),
            sn.assert_lt(sn.abs(sn.extractsingle(
                r'\$1\s+=\s+(?P<result>\S+)', self.stdout,
                'result', float)), 1e-5)
        ])

        self.maintainers = ['MKr', 'JG']
        self.tags = {'production', 'craype'}
