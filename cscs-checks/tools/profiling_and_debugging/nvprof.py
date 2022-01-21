# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class NvprofCheck(rfm.RegressionTest):
    def __init__(self):
        self.descr = 'Checks the nvprof tool'
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'arolla:cn', 'tsa:cn']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.num_gpus_per_node = 1
        self.num_tasks_per_node = 1
        self.sourcesdir = 'src/Cuda'
        self.executable = 'nvprof'
        self.target_executable = './jacobi'
        self.build_system = 'Make'
        self.build_system.cflags = [
            '-g', '-D_CSCS_ITMAX=100', '-DOMP_MEMLOCALITY', '-DUSE_MPI',
            '-DEVS_PER_NODE=1', '-fopenmp', '-std=c99'
        ]
        self.build_system.cxxflags = ['-g', '-G']
        self.build_system.ldflags = ['-g', '-fopenmp', '-std=c99', '-lstdc++']
        if self.current_system.name in ['arolla', 'tsa']:
            self.exclusive_access = True
            self.modules = ['cuda/10.1.243']
            self.build_system.ldflags = ['-lstdc++', '-lm',
                                         '-L$EBROOTCUDA/lib64', '-lcudart']
        else:
            self.modules = ['craype-accel-nvidia60', 'cdt-cuda']

        self.executable_opts = [self.target_executable]
        # Reminder: NVreg_RestrictProfilingToAdminUsers=0 (RFC-16) needed since
        # cuda/10.1
        self.postrun_cmds = ['cat /etc/modprobe.d/nvidia.conf']
        self.sanity_patterns = sn.all([
            sn.assert_found(f'Profiling application: {self.target_executable}',
                            self.stderr),
            sn.assert_found('[CUDA memcpy HtoD]', self.stderr),
            sn.assert_found('[CUDA memcpy DtoH]', self.stderr),
            sn.assert_found(r'\s+100(\s+\S+){3}\s+jacobi_kernel', self.stderr),
        ])
        self.maintainers = ['JG', 'SK']
        self.tags = {'production', 'craype', 'maintenance'}
