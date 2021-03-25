# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.osext as osext
import reframe.utility.sanity as sn


@rfm.simple_test
class GpuDirectAccCheck(rfm.RegressionTest):
    def __init__(self):
        self.descr = 'tests gpu-direct for Fortran OpenACC'
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'arolla:cn', 'tsa:cn']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-pgi']
        if self.current_system.name in ['daint', 'dom']:
            self.modules = ['craype-accel-nvidia60']
            self.variables = {
                'MPICH_RDMA_ENABLED_CUDA': '1',
                'CRAY_CUDA_MPS': '1',
            }
            self.num_tasks = 2
            self.num_gpus_per_node = 1
            self.num_tasks_per_node = 1
            self.num_tasks = 8
            self.num_gpus_per_node = 8
            self.num_tasks_per_node = 8
        if self.current_system.name in ['arolla', 'tsa']:
            self.exclusive_access = True
            self.variables = {
                'G2G': '1'
            }
            self.num_tasks = 8
            self.num_gpus_per_node = 8
            self.num_tasks_per_node = 8

        self.sourcepath = 'gpu_direct_acc.F90'
        self.build_system = 'SingleSource'
        self.prebuild_cmds = ['module list -l']
        self.sanity_patterns = sn.all([
            sn.assert_found(r'GPU with OpenACC', self.stdout),
            sn.assert_found(r'Result :\s+OK', self.stdout)
        ])
        self.launch_options = []
        self.maintainers = ['AJ', 'MKr']
        self.tags = {'production', 'mch', 'craype'}

    @rfm.run_before('compile')
    def setflags(self):
        if self.current_environ.name.startswith('PrgEnv-cray'):
            self.build_system.fflags = ['-hacc', '-hnoomp']
        elif self.current_environ.name.startswith('PrgEnv-pgi'):
            self.build_system.fflags = ['-acc']
            if self.current_system.name in ['daint', 'dom']:
                self.build_system.fflags += ['-ta=tesla:cc60', '-Mnorpath']
            elif self.current_system.name in ['arolla', 'tsa']:
                self.build_system.fflags += ['-ta=tesla:cc70']

    @rfm.run_before('compile')
    def cdt2008_pgi_workaround(self):
        cdt = osext.cray_cdt_version()
        if not cdt:
            return

        if (self.current_environ.name == 'PrgEnv-pgi' and cdt == '20.08'):
            self.variables['CUDA_HOME'] = '$CUDATOOLKIT_HOME'
