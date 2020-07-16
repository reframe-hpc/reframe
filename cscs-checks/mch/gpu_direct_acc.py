# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import reframe as rfm
import reframe.utility.os_ext as os_ext
import reframe.utility.sanity as sn


@rfm.required_version('>=2.14')
@rfm.simple_test
class GpuDirectAccCheck(rfm.RegressionTest):
    def __init__(self):
        self.descr = 'tests gpu-direct for Fortran OpenACC'
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'kesch:cn', 'tiger:gpu',
                              'arolla:cn', 'tsa:cn']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-pgi']
        if self.current_system.name in ['daint', 'dom', 'tiger']:
            self.modules = ['craype-accel-nvidia60']
            self.variables = {
                'MPICH_RDMA_ENABLED_CUDA': '1',
            }

            if self.current_system.name in ['tiger']:
                craypath = '%s:$PATH' % os.environ['CRAY_BINUTILS_BIN']
                self.variables['PATH'] = craypath

            self.num_tasks = 2
            self.num_gpus_per_node = 1
            self.num_tasks_per_node = 1
        elif self.current_system.name == 'kesch':
            self.exclusive_access = True
            self.modules = ['cudatoolkit/8.0.61']
            self.variables = {
                'CRAY_ACCEL_TARGET': 'nvidia35',
                'MV2_USE_CUDA': '1',
                'G2G': '1'
            }
            self.num_tasks = 8
            self.num_gpus_per_node = 8
            self.num_tasks_per_node = 8
        elif self.current_system.name in ['arolla', 'tsa']:
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
            elif self.current_system.name == 'kesch':
                self.build_system.fflags += ['-ta=tesla:cc35']
            elif self.current_system.name in ['arolla', 'tsa']:
                self.build_system.fflags += ['-ta=tesla:cc70']

    @rfm.run_before('compile')
    def cray_linker_workaround(self):
        # NOTE: Workaround for using CCE < 9.1 in CLE7.UP01.PS03 and above
        # See Patch Set README.txt for more details.
        if (self.current_system.name == 'dom' and
            self.current_environ.name == 'PrgEnv-cray'):
            self.variables['LINKER_X86_64'] = '/usr/bin/ld'

    @rfm.run_before('compile')
    def cdt2006_pgi_workaround(self):
        cdt = os_ext.cray_cdt_version()
        if not cdt:
            return

        if (self.current_environ.name == 'PrgEnv-pgi' and cdt == '20.06'):
            self.variables['CUDA_HOME'] = '$CUDATOOLKIT_HOME'
