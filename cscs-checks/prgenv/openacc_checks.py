# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.osext as osext
import reframe.utility.sanity as sn


@rfm.parameterized_test(['mpi'], ['nompi'])
class OpenACCFortranCheck(rfm.RegressionTest):
    def __init__(self, variant):
        if variant == 'nompi':
            self.num_tasks = 1
        else:
            self.num_tasks = 2

        self.valid_systems = ['daint:gpu', 'dom:gpu', 'arolla:cn', 'tsa:cn']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-pgi']
        self.sourcesdir = 'src/openacc'
        if self.num_tasks == 1:
            self.sourcepath = 'vecAdd_openacc_nompi.f90'
            if self.current_system.name in ['arolla', 'tsa']:
                self.valid_prog_environs = ['PrgEnv-pgi-nompi']
        else:
            self.sourcepath = 'vecAdd_openacc_mpi.f90'

        if self.current_system.name in ['daint', 'dom']:
            self.modules = ['craype-accel-nvidia60']
        elif self.current_system.name in ['arolla', 'tsa']:
            self.exclusive_access = True
            self.variables = {
                'CRAY_ACCEL_TARGET': 'nvidia70',
                'MV2_USE_CUDA': '1'
            }

        self.executable = self.name
        self.build_system = 'SingleSource'
        self.num_gpus_per_node = 1
        self.num_tasks_per_node = 1
        result = sn.extractsingle(r'final result:\s+(?P<result>\d+\.?\d*)',
                                  self.stdout, 'result', float)
        self.sanity_patterns = sn.assert_reference(result, 1., -1e-5, 1e-5)

        self.maintainers = ['TM', 'AJ']
        self.tags = {'production', 'craype'}

    @run_before('compile')
    def setflags(self):
        if self.current_environ.name.startswith('PrgEnv-cray'):
            self.build_system.fflags = ['-hacc', '-hnoomp']
        elif self.current_environ.name.startswith('PrgEnv-pgi'):
            if self.current_system.name in ['daint', 'dom']:
                self.build_system.fflags = ['-acc', '-ta=tesla:cc60']
            elif self.current_system.name in ['arolla', 'tsa']:
                self.build_system.fflags = ['-acc', '-ta=tesla:cc70']

    @run_before('compile')
    def cdt2008_pgi_workaround(self):
        cdt = osext.cray_cdt_version()
        if not cdt:
            return

        if (self.current_environ.name == 'PrgEnv-pgi' and cdt == '20.08'):
            self.variables.update({'CUDA_HOME': '$CUDATOOLKIT_HOME'})
