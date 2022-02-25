# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class OpenACCFortranCheck(rfm.RegressionTest):
    variant = parameter(['mpi', 'nompi'])
    valid_systems = ['daint:gpu', 'dom:gpu', 'arolla:cn', 'tsa:cn']
    valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-nvidia']
    sourcesdir = 'src/openacc'
    build_system = 'SingleSource'
    num_gpus_per_node = 1
    num_tasks_per_node = 1
    maintainers = ['TM', 'AJ']
    tags = {'production', 'craype'}

    @run_after('init')
    def set_numtasks(self):
        if self.variant == 'nompi':
            self.num_tasks = 1
            self.sourcepath = 'vecAdd_openacc_nompi.f90'
            if self.current_system.name in ['arolla', 'tsa']:
                self.valid_prog_environs = ['PrgEnv-pgi-nompi']
        else:
            self.num_tasks = 2
            self.sourcepath = 'vecAdd_openacc_mpi.f90'

    @run_after('setup')
    def set_executable(self):
        self.executable = self.name

    @run_before('compile')
    def set_variables(self):
        if self.current_system.name in ['arolla', 'tsa']:
            self.exclusive_access = True
            self.variables = {
                'CRAY_ACCEL_TARGET': 'nvidia70',
                'MV2_USE_CUDA': '1'
            }

    @run_before('compile')
    def setflags(self):
        if self.current_environ.name.startswith('PrgEnv-cray'):
            self.build_system.fflags = ['-hacc', '-hnoomp']
        elif (self.current_environ.name.startswith('PrgEnv-pgi') or
              self.current_environ.name == 'PrgEnv-nvidia'):
            if self.current_system.name in ['daint', 'dom']:
                self.build_system.fflags = ['-acc', '-ta=tesla:cc60']
            elif self.current_system.name in ['arolla', 'tsa']:
                self.build_system.fflags = ['-acc', '-ta=tesla:cc70']

    @run_before('sanity')
    def set_sanity(self):
        result = sn.extractsingle(r'final result:\s+(?P<result>\d+\.?\d*)',
                                  self.stdout, 'result', float)
        self.sanity_patterns = sn.assert_reference(result, 1., -1e-5, 1e-5)
