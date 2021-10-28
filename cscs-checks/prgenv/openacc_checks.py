# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.osext as osext
import reframe.utility.sanity as sn


@rfm.simple_test
class OpenACCFortranCheck(rfm.RegressionTest):
    variant = parameter(['mpi', 'nompi'])
    valid_systems = ['daint:gpu', 'dom:gpu', 'arolla:cn', 'tsa:cn']
    valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-pgi', 'PrgEnv-nvidia']
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
    def set_modules(self):
        if (self.current_system.name in ['daint', 'dom'] and
            self.current_environ.name != 'PrgEnv-nvidia'):
            self.modules = ['craype-accel-nvidia60']

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

    # FIXME: PGI 20.x does not support CUDA 11, see case #275674
    @run_before('compile')
    def cudatoolkit_pgi_20x_workaround(self):
        if self.current_system.name in {'daint', 'dom'}:
            cudatoolkit_version = '10.2.89_3.28-2.1__g52c0314'
            self.variables['CUDA_HOME'] = '$CUDATOOLKIT_HOME'
            self.modules += [f'cudatoolkit/{cudatoolkit_version}']

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

    @run_before('compile')
    def cdt2008_pgi_workaround(self):
        cdt = osext.cray_cdt_version()
        if not cdt:
            return

        if (self.current_environ.name == 'PrgEnv-pgi' and cdt == '20.08'):
            self.variables.update({'CUDA_HOME': '$CUDATOOLKIT_HOME'})

    @run_before('sanity')
    def set_sanity(self):
        result = sn.extractsingle(r'final result:\s+(?P<result>\d+\.?\d*)',
                                  self.stdout, 'result', float)
        self.sanity_patterns = sn.assert_reference(result, 1., -1e-5, 1e-5)
