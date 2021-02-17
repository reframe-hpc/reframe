# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.osext as osext
import reframe.utility.sanity as sn


@rfm.simple_test
class DefaultPrgEnvCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.descr = 'Ensure PrgEnv-cray is loaded by default'
        self.valid_prog_environs = ['builtin']
        self.valid_systems = ['daint:login', 'dom:login', 'eiger:login']
        self.executable = 'module'
        self.executable_opts = ['list', '-t']
        self.maintainers = ['TM', 'CB']
        self.tags = {'production', 'craype'}
        self.sanity_patterns = sn.assert_found(r'^PrgEnv-cray', self.stderr)

    @rfm.run_before('setup')
    def setup_eiger(self):
        if self.current_system.name == 'eiger':
            self.executable_opts = ['list']
            self.sanity_patterns = sn.assert_found(r'1\) cpe-cray',
                                                   self.stderr)


@rfm.simple_test
class EnvironmentCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.descr = 'Ensure programming environment is loaded correctly'
        self.valid_systems = ['daint:login', 'dom:login', 'eiger:login']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu', 'PrgEnv-pgi',
                                    'PrgEnv-intel', 'PrgEnv-aocc']
        self.executable = 'module'
        self.executable_opts = ['list', '-t']
        self.sanity_patterns = sn.assert_found(self.env_module_patt,
                                               self.stderr)
        self.maintainers = ['TM', 'CB']
        self.tags = {'production', 'craype'}

    @rfm.run_before('setup')
    def setup_eiger(self):
        if self.current_system.name == 'eiger':
            self.executable_opts = ['list']

    @property
    @sn.sanity_function
    def env_module_patt(self):
        # NOTE: On eiger, the first module of each programming environment,
        # follows the 'cpe-<name>' pattern where <name> corresponds to the
        # 'PrgEnv-<name>' used.
        if self.current_system.name == 'eiger':
            return rf'1\) cpe-{self.current_environ.name[7:]}'

        return rf'^{self.current_environ.name}'


class CrayVariablesCheckMixin:
    def __init__(self, module_name):
        self.descr = 'Check for standard Cray variables'
        self.valid_prog_environs = ['builtin']
        self.executable = 'module'
        self.executable_opts = ['show', module_name]
        envvar_prefix = module_name.upper().replace('-', '_')
        self.sanity_patterns = sn.all([
            sn.assert_found(f'{envvar_prefix}_PREFIX', self.stderr),
            sn.assert_found(f'{envvar_prefix}_VERSION', self.stderr)
        ])
        self.tags = {'production', 'craype'}


@rfm.parameterized_test(['cray-fftw'], ['cray-hdf5'], ['cray-hdf5-parallel'],
                        ['cray-libsci'], ['cray-mpich'], ['cray-netcdf'],
                        ['cray-netcdf-hdf5parallel'], ['cray-petsc'],
                        ['cray-petsc-complex-64'], ['cray-python'],
                        ['cray-R'], ['cray-tpsl'], ['cray-tpsl-64'],
                        ['cudatoolkit'], ['gcc'], ['papi'], ['pmi'])
class CrayVariablesCheck(rfm.RunOnlyRegressionTest, CrayVariablesCheckMixin):
    def __init__(self, module_name):
        CrayVariablesCheckMixin.__init__(self, module_name)
        self.valid_systems = ['daint:login', 'dom:login']

        # FIXME: These modules should be fixed in later releases,
        # while gcc was fixed in 20.11

        cdt = osext.cray_cdt_version()
        if ((cdt and cdt <= '20.11' and
             module_name in ['cray-petsc-complex',
                             'cray-petsc-complex-64',
                             'cudatoolkit']) or
            (cdt and cdt < '20.11' and module_name == 'gcc')):
            self.valid_systems = []

        self.maintainers = ['EK', 'VH']


@rfm.parameterized_test(['cray-fftw'], ['cray-hdf5'], ['cray-hdf5-parallel'],
                        ['cray-libsci'], ['cray-mpich'], ['cray-openshmemx'],
                        ['cray-parallel-netcdf'], ['cray-pmi'],
                        ['cray-python'], ['cray-R'], ['gcc'], ['papi'])
class CrayVariablesCheckEiger(rfm.RunOnlyRegressionTest,
                              CrayVariablesCheckMixin):
    def __init__(self, module_name):
        CrayVariablesCheckMixin.__init__(self, module_name)
        self.valid_systems = ['eiger:login']

        # FIXME: These modules should be fixed in later releases

        if module_name in {'cray-fftw', 'cray-python', 'cray-mpich'}:
            self.valid_systems = []

        self.maintainers = ['TM']
