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
        self.maintainers = ['TM', 'CB']
        self.tags = {'production', 'craype'}
        self.sanity_patterns = sn.assert_found(r'^PrgEnv-cray', self.stderr)

        if self.current_system.name == 'eiger':
            self.executable_opts = ['list']
            prgenv_patt = r'1\) cpe-cray'
        else:
            self.executable_opts = ['list', '-t']
            prgenv_patt = r'^PrgEnv-cray'

        self.sanity_patterns = sn.assert_found(prgenv_patt, self.stderr)


@rfm.simple_test
class EnvironmentCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.descr = 'Ensure programming environment is loaded correctly'
        self.valid_systems = ['daint:login', 'dom:login', 'eiger:login']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu', 'PrgEnv-pgi',
                                    'PrgEnv-intel', 'PrgEnv-aocc']
        self.executable = 'module'
        if self.current_system.name == 'eiger':
            self.executable_opts = ['list']
        else:
            self.executable_opts = ['list', '-t']

        self.maintainers = ['TM', 'CB']
        self.tags = {'production', 'craype'}

    @rfm.run_before('sanity')
    def set_sanity(self):
        # NOTE: On eiger, the first module of each programming environment,
        # follows the 'cpe-<name>' pattern where <name> corresponds to the
        # 'PrgEnv-<name>' used.
        if self.current_system.name == 'eiger':
            module_patt = rf'1\) cpe-{self.current_environ.name[7:]}'
        else:
            module_patt = rf'^{self.current_environ.name}'

        self.sanity_patterns = sn.assert_found(module_patt, self.stderr)


class CrayVariablesCheckBase(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.descr = 'Check for standard Cray variables'
        self.valid_prog_environs = ['builtin']
        self.executable = 'module'
        self.executable_opts = ['show', self.cray_module]
        envvar_prefix = self.cray_module.upper().replace('-', '_')
        self.sanity_patterns = sn.all([
            sn.assert_found(f'{envvar_prefix}_PREFIX', self.stderr),
            sn.assert_found(f'{envvar_prefix}_VERSION', self.stderr)
        ])
        self.tags = {'production', 'craype'}


@rfm.simple_test
class CrayVariablesCheck(CrayVariablesCheckBase):
    cray_module = parameter([
        'cray-fftw', 'cray-hdf5', 'cray-hdf5-parallel', 'cray-libsci',
        'cray-mpich', 'cray-netcdf', 'cray-netcdf-hdf5parallel', 'cray-petsc',
        'cray-petsc-complex-64', 'cray-python', 'cray-R', 'cray-tpsl',
        'cray-tpsl-64', 'cudatoolkit', 'gcc', 'papi', 'pmi'
    ])

    def __init__(self):
        super().__init__()
        self.valid_systems = ['daint:login', 'dom:login']

        # FIXME: These modules should be fixed in later releases,
        # while gcc was fixed in 20.11

        cdt = osext.cray_cdt_version()
        if ((cdt and cdt <= '20.11' and
             self.cray_module in ['cray-petsc-complex',
                                  'cray-petsc-complex-64',
                                  'cudatoolkit']) or
            (cdt and cdt < '20.11' and module_name == 'gcc')):
            self.valid_systems = []

        self.maintainers = ['EK', 'VH']


@rfm.simple_test
class CrayVariablesCheckEiger(CrayVariablesCheckBase):
    cray_module = parameter([
        'cray-fftw', 'cray-hdf5', 'cray-hdf5-parallel', 'cray-libsci',
        'cray-mpich', 'cray-openshmemx', 'cray-parallel-netcdf', 'cray-pmi',
        'cray-python', 'cray-R', 'gcc', 'papi'
    ])

    def __init__(self):
         super().__init__()
         self.valid_systems = ['eiger:login']

         # FIXME: These modules should be fixed in later releases

         if self.cray_module in {'cray-fftw', 'cray-python', 'cray-mpich'}:
             self.valid_systems = []

         self.maintainers = ['TM']
