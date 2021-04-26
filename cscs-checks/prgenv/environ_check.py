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
        self.valid_systems = ['daint:login', 'dom:login',
                              'eiger:login', 'pilatus:login']
        self.executable = 'module'
        self.maintainers = ['TM', 'CB']
        self.tags = {'production', 'craype'}
        self.sanity_patterns = sn.assert_found(r'^PrgEnv-cray', self.stderr)

        self.executable_opts = ['--terse', 'list']
        prgenv_patt = r'^PrgEnv-cray'
        self.sanity_patterns = sn.assert_found(prgenv_patt, self.stderr)


@rfm.simple_test
class EnvironmentCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.descr = 'Ensure programming environment is loaded correctly'
        self.valid_systems = ['daint:login', 'dom:login',
                              'eiger:login', 'pilatus:login']
        self.valid_prog_environs = ['PrgEnv-aocc', 'PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi']
        self.executable = 'module'
        self.executable_opts = ['--terse', 'list']

        self.maintainers = ['TM', 'CB']
        self.tags = {'production', 'craype'}

    @rfm.run_before('sanity')
    def set_sanity(self):
        module_patt = rf'^{self.current_environ.name}'

        self.sanity_patterns = sn.assert_found(module_patt, self.stderr)


class CrayVariablesCheck(rfm.RunOnlyRegressionTest):
    cray_module = parameter()

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
        self.maintainers = ['EK', 'TM']


@rfm.simple_test
class CrayVariablesCheckDaint(CrayVariablesCheck):
    cray_module = parameter([
        'cray-fftw', 'cray-hdf5', 'cray-hdf5-parallel', 'cray-libsci',
        'cray-mpich', 'cray-netcdf', 'cray-netcdf-hdf5parallel', 'cray-petsc',
        'cray-petsc-complex-64', 'cray-python', 'cray-R', 'cray-tpsl',
        'cray-tpsl-64', 'cudatoolkit', 'gcc', 'papi', 'pmi'
    ])

    def __init__(self):
        super().__init__()
        self.valid_systems = ['daint:login', 'dom:login']

        # FIXME: These modules should be fixed in later releases
        cdt = osext.cray_cdt_version()
        if ((cdt and cdt <= '20.11' and
             self.cray_module in {'cray-petsc-complex',
                                  'cray-petsc-complex-64',
                                  'cudatoolkit'})):
            self.valid_systems = []


@rfm.simple_test
class CrayVariablesCheckEiger(CrayVariablesCheck):
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
