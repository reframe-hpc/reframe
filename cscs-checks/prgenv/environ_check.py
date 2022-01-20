# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.osext as osext
import reframe.utility.sanity as sn


@rfm.simple_test
class DefaultPrgEnvCheck(rfm.RunOnlyRegressionTest):
    descr = 'Ensure PrgEnv-cray is loaded by default'
    valid_prog_environs = ['builtin']
    valid_systems = ['daint:login', 'dom:login',
                     'eiger:login', 'pilatus:login']
    executable = 'module'
    executable_opts = ['--terse', 'list']
    maintainers = ['TM', 'CB']
    tags = {'production', 'craype'}

    @run_before('sanity')
    def set_sanity(self):
        self.sanity_patterns = sn.assert_found(r'^PrgEnv-cray', self.stderr)


@rfm.simple_test
class EnvironmentCheck(rfm.RunOnlyRegressionTest):
    descr = 'Ensure programming environment is loaded correctly'
    valid_systems = ['daint:login', 'dom:login',
                     'eiger:login', 'pilatus:login']
    valid_prog_environs = ['PrgEnv-aocc', 'PrgEnv-cray', 'PrgEnv-gnu',
                           'PrgEnv-intel', 'PrgEnv-pgi', 'PrgEnv-nvidia']
    executable = 'module'
    executable_opts = ['--terse', 'list']
    maintainers = ['TM', 'CB']
    tags = {'production', 'craype'}

    @run_before('sanity')
    def set_sanity(self):
        module_patt = rf'^{self.current_environ.name}'
        self.sanity_patterns = sn.assert_found(module_patt, self.stderr)


class CrayVariablesCheck(rfm.RunOnlyRegressionTest):
    cray_module = parameter()
    descr = 'Check for standard Cray variables'
    valid_prog_environs = ['builtin']
    executable = 'module'
    tags = {'production', 'craype'}
    maintainers = ['EK', 'TM']

    @run_before('run')
    def set_exec_opts(self):
        self.executable_opts = ['show', self.cray_module]

    @run_before('sanity')
    def set_sanity(self):
        envvar_prefix = self.cray_module.upper().replace('-', '_')
        self.sanity_patterns = sn.all([
            sn.assert_found(f'{envvar_prefix}_PREFIX', self.stderr),
            sn.assert_found(f'{envvar_prefix}_VERSION', self.stderr)
        ])


@rfm.simple_test
class CrayVariablesCheckDaint(CrayVariablesCheck):
    cray_module = parameter([
        'cray-fftw', 'cray-hdf5', 'cray-hdf5-parallel', 'cray-libsci',
        'cray-mpich', 'cray-netcdf', 'cray-netcdf-hdf5parallel', 'cray-petsc',
        'cray-petsc-complex-64', 'cray-python', 'cray-R', 'cray-tpsl',
        'cray-tpsl-64', 'cudatoolkit', 'gcc', 'papi', 'pmi'
    ])
    valid_systems = ['daint:login', 'dom:login']

    @run_after('init')
    def skip_modules(self):
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
    valid_systems = ['eiger:login']

    @run_after('init')
    def skip_modules(self):
        # FIXME: These modules should be fixed in later releases
        if self.cray_module in {'cray-fftw', 'cray-python', 'cray-mpich'}:
            self.valid_systems = []
