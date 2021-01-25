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
        self.valid_systems = ['daint:login', 'dom:login']
        self.executable = 'module'
        self.executable_opts = ['list', '-t']
        self.maintainers = ['TM', 'CB']
        self.tags = {'production', 'craype'}
        self.sanity_patterns = sn.assert_found(r'^PrgEnv-cray', self.stderr)


@rfm.simple_test
class EnvironmentCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.descr = 'Ensure programming environment is loaded correctly'
        self.valid_systems = ['daint:login', 'dom:login']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi']

        self.executable = 'module'
        self.executable_opts = ['list', '-t']
        self.sanity_patterns = sn.assert_found(self.env_module_patt,
                                               self.stderr)
        self.maintainers = ['TM', 'CB']
        self.tags = {'production', 'craype'}

    @property
    @sn.sanity_function
    def env_module_patt(self):
        return r'^%s' % self.current_environ.name


@rfm.parameterized_test(['cray-fftw'], ['cray-hdf5'], ['cray-hdf5-parallel'],
                        ['cray-libsci'], ['cray-netcdf'],
                        ['cray-netcdf-hdf5parallel'], ['cray-petsc'],
                        ['cray-petsc-64'], ['cray-petsc-complex'],
                        ['cray-petsc-complex-64'], ['cray-python'],
                        ['cray-R'], ['cray-tpsl'], ['cray-tpsl-64'],
                        ['cudatoolkit'], ['gcc'], ['papi'], ['pmi'])
class CrayVariablesCheck(rfm.RunOnlyRegressionTest):
    def __init__(self, module_name):
        self.descr = 'Check for standard Cray variables'
        self.valid_systems = ['daint:login', 'dom:login']
        self.valid_prog_environs = ['builtin']
        self.executable = 'module'
        self.executable_opts = ['show', module_name]
        envvar_prefix = module_name.upper().replace('-', '_')
        self.sanity_patterns = sn.all([
            sn.assert_found(f'{envvar_prefix}_PREFIX', self.stderr),
            sn.assert_found(f'{envvar_prefix}_VERSION', self.stderr)
        ])

        # FIXME: These modules should be fixed in later releases
        cdt = osext.cray_cdt_version()
        if (cdt and cdt <= '20.11' and
            module_name in ['cray-petsc-complex',
                            'cray-petsc-complex-64',
                            'cudatoolkit']):
            self.valid_systems = []

        self.maintainers = ['EK', 'VH']
        self.tags = {'production', 'craype'}
