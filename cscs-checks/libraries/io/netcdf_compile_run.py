# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class NetCDFTest(rfm.RegressionTest):
    lang = parameter(['cpp', 'c', 'f90'])
    linkage = parameter(['dynamic', 'static'])
    valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
                     'arolla:cn', 'tsa:cn']
    build_system = 'SingleSource'
    num_tasks = 1
    num_tasks_per_node = 1
    maintainers = ['AJ', 'SO']
    tags = {'production', 'craype', 'external-resources', 'health'}
    lang_names = {
            'c': 'C',
            'cpp': 'C++',
            'f90': 'Fortran 90'
    }

    @run_after('init')
    def set_description(self):
        self.descr = (f'{self.lang_names[self.lang]} NetCDF '
                      f'{self.linkage.capitalize()}')

    @run_after('init')
    def setup_prgenvs(self):
        if self.linkage == 'dynamic':
            self.valid_systems += ['eiger:mc', 'pilatus:mc']

        if self.current_system.name in ['daint', 'dom']:
            self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                        'PrgEnv-intel', 'PrgEnv-pgi',
                                        'PrgEnv-nvidia']
            self.modules = ['cray-netcdf']
        elif self.current_system.name in ['arolla', 'tsa']:
            self.exclusive_access = True
            self.valid_prog_environs = ['PrgEnv-gnu-nompi', 'PrgEnv-pgi-nompi']
        elif self.current_system.name in ['eiger', 'pilatus']:
            # no cray-netcdf as of PE 21.02 with PrgEnv-intel
            self.valid_prog_environs = ['PrgEnv-aocc', 'PrgEnv-cray',
                                        'PrgEnv-gnu']
            self.modules = ['cray-hdf5', 'cray-netcdf']
        else:
            self.valid_prog_environs = []

    @run_before('compile')
    def set_sources(self):
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'netcdf')
        self.sourcepath = f'netcdf_read_write.{self.lang}'

    @run_before('compile')
    def setflags(self):
        if self.current_system.name in ['arolla', 'tsa']:
            self.modules = ['netcdf', 'netcdf-c++', 'netcdf-fortran']
            self.build_system.cppflags = [
                '-I$EBROOTNETCDF/include',
                '-I$EBROOTNETCDFMINCPLUSPLUS/include',
                '-I$EBROOTNETCDFMINFORTRAN/include'
            ]
            self.build_system.ldflags = [
                '-L$EBROOTNETCDF/lib',
                '-L$EBROOTNETCDFMINCPLUSPLUS/lib',
                '-L$EBROOTNETCDFMINFORTRAN/lib',
                '-L$EBROOTNETCDF/lib64',
                '-L$EBROOTNETCDFMINCPLUSPLUS/lib64',
                '-L$EBROOTNETCDFMINFORTRAN/lib64',
                '-lnetcdf', '-lnetcdf_c++4', '-lnetcdff'
            ]
        else:
            self.build_system.ldflags = [f'-{self.linkage}']

    @run_before('sanity')
    def set_sanity(self):
        self.sanity_patterns = sn.assert_found(r'SUCCESS', self.stdout)
