# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.parameterized_test(*([lang, linkage] for lang in ['cpp', 'c', 'f90']
                          for linkage in ['dynamic', 'static']))
class NetCDFTest(rfm.RegressionTest):
    def __init__(self, lang, linkage):
        lang_names = {
            'c': 'C',
            'cpp': 'C++',
            'f90': 'Fortran 90'
        }
        self.lang = lang
        self.linkage = linkage
        self.descr = f'{lang_names[lang]} NetCDF {linkage.capitalize()}'
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
                              'arolla:cn', 'tsa:cn']
        if linkage == 'dynamic':
            self.valid_systems += ['eiger:mc', 'pilatus:mc']

        if self.current_system.name in ['daint', 'dom']:
            self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                        'PrgEnv-intel', 'PrgEnv-pgi']
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

        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'netcdf')
        self.build_system = 'SingleSource'
        self.sourcepath = 'netcdf_read_write.' + lang
        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.sanity_patterns = sn.assert_found(r'SUCCESS', self.stdout)
        self.maintainers = ['AJ', 'SO']
        self.tags = {'production', 'craype', 'external-resources', 'health'}

    @rfm.run_before('compile')
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
