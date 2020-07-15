# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.14')
@rfm.parameterized_test(['static'], ['dynamic'])
class TrilinosTest(rfm.RegressionTest):
    def __init__(self, linkage):
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
                              'tiger:gpu']
        # NOTE: PrgEnv-cray in dynamic does not work because of CrayBug/809265
        self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-intel']
        # NOTE: PrgEnv-cray_classic does not support trilinos
        if linkage == 'static':
            self.valid_prog_environs += ['PrgEnv-cray']

        self.build_system = 'SingleSource'
        self.build_system.ldflags = ['-%s' % linkage, '-lparmetis']
        self.build_system.cppflags = ['-DHAVE_MPI', '-DEPETRA_MPI']
        self.prgenv_flags = {
            'PrgEnv-cray': ['-fopenmp', '-O2', '-ffast-math', '-std=c++11',
                            '-Wno-everything'],
            'PrgEnv-cray_classic': ['-homp', '-hstd=c++11', '-hmsglevel_4'],
            'PrgEnv-gnu': ['-fopenmp', '-std=c++11', '-w', '-fpermissive'],
            'PrgEnv-intel': ['-qopenmp', '-w', '-std=c++11'],
            'PrgEnv-pgi': ['-mp', '-w']
        }
        self.sourcepath = 'example_AmesosFactory_HB.cpp'
        self.prerun_cmds = ['wget ftp://math.nist.gov/pub/MatrixMarket2/'
                            'misc/hamm/add20.rua.gz', 'gunzip add20.rua.gz']
        self.executable_opts = ['add20.rua']
        self.modules = ['cray-mpich', 'cray-hdf5-parallel',
                        'cray-tpsl', 'cray-trilinos']
        self.num_tasks = 2
        self.num_tasks_per_node = 2
        self.variables = {'OMP_NUM_THREADS': '1'}
        self.sanity_patterns = sn.assert_found(r'After Amesos solution',
                                               self.stdout)

        self.maintainers = ['AJ', 'CB']
        self.tags = {'production', 'craype'}

    @rfm.run_before('compile')
    def set_cxxflags(self):
        flags = self.prgenv_flags[self.current_environ.name]
        self.build_system.cxxflags = flags

    @rfm.run_before('compile')
    def cdt2006_workaround_intel(self):
        if self.current_environ.name.startswith('PrgEnv-intel'):
            if '20.06' in os.getenv('MODULERCFILE', ''):
                self.modules += ['cray-netcdf-hdf5parallel']
                self.prebuild_cmds = [
                    'ln -s $CRAY_NETCDF_HDF5PARALLEL_PREFIX/lib/pkgconfig/'\
                        'netcdf-cxx4_parallel.pc netcdf_c++4_parallel.pc',
                    'export PKG_CONFIG_PATH=`pwd`:$PKG_CONFIG_PATH']
