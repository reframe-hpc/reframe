# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.os_ext as os_ext
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
        self.linkage = linkage

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
        if (self.current_environ.name == 'PrgEnv-intel' and
            os_ext.cray_cdt_version() == '20.06'):
            self.modules += ['cray-netcdf-hdf5parallel']
            self.prebuild_cmds = [
                'ln -s $CRAY_NETCDF_HDF5PARALLEL_PREFIX/lib/pkgconfig/'
                'netcdf-cxx4_parallel.pc netcdf_c++4_parallel.pc'
            ]
            self.variables['PKG_CONFIG_PATH'] = '.:$PKG_CONFIG_PATH'

    @rfm.run_before('compile')
    def cdt2006_workaround_dynamic(self):
        if (os_ext.cray_cdt_version() == '20.06' and
            self.linkage == 'dynamic' and
            self.current_environ.name == 'PrgEnv-gnu'):
            self.variables['PATH'] = (
                '/opt/cray/pe/cce/10.0.1/cce-clang/x86_64/bin:$PATH'
            )
            self.prgenv_flags[self.current_environ.name] += ['-fuse-ld=lld']

            # GCC >= 9 is required for the above option; our CUDA-friendly CDT
            # uses GCC 8 as default.
            self.modules += ['gcc/9.3.0']
