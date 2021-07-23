# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.osext as osext
import reframe.utility.sanity as sn


@rfm.simple_test
class TrilinosTest(rfm.RegressionTest):
    linkage = parameter(['static', 'dynamic'])
    valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc']
    valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-intel']
    prgenv_flags = {
        'PrgEnv-cray': ['-fopenmp', '-O2', '-ffast-math', '-std=c++11',
                        '-Wno-everything'],
        'PrgEnv-cray_classic': ['-homp', '-hstd=c++11', '-hmsglevel_4'],
        'PrgEnv-gnu': ['-fopenmp', '-std=c++11', '-w', '-fpermissive'],
        'PrgEnv-intel': ['-qopenmp', '-w', '-std=c++11'],
        'PrgEnv-pgi': ['-mp', '-w']
    }
    sourcepath = 'example_AmesosFactory_HB.cpp'
    prerun_cmds = ['wget ftp://math.nist.gov/pub/MatrixMarket2/'
                   'misc/hamm/add20.rua.gz', 'gunzip add20.rua.gz']
    executable_opts = ['add20.rua']
    modules = ['cray-mpich', 'cray-hdf5-parallel', 'cray-tpsl',
               'cray-trilinos']
    num_tasks = 2
    num_tasks_per_node = 2
    maintainers = ['AJ', 'CB']
    tags = {'production', 'craype'}

    @run_after('init')
    def extend_valid_prog_environments(self):
        # NOTE: PrgEnv-cray in dynamic does not work because of CrayBug/809265
        # NOTE: PrgEnv-cray_classic does not support trilinos
        if self.linkage == 'static':
            self.valid_prog_environs += ['PrgEnv-cray']

    @sanity_function
    def assert_solution(self):
        return sn.assert_found(r'After Amesos solution', self.stdout)

    @run_before('compile')
    def set_build_system_opts(self):
        self.build_system = 'SingleSource'
        self.build_system.ldflags = [f'-{self.linkage}', f'-lparmetis']
        self.build_system.cppflags = ['-DHAVE_MPI', '-DEPETRA_MPI']
        flags = self.prgenv_flags[self.current_environ.name]
        self.build_system.cxxflags = flags

    @run_before('compile')
    def cdt2006_workaround_intel(self):
        if (self.current_environ.name == 'PrgEnv-intel' and
            osext.cray_cdt_version() == '20.06'):
            self.modules += ['cray-netcdf-hdf5parallel']
            self.prebuild_cmds = [
                'ln -s $CRAY_NETCDF_HDF5PARALLEL_PREFIX/lib/pkgconfig/'
                'netcdf-cxx4_parallel.pc netcdf_c++4_parallel.pc'
            ]
            self.variables['PKG_CONFIG_PATH'] = '.:$PKG_CONFIG_PATH'

    @run_before('compile')
    def cdt2006_workaround_dynamic(self):
        if (osext.cray_cdt_version() == '20.06' and
            self.linkage == 'dynamic' and
            self.current_environ.name == 'PrgEnv-gnu'):
            self.variables['PATH'] = (
                '/opt/cray/pe/cce/10.0.1/cce-clang/x86_64/bin:$PATH'
            )
            self.prgenv_flags[self.current_environ.name] += ['-fuse-ld=lld']

            # GCC >= 9 is required for the above option; our CUDA-friendly CDT
            # uses GCC 8 as default.
            self.modules += ['gcc/9.3.0']

    @run_before('run')
    def prepare_run(self):
        self.variables = {'OMP_NUM_THREADS': '1'}
