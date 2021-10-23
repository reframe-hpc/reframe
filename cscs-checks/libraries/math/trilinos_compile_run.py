# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class TrilinosTest(rfm.RegressionTest):
    linkage = parameter(['static', 'dynamic'])
    valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc']
    valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu', 'PrgEnv-intel']
    prgenv_flags = {
        'PrgEnv-cray': ['-fopenmp', '-O2', '-ffast-math', '-std=c++11',
                        '-Wno-everything'],
        'PrgEnv-gnu': ['-fopenmp', '-std=c++11', '-w', '-fpermissive'],
        'PrgEnv-intel': ['-qopenmp', '-w', '-std=c++11'],
    }
    sourcepath = 'example_AmesosFactory_HB.cpp'
    prerun_cmds = ['wget https://math.nist.gov/pub/MatrixMarket2/misc/hamm/'
                   'add20.rua.gz', 'gunzip add20.rua.gz']
    executable_opts = ['add20.rua']
    modules = ['cray-mpich', 'cray-hdf5-parallel', 'cray-tpsl',
               'cray-trilinos']
    num_tasks = 2
    num_tasks_per_node = 2
    maintainers = ['AJ', 'CB']
    tags = {'production', 'craype'}

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

    @run_before('run')
    def prepare_run(self):
        self.variables = {'OMP_NUM_THREADS': '1'}
