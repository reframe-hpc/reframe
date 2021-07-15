# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

from datetime import datetime

import re
import reframe as rfm
import reframe.utility.sanity as sn


class HelloWorldBaseTest(rfm.RegressionTest):
    linking = parameter(['dynamic', 'static'])
    lang = parameter(['c', 'cpp', 'f90'])
    prgenv_flags = {}
    lang_names = {
        'c': 'C',
        'cpp': 'C++',
        'f90': 'Fortran 90'
    }
    sourcepath = 'hello_world'
    build_system = 'SingleSource'
    valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
                     'arolla:cn', 'arolla:pn', 'tsa:cn', 'tsa:pn']
    valid_prog_environs = ['PrgEnv-aocc', 'PrgEnv-cray',
                           'PrgEnv-cray_classic', 'PrgEnv-gnu',
                           'PrgEnv-intel', 'PrgEnv-pgi',
                           'PrgEnv-gnu-nocuda', 'PrgEnv-pgi-nocuda']
    exclusive_access = True
    compilation_time_seconds = None
    maintainers = ['VH', 'EK']
    tags = {'production', 'craype'}

    @run_after('init')
    def set_description(self):
        self.descr = f'{self.lang_names[self.lang]} Hello, World'

    @run_after('init')
    def add_dynamic_only_systems(self):
        if self.linking == 'dynamic':
            self.valid_systems += ['eiger:mc', 'pilatus:mc']

    @run_after('init')
    def set_craylinktype_env_variable(self):
        self.variables['CRAYPE_LINK_TYPE'] = self.linking

    @sanity_function
    def set_sanity_patterns(self):
        result = sn.findall(r'Hello, World from thread \s*(\d+) out '
                            r'of \s*(\d+) from process \s*(\d+) out of '
                            r'\s*(\d+)', self.stdout)

        num_tasks = sn.getattr(self, 'num_tasks')
        num_cpus_per_task = sn.getattr(self, 'num_cpus_per_task')

        def tid(match):
            return int(match.group(1))

        def num_threads(match):
            return int(match.group(2))

        def rank(match):
            return int(match.group(3))

        def num_ranks(match):
            return int(match.group(4))

        return sn.all(sn.chain(
                [sn.assert_eq(sn.count(result), num_tasks*num_cpus_per_task)],
                sn.map(lambda x: sn.assert_lt(tid(x), num_threads(x)), result),
                sn.map(lambda x: sn.assert_lt(rank(x), num_ranks(x)), result),
                sn.map(
                    lambda x: sn.assert_lt(tid(x), num_cpus_per_task), result
                ),
                sn.map(
                    lambda x: sn.assert_eq(num_threads(x), num_cpus_per_task),
                    result
                ),
                sn.map(lambda x: sn.assert_lt(rank(x), num_tasks), result),
                sn.map(
                    lambda x: sn.assert_eq(num_ranks(x), num_tasks), result
                ),
            ))

    @run_after('init')
    def set_performance_patterns(self):
        self.perf_patterns = {
            'compilation_time': sn.getattr(self, 'compilation_time_seconds')
        }
        self.reference = {
            '*': {
                'compilation_time': (60, None, 0.1, 's')
            }
        }

    @run_before('compile')
    def setflags(self):
        envname = re.sub(r'(PrgEnv-\w+).*', lambda m: m.group(1),
                         self.current_environ.name)
        try:
            prgenv_flags = self.prgenv_flags[envname]
        except KeyError:
            prgenv_flags = []

        self.build_system.cflags = prgenv_flags
        self.build_system.cxxflags = prgenv_flags
        self.build_system.fflags = prgenv_flags

    @run_before('compile')
    def compile_timer_start(self):
        self.compilation_time_seconds = datetime.now()

    @run_after('compile')
    def compile_timer_end(self):
        elapsed = datetime.now() - self.compilation_time_seconds
        self.compilation_time_seconds = elapsed.total_seconds()


@rfm.simple_test
class HelloWorldTestSerial(HelloWorldBaseTest):
    sourcesdir = 'src/serial'
    num_tasks = 1
    num_tasks_per_node = 1
    num_cpus_per_task = 1

    @run_after('init')
    def extend_valid_prog_environs(self):
        self.valid_prog_environs += ['PrgEnv-gnu-nompi', 'PrgEnv-pgi-nompi',
                                     'PrgEnv-gnu-nompi-nocuda',
                                     'PrgEnf-pgi-nompi-nocuda']
        if (self.current_system.name in ['arolla', 'tsa'] and
            linking == 'dynamic'):
            self.valid_prog_environs += ['PrgEnv-pgi-nompi',
                                         'PrgEnv-pgi-nompi-nocuda',
                                         'PrgEnv-gnu-nompi',
                                         'PrgEnv-gnu-nompi-nocuda']

    @run_after('init')
    def update_description(self):
        self.descr += ' Serial ' + self.linking.capitalize()

    @run_after('init')
    def update_sourcepath(self):
        self.sourcepath += '_serial.' + self.lang


@rfm.simple_test
class HelloWorldTestOpenMP(HelloWorldBaseTest):
    sourcesdir = 'src/openmp'
    num_tasks = 1
    num_tasks_per_node = 1
    num_cpus_per_task = 4

    @run_after('init')
    def set_prgenv_compilation_flags_map(self):
        self.prgenv_flags = {
            'PrgEnv-aocc': ['-fopenmp'],
            'PrgEnv-cray': ['-homp' if self.lang == 'f90' else '-fopenmp'],
            'PrgEnv-cray_classic': ['-homp'],
            'PrgEnv-gnu': ['-fopenmp'],
            'PrgEnv-intel': ['-qopenmp'],
            'PrgEnv-pgi': ['-mp']
        }

    @run_after('init')
    def extend_valid_prog_environs(self):
        if (self.current_system.name in ['arolla', 'tsa'] and
            linking == 'dynamic'):
            self.valid_prog_environs += ['PrgEnv-pgi-nompi',
                                         'PrgEnv-pgi-nompi-nocuda',
                                         'PrgEnv-gnu-nompi',
                                         'PrgEnv-gnu-nompi-nocuda']

    @run_after('init')
    def update_description(self):
        self.descr += ' OpenMP ' + self.linking.capitalize()

    @run_after('init')
    def update_sourcepath(self):
        self.sourcepath += '_openmp.' + self.lang

    @run_after('init')
    def set_omp_env_variable(self):
        # On SLURM there is no need to set OMP_NUM_THREADS if one defines
        # num_cpus_per_task, but adding for completeness and portability
        self.variables['OMP_NUM_THREADS'] = str(self.num_cpus_per_task)


@rfm.simple_test
class HelloWorldTestMPI(HelloWorldBaseTest):
    sourcesdir = 'src/mpi'
    # for the MPI test the self.num_tasks_per_node should always be one. If
    # not, the test will fail for the total number of lines in the output
    # file is different then self.num_tasks * self.num_tasks_per_node
    num_tasks = 2
    num_tasks_per_node = 1
    num_cpus_per_task = 1

    @run_after('init')
    def update_description(self):
        self.descr += ' MPI ' + self.linking.capitalize()

    @run_after('init')
    def update_sourcepath(self):
        self.sourcepath += '_mpi.' + self.lang


@rfm.simple_test
class HelloWorldTestMPIOpenMP(HelloWorldBaseTest):
    sourcesdir = 'src/mpi_openmp'
    num_tasks = 6
    num_tasks_per_node = 3
    num_cpus_per_task = 4

    @run_after('init')
    def set_prgenv_compilation_flags_map(self):
        self.prgenv_flags = {
            'PrgEnv-aocc': ['-fopenmp'],
            'PrgEnv-cray': ['-homp' if self.lang == 'f90' else '-fopenmp'],
            'PrgEnv-cray_classic': ['-homp'],
            'PrgEnv-gnu': ['-fopenmp'],
            'PrgEnv-intel': ['-qopenmp'],
            'PrgEnv-pgi': ['-mp']
        }

    @run_after('init')
    def update_description(self):
        self.descr += ' MPI + OpenMP ' + self.linking.capitalize()

    @run_after('init')
    def update_sourcepath(self):
        self.sourcepath += '_mpi_openmp.' + self.lang

    @run_after('init')
    def set_omp_env_variable(self):
        # On SLURM there is no need to set OMP_NUM_THREADS if one defines
        # num_cpus_per_task, but adding for completeness and portability
        self.variables['OMP_NUM_THREADS'] = str(self.num_cpus_per_task)
