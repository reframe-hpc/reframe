# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

from datetime import datetime

import re
import reframe as rfm
import reframe.utility.sanity as sn


class HelloWorldBaseTest(rfm.RegressionTest):
    def __init__(self, variant, lang, linkage):
        self.linkage = linkage
        self.variables = {'CRAYPE_LINK_TYPE': linkage}
        self.prgenv_flags = {}
        self.lang_names = {
            'c': 'C',
            'cpp': 'C++',
            'f90': 'Fortran 90'
        }
        self.descr = f'{self.lang_names[lang]} Hello World'
        self.sourcepath = 'hello_world'
        self.build_system = 'SingleSource'
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
                              'arolla:cn', 'arolla:pn', 'tsa:cn', 'tsa:pn']
        if linkage == 'dynamic':
            self.valid_systems += ['eiger:mc', 'pilatus:mc']

        self.valid_prog_environs = ['PrgEnv-aocc', 'PrgEnv-cray',
                                    'PrgEnv-cray_classic', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi',
                                    'PrgEnv-gnu-nocuda', 'PrgEnv-pgi-nocuda']

        if self.current_system.name in ['arolla', 'tsa']:
            self.exclusive_access = True

        self.compilation_time_seconds = None

        result = sn.findall(r'Hello World from thread \s*(\d+) out '
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

        self.sanity_patterns = sn.all(
            sn.chain(
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
            )
        )
        self.perf_patterns = {
            'compilation_time': sn.getattr(self, 'compilation_time_seconds')
        }
        self.reference = {
            '*': {
                'compilation_time': (60, None, 0.1, 's')
            }
        }

        self.maintainers = ['VH', 'EK']
        self.tags = {'production', 'craype'}

    @run_before('compile')
    def setflags(self):
        envname = re.sub(r'(PrgEnv-\w+).*', lambda m: m.group(1),
                         self.current_environ.name)
        prgenv_flags = self.prgenv_flags[envname]
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


@rfm.parameterized_test(*([lang, linkage]
                          for lang in ['cpp', 'c', 'f90']
                          for linkage in ['dynamic', 'static']))
class HelloWorldTestSerial(HelloWorldBaseTest):
    def __init__(self, lang, linkage):
        super().__init__('serial', lang, linkage)
        self.sourcesdir = 'src/serial'
        self.valid_prog_environs += ['PrgEnv-gnu-nompi', 'PrgEnv-pgi-nompi',
                                     'PrgEnv-gnu-nompi-nocuda',
                                     'PrgEnf-pgi-nompi-nocuda']
        self.sourcepath += '_serial.' + lang
        self.descr += ' Serial ' + linkage.capitalize()
        self.prgenv_flags = {
            'PrgEnv-aocc': [],
            'PrgEnv-cray': [],
            'PrgEnv-cray_classic': [],
            'PrgEnv-gnu': [],
            'PrgEnv-intel': [],
            'PrgEnv-pgi': []
        }

        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.num_cpus_per_task = 1
        if (self.current_system.name in ['arolla', 'tsa'] and
            linkage == 'dynamic'):
            self.valid_prog_environs += ['PrgEnv-pgi-nompi',
                                         'PrgEnv-pgi-nompi-nocuda',
                                         'PrgEnv-gnu-nompi',
                                         'PrgEnv-gnu-nompi-nocuda']


@rfm.parameterized_test(*([lang, linkage]
                          for lang in ['cpp', 'c', 'f90']
                          for linkage in ['dynamic', 'static']))
class HelloWorldTestOpenMP(HelloWorldBaseTest):
    def __init__(self, lang, linkage):
        super().__init__('openmp', lang, linkage)
        self.sourcesdir = 'src/openmp'
        self.sourcepath += '_openmp.' + lang
        self.descr += ' OpenMP ' + str.capitalize(linkage)
        self.prgenv_flags = {
            'PrgEnv-aocc': ['-fopenmp'],
            'PrgEnv-cray': ['-homp' if lang == 'F90' else '-fopenmp'],
            'PrgEnv-cray_classic': ['-homp'],
            'PrgEnv-gnu': ['-fopenmp'],
            'PrgEnv-intel': ['-qopenmp'],
            'PrgEnv-pgi': ['-mp']
        }

        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.num_cpus_per_task = 4
        if (self.current_system.name in ['arolla', 'tsa'] and
            linkage == 'dynamic'):
            self.valid_prog_environs += ['PrgEnv-pgi-nompi',
                                         'PrgEnv-pgi-nompi-nocuda',
                                         'PrgEnv-gnu-nompi',
                                         'PrgEnv-gnu-nompi-nocuda']

        # On SLURM there is no need to set OMP_NUM_THREADS if one defines
        # num_cpus_per_task, but adding for completeness and portability
        self.variables = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task)
        }


@rfm.parameterized_test(*([lang, linkage]
                          for lang in ['cpp', 'c', 'f90']
                          for linkage in ['dynamic', 'static']))
class HelloWorldTestMPI(HelloWorldBaseTest):
    def __init__(self, lang, linkage):
        super().__init__('mpi', lang, linkage)
        self.sourcesdir = 'src/mpi'
        self.sourcepath += '_mpi.' + lang
        self.descr += ' MPI ' + linkage.capitalize()
        self.prgenv_flags = {
            'PrgEnv-aocc': [],
            'PrgEnv-cray': [],
            'PrgEnv-cray_classic': [],
            'PrgEnv-gnu': [],
            'PrgEnv-intel': [],
            'PrgEnv-pgi': []
        }

        # for the MPI test the self.num_tasks_per_node should always be one. If
        # not, the test will fail for the total number of lines in the output
        # file is different then self.num_tasks * self.num_tasks_per_node
        self.num_tasks = 2
        self.num_tasks_per_node = 1
        self.num_cpus_per_task = 1


@rfm.parameterized_test(*([lang, linkage]
                          for lang in ['cpp', 'c', 'f90']
                          for linkage in ['dynamic', 'static']))
class HelloWorldTestMPIOpenMP(HelloWorldBaseTest):
    def __init__(self, lang, linkage):
        super().__init__('mpi_openmp', lang, linkage)
        self.sourcesdir = 'src/mpi_openmp'
        self.sourcepath += '_mpi_openmp.' + lang
        self.descr += ' MPI + OpenMP ' + linkage.capitalize()
        self.prgenv_flags = {
            'PrgEnv-aocc': ['-fopenmp'],
            'PrgEnv-cray': ['-homp' if lang == 'F90' else '-fopenmp'],
            'PrgEnv-cray_classic': ['-homp'],
            'PrgEnv-gnu': ['-fopenmp'],
            'PrgEnv-intel': ['-qopenmp'],
            'PrgEnv-pgi': ['-mp']
        }

        self.num_tasks = 6
        self.num_tasks_per_node = 3
        self.num_cpus_per_task = 4

        # On SLURM there is no need to set OMP_NUM_THREADS if one defines
        # num_cpus_per_task, but adding for completeness and portability
        self.variables = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task)
        }
