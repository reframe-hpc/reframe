# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.parameterized_test(['single'], ['funneled'], ['serialized'], ['multiple'])
class MpiInitTest(rfm.RegressionTest):
    '''This test checks the value returned by calling MPI_Init_thread.

    Output should look the same for all prgenv,
    (mpi_thread_multiple seems to be not supported):

    # 'single':
    ['mpi_thread_supported=MPI_THREAD_SINGLE
      mpi_thread_queried=MPI_THREAD_SINGLE 0'],

    # 'funneled':
    ['mpi_thread_supported=MPI_THREAD_FUNNELED
      mpi_thread_queried=MPI_THREAD_FUNNELED 1'],

    # 'serialized':
    ['mpi_thread_supported=MPI_THREAD_SERIALIZED
      mpi_thread_queried=MPI_THREAD_SERIALIZED 2'],

    # 'multiple':
    ['mpi_thread_supported=MPI_THREAD_SERIALIZED
      mpi_thread_queried=MPI_THREAD_SERIALIZED 2']

    '''

    def __init__(self, required_thread):
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
                              'eiger:mc']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu', 'PrgEnv-pgi',
                                    'PrgEnv-intel', 'PrgEnv-aocc']
        self.build_system = 'SingleSource'
        self.sourcesdir = 'src/mpi_thread'
        self.sourcepath = 'mpi_init_thread.cpp'
        self.cppflags = {
            'single':     ['-D_MPI_THREAD_SINGLE'],
            'funneled':   ['-D_MPI_THREAD_FUNNELED'],
            'serialized': ['-D_MPI_THREAD_SERIALIZED'],
            'multiple':   ['-D_MPI_THREAD_MULTIPLE']
        }
        self.build_system.cppflags = self.cppflags[required_thread]
        self.time_limit = '1m'
        self.maintainers = ['JG', 'AJ']
        self.tags = {'production', 'craype'}

    @rfm.run_before('sanity')
    def set_sanity(self):
        # {{{ 0/ MPICH version:
        # MPI VERSION  : CRAY MPICH version 7.7.15 (ANL base 3.2)
        # MPI VERSION  : CRAY MPICH version 8.0.16.17 (ANL base 3.3)
        regex = r'= MPI VERSION\s+: CRAY MPICH version \S+ \(ANL base (\S+)\)'
        stdout = os.path.join(self.stagedir, sn.evaluate(self.stdout))
        mpich_version = sn.extractsingle(regex, stdout, 1)
        self.mpithread_version = {
            '3.2': {
                'MPI_THREAD_SINGLE': 0,
                'MPI_THREAD_FUNNELED': 1,
                'MPI_THREAD_SERIALIZED': 2,
                # required=MPI_THREAD_MULTIPLE/supported=MPI_THREAD_SERIALIZED
                'MPI_THREAD_MULTIPLE': 2
            },
            '3.3': {
                'MPI_THREAD_SINGLE': 0,
                'MPI_THREAD_FUNNELED': 1,
                'MPI_THREAD_SERIALIZED': 2,
                'MPI_THREAD_MULTIPLE': 3
            }
        }
        # }}}
        regex = (r'^mpi_thread_required=(\w+)\s+mpi_thread_supported=\w+'
                 r'\s+mpi_thread_queried=\w+\s+(\d)')
        required_thread = sn.extractsingle(regex, stdout, 1)
        found_mpithread = sn.extractsingle(regex, stdout, 2, int)
        self.sanity_patterns = sn.all([
            sn.assert_found(r'tid=0 out of 1 from rank 0 out of 1',
                            stdout, msg='sanity: not found'),
            sn.assert_eq(found_mpithread,
                         self.mpithread_version[sn.evaluate(mpich_version)]
                                               [sn.evaluate(required_thread)],
                         msg='sanity_eq: {0} != {1}')
        ])


@rfm.simple_test
class MpiHelloTest(rfm.RegressionTest):
    def __init__(self):
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
                              'arolla:cn', 'arolla:pn', 'tsa:cn', 'tsa:pn',
                              'eiger:mc']
        self.valid_prog_environs = ['PrgEnv-cray']
        if self.current_system.name in ['arolla', 'tsa']:
            self.exclusive_access = True
            self.valid_prog_environs = ['PrgEnv-gnu']

        self.descr = 'MPI Hello World'
        self.sourcesdir = 'src/mpi'
        self.sourcepath = 'mpi_helloworld.c'
        self.maintainers = ['RS', 'AJ']
        self.num_tasks_per_node = 1
        self.num_tasks = 0
        num_processes = sn.extractsingle(
            r'Received correct messages from (?P<nprocs>\d+) processes',
            self.stdout, 'nprocs', int)
        self.sanity_patterns = sn.assert_eq(num_processes,
                                            self.num_tasks_assigned-1)
        self.tags = {'diagnostic', 'ops', 'craype'}

    @property
    @sn.sanity_function
    def num_tasks_assigned(self):
        return self.job.num_tasks
