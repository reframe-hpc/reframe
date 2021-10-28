# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm

from hpctestlib.microbenchmarks.cpu.stream import Stream


@rfm.simple_test
class stream_check(Stream):
    '''Stream benchmark test.'''

    valid_systems = [
        'daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
        'arolla:cn', 'arolla:pn', 'tsa:cn', 'tsa:pn', 'ault:a64fx'
    ]
    valid_prog_environs = [
        'PrgEnv-cray', 'PrgEnv-gnu', 'PrgEnv-intel', 'PrgEnv-pgi',
        'PrgEnv-nvidia'
    ]
    triad_reference = variable(
        dict, value={
            'PrgEnv-cray': {
                'daint:gpu': {'triad': (44000, -0.05, None, 'MB/s')},
                'daint:mc': {'triad': (89000, -0.05, None, 'MB/s')},
                'dom:gpu': {'triad': (44000, -0.05, None, 'MB/s')},
                'dom:mc': {'triad': (89000, -0.05, None, 'MB/s')},
            },
            'PrgEnv-gnu': {
                'daint:gpu': {'triad': (43800, -0.05, None, 'MB/s')},
                'daint:mc': {'triad': (88500, -0.05, None, 'MB/s')},
                'dom:gpu': {'triad': (43800, -0.05, None, 'MB/s')},
                'dom:mc': {'triad': (87500, -0.05, None, 'MB/s')},
            },
            'PrgEnv-intel': {
                'daint:gpu': {'triad': (59500, -0.05, None, 'MB/s')},
                'daint:mc': {'triad': (119000, -0.05, None, 'MB/s')},
                'dom:gpu': {'triad': (59500, -0.05, None, 'MB/s')},
                'dom:mc': {'triad': (119000, -0.05, None, 'MB/s')},
            },
            'PrgEnv-pgi': {
                'daint:gpu': {'triad': (44500, -0.05, None, 'MB/s')},
                'daint:mc': {'triad': (88500, -0.05, None, 'MB/s')},
                'dom:gpu': {'triad': (44500, -0.05, None, 'MB/s')},
                'dom:mc': {'triad': (88500, -0.05, None, 'MB/s')},
            },
            'PrgEnv-fujitsu': {
                'ault:a64fx': {'triad': (85500, -0.05, None, 'MB/s')},
            },
        }
    )
    num_tasks = 1
    tags = {'production', 'craype'}

    @run_after('init')
    def filter_valid_prog_environs(self):
        '''Special conditions for arolla and tsa.'''
        if self.current_system.name in ['arolla', 'tsa']:
            self.valid_prog_environs = ['PrgEnv-gnu']
        elif self.current_system.name in ['ault']:
            self.valid_prog_environs = ['PrgEnv-fujitsu']

    @run_after('setup')
    def set_num_cpus_per_task(self):
        '''Set the num cpus based on the autodetected topology.'''
        proc = self.current_partition.processor
        pname = self.current_partition.fullname
        if not proc.info:
            self.skip(f'no topology information found for partition {pname!r}')

        self.num_cpus_per_task = proc.num_cpus // proc.num_cpus_per_core

    @run_before('compile')
    def set_compiler_flags(self):
        '''Set build flags for the different environments.'''

        envname = self.current_environ.name
        if envname in ('PrgEnv-cray', 'PrgEnv-gnu'):
            self.build_system.cflags += ['-fopenmp', '-O3']
        elif envname in ('PrgEnv-intel'):
            self.build_system.cflags += ['-qopenmp', '-O3']
        elif envname in ('PrgEnv-pgi', 'PrgEnv-nvidia'):
            self.build_system.cflags += ['-mp', '-O3']
        elif envname in ('PrgEnv-fujitsu'):
            self.build_system.cflags += ['-fopenmp', '-mt', '-O3']
            self.build_system.ldflags += ['-mt']

    @run_before('run')
    def set_env_vars(self):
        '''Special environment treatment for the PrgEnv-pgi.'''
        if self.current_environ.name == 'PrgEnv-pgi':
            self.variables['OMP_PROC_BIND'] = 'true'

    @run_before('performance')
    def set_triad_references(self):
        '''Set performance refs as defined in ``triad_reference``.'''
        envname = self.current_environ.name
        if envname in self.triad_reference:
            self.reference = self.triad_reference[envname]
