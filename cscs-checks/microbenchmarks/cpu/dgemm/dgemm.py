# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm

from hpctestlib.microbenchmarks.cpu.dgemm import Dgemm


@rfm.simple_test
class dgemm_check(Dgemm):
    '''CSCS DGEMM check.

    The matrix dimensions are set in the base class.
    Every node reports its performance in Gflops/s. To do so, this class
    overrides the performance patterns and references from the base test.
    This is done in the ``set_perf_patterns`` pre-performance hook.
    '''

    valid_systems = [
        'daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
        'arolla:cn', 'arolla:pn', 'tsa:cn', 'tsa:pn',
        'eiger:mc', 'pilatus:mc', 'ault:a64fx'
    ]
    num_tasks = 0
    sys_reference = variable(
        dict, value={
            'daint:gpu':  (300.0, -0.15, None, 'Gflop/s'),
            'daint:mc':   (1040.0, -0.15, None, 'Gflop/s'),
            'dom:gpu':    (300.0, -0.15, None, 'Gflop/s'),
            'dom:mc':     (1040.0, -0.15, None, 'Gflop/s'),
            'eiger:mc':   (3200.0, -0.15, None, 'Gflop/s'),
            'pilatus:mc': (3200.0, -0.15, None, 'Gflop/s'),
            'ault:a64fx': (1930.0, -0.15, None, 'Gflop/s'),
            '*':          (None, None, None, 'Gflop/s'),
        },
    )
    tags = {'benchmark', 'diagnostic', 'craype'}

    @run_after('init')
    def set_valid_prog_environs(self):
        if self.current_system.name in ['daint', 'dom']:
            self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-intel']
        elif self.current_system.name in ['arolla', 'tsa']:
            self.valid_prog_environs = ['PrgEnv-gnu-nompi']
        elif self.current_system.name in ['eiger', 'pilatus']:
            self.valid_prog_environs = ['PrgEnv-gnu']
        elif self.current_system.name in ['ault']:
            self.valid_prog_environs = ['PrgEnv-fujitsu']

    @run_after('setup')
    def skip_incompatible_combinations(self):
        '''Fujitsu env only available in ault's a64fx partition.'''
        if self.current_environ.name.startswith('PrgEnv-fujitsu'):
            self.skip_if(
                self.current_partition.fullname not in ('ault:a64fx')
            )

    @run_after('setup')
    def set_num_cpus_per_task(self):
        proc = self.current_partition.processor
        pname = self.current_partition.fullname
        if not proc.info:
            self.skip(f'no topology information found for partition {pname!r}')

        self.num_cpus_per_task = proc.num_cpus // proc.num_cpus_per_core

    @run_before('compile')
    def set_flags(self):
        if self.current_environ.name.startswith('PrgEnv-gnu'):
            self.build_system.cflags += ['-fopenmp']
        elif self.current_environ.name.startswith('PrgEnv-intel'):
            self.build_system.cppflags = [
                '-DMKL_ILP64', '-I${MKLROOT}/include'
            ]
            self.build_system.cflags += ['-qopenmp']
            self.build_system.ldflags = [
                '-mkl', '-static-intel', '-liomp5', '-lpthread', '-lm', '-ldl'
            ]
        elif self.current_environ.name.startswith('PrgEnv-fujitsu'):
            self.build_system.cflags += ['-fopenmp', '-Nlibomp', '-mt']
            self.build_system.ldflags += ['-SSL2BLAMP', '-mt']

        if self.current_partition.fullname in ['arolla:cn', 'arolla:pn',
                                               'tsa:cn', 'tsa:pn']:
            self.build_system.cflags += ['-I$EBROOTOPENBLAS/include']
            self.build_system.ldflags = ['-L$EBROOTOPENBLAS/lib', '-lopenblas',
                                         '-lpthread', '-lgfortran']
