# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.osext as osext

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
                self.current_partition.fullname not in {'ault:a64fx'}
            )

    @run_after('setup')
    def set_num_cpus_per_task(self):
        if self.current_partition.fullname in ['daint:gpu', 'dom:gpu']:
            self.num_cpus_per_task = 12
        elif self.current_partition.fullname in ['daint:mc', 'dom:mc']:
            self.num_cpus_per_task = 36
        elif self.current_partition.fullname in ['arolla:cn', 'tsa:cn']:
            self.num_cpus_per_task = 16
        elif self.current_partition.fullname in ['arolla:pn', 'tsa:pn']:
            self.num_cpus_per_task = 40
        elif self.current_partition.fullname in ['eiger:mc', 'pilatus:mc']:
            self.num_cpus_per_task = 128
        elif self.current_partition.fullname in ['ault:a64fx']:
            self.num_cpus_per_task = 48

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

    @run_before('performance')
    def set_perf_patterns(self):
        '''Override base performance patterns.

        Set each node as a performance variable reporting the Gflop/s.
        The ``reference`` values for each node are extracted from the
        ``sys_reference`` dict.
        '''

        part_name = self.current_partition.fullname
        with osext.change_dir(self.stagedir):
            node_names = sn.evaluate(self.get_nodenames())

        # If part_name not in sys_reference, default back to '*'
        if part_name not in self.sys_reference:
            part_name = '*'

        # Set references and perf patterns.
        self.reference = {
            part_name: {
                nid: self.sys_reference[part_name] for nid in node_names
            }
        }
        self.perf_patterns = {
            nid: self.get_node_performance(nid) for nid in node_names
        }
