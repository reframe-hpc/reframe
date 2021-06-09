# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class DGEMMTest(rfm.RegressionTest):
    def __init__(self):
        self.descr = 'DGEMM performance test'
        self.sourcepath = 'dgemm.c'
        self.sanity_patterns = self.eval_sanity()

        # the perf patterns are automaticaly generated inside sanity
        self.perf_patterns = {}
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
                              'arolla:cn', 'arolla:pn', 'tsa:cn', 'tsa:pn',
                              'eiger:mc', 'pilatus:mc']
        if self.current_system.name in ['daint', 'dom']:
            self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-intel']
        elif self.current_system.name in ['arolla', 'tsa']:
            self.valid_prog_environs = ['PrgEnv-gnu-nompi']
        elif self.current_system.name in ['eiger', 'pilatus']:
            self.valid_prog_environs = ['PrgEnv-gnu']
        else:
            self.valid_prog_environs = []

        self.num_tasks = 0
        self.use_multithreading = False
        self.executable_opts = ['6144', '12288', '3072']
        self.build_system = 'SingleSource'
        self.build_system.cflags = ['-O3']
        self.sys_reference = {
            'daint:gpu': (300.0, -0.15, None, 'Gflop/s'),
            'daint:mc': (1040.0, -0.15, None, 'Gflop/s'),
            'dom:gpu': (300.0, -0.15, None, 'Gflop/s'),
            'dom:mc': (1040.0, -0.15, None, 'Gflop/s'),
            'eiger:mc': (3200.0, -0.15, None, 'Gflop/s'),
            'pilatus:mc': (3200.0, -0.15, None, 'Gflop/s'),
        }
        self.maintainers = ['AJ', 'VH']
        self.tags = {'benchmark', 'diagnostic', 'craype'}

    @run_before('compile')
    def setflags(self):
        if self.current_environ.name.startswith('PrgEnv-gnu'):
            self.build_system.cflags += ['-fopenmp']
        elif self.current_environ.name.startswith('PrgEnv-intel'):
            self.build_system.cppflags = [
                '-DMKL_ILP64', '-I${MKLROOT}/include'
            ]
            self.build_system.cflags = ['-qopenmp']
            self.build_system.ldflags = [
                '-mkl', '-static-intel', '-liomp5', '-lpthread', '-lm', '-ldl'
            ]

        if self.current_partition.fullname in ['arolla:cn', 'arolla:pn',
                                               'tsa:cn', 'tsa:pn']:
            self.build_system.cflags += ['-I$EBROOTOPENBLAS/include']
            self.build_system.ldflags = ['-L$EBROOTOPENBLAS/lib', '-lopenblas',
                                         '-lpthread', '-lgfortran']

    @run_before('run')
    def set_tasks(self):
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

        if self.num_cpus_per_task:
            self.variables = {
                'OMP_NUM_THREADS': str(self.num_cpus_per_task),
                'OMP_BIND': 'cores',
                'OMP_PROC_BIND': 'spread',
                'OMP_SCHEDULE': 'static'
            }

    @sn.sanity_function
    def eval_sanity(self):
        all_tested_nodes = sn.evaluate(sn.extractall(
            r'(?P<hostname>\S+):\s+Time for \d+ DGEMM operations',
            self.stdout, 'hostname'))
        num_tested_nodes = len(all_tested_nodes)
        failure_msg = ('Requested %s node(s), but found %s node(s)' %
                       (self.job.num_tasks, num_tested_nodes))
        sn.evaluate(sn.assert_eq(num_tested_nodes, self.job.num_tasks,
                                 msg=failure_msg))

        for hostname in all_tested_nodes:
            partition_name = self.current_partition.fullname
            ref_name = '%s:%s' % (partition_name, hostname)
            self.reference[ref_name] = self.sys_reference.get(
                partition_name, (0.0, None, None, 'Gflop/s')
            )
            self.perf_patterns[hostname] = sn.extractsingle(
                r'%s:\s+Avg\. performance\s+:\s+(?P<gflops>\S+)'
                r'\sGflop/s' % hostname, self.stdout, 'gflops', float)

        return True
