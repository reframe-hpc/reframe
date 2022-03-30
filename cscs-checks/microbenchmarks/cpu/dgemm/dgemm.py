# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class DGEMMTest(rfm.RegressionTest):
    descr = 'DGEMM performance test'
    sourcepath = 'dgemm.c'

    # the perf patterns are automaticaly generated inside sanity
    perf_patterns = {}
    valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
                     'arolla:cn', 'arolla:pn', 'tsa:cn', 'tsa:pn',
                     'eiger:mc', 'pilatus:mc']
    num_tasks = 0
    use_multithreading = False
    executable_opts = ['6144', '12288', '3072']
    build_system = 'SingleSource'
    arch_refs = {
        'haswell@12c': (300.0, -0.15, None, 'Gflop/s'),
        'broadwell@36c': (1040.0, -0.15, None, 'Gflop/s'),
        'zen2@128c': (3200.0, -0.15, None, 'Gflop/s'),
        # FIXME: no refs for tsa/arolla
    }
    maintainers = ['AJ', 'VH']
    tags = {'benchmark', 'diagnostic', 'craype'}

    @run_after('init')
    def setup_filtering_criteria(self):
        # FIXME: Revise this as soon as GH #1852 is addressed
        if self.current_system.name in ('daint', 'dom'):
            self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-intel']
        elif self.current_system.name in ('arolla', 'tsa'):
            self.valid_prog_environs = ['PrgEnv-gnu-nompi']
        elif self.current_system.name in ('eiger', 'pilatus'):
            self.valid_prog_environs = ['PrgEnv-gnu']
        else:
            self.valid_prog_environs = []

    @run_before('compile')
    def set_compile_flags(self):
        # FIXME: To avoid using the `startswith()` we can now use the
        # environment `extras` to encode the compiler family, if properly
        # configured in the environment definitions.

        self.build_system.cflags = ['-O3']
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

        if self.current_system.name in ('arolla', 'tsa'):
            self.build_system.cflags += ['-I$EBROOTOPENBLAS/include']
            self.build_system.ldflags = ['-L$EBROOTOPENBLAS/lib', '-lopenblas',
                                         '-lpthread', '-lgfortran']

    @run_before('run')
    def prepare_run(self):
        self.skip_if_no_procinfo()
        self.num_cpus_per_task = self.current_partition.processor.num_cores
        self.variables = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'OMP_BIND': 'cores',
            'OMP_PROC_BIND': 'spread',
            'OMP_SCHEDULE': 'static'
        }

    @sanity_function
    def validate(self):
        # FIXME: This is currently complicated due to GH #2334

        all_tested_nodes = sn.evaluate(sn.extractall(
            r'(?P<hostname>\S+):\s+Time for \d+ DGEMM operations',
            self.stdout, 'hostname'))
        num_tested_nodes = len(all_tested_nodes)
        failure_msg = ('Requested %s node(s), but found %s node(s)' %
                       (self.job.num_tasks, num_tested_nodes))
        sn.evaluate(sn.assert_eq(num_tested_nodes, self.job.num_tasks,
                                 msg=failure_msg))

        pname = self.current_partition.fullname
        arch = self.current_partition.processor.arch
        for hostname in all_tested_nodes:
            key = f'{arch}@{self.num_cpus_per_task}c'
            if key in self.arch_refs:
                self.reference[f'{pname}:{hostname}'] = self.arch_refs[key]

            self.perf_patterns[hostname] = sn.extractsingle(
                fr'{hostname}:\s+Avg\. performance\s+:\s+(?P<gflops>\S+)'
                fr'\sGflop/s', self.stdout, 'gflops', float)

        return True
