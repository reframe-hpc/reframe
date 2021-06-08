# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


__all__ = ['Dgemm']


class Dgemm(rfm.RegressionTest, pin_prefix=True):
    '''Dgemm benchmark.

    Derived test must specify the variables ``num_tasks`` and
    ``num_cpus_per_task``.

    The matrix sizes can be controlled through executable options. By default,
    this test sets these as ``m=6144``, ``n=12288`` and ``k=3072``. Derived
    tests are free to change these parameters at their convenience. The
    performance of this tests is measured by the lowest performing node in
    ``Gflops/s``.
    '''

    num_tasks = required
    num_cpus_per_task = required

    descr = 'DGEMM performance test'
    sourcepath = 'dgemm.c'
    use_multithreading = False
    executable_opts = ['6144', '12288', '3072']
    build_system = 'SingleSource'
    reference = {
        '*': {
            'min_perf': (None, None, None, 'Gflops/s')
        }
    }
    maintainers = ['AJ', 'VH']

    @run_before('compile')
    def set_c_flags(self):
        self.build_system.cflags += ['-O3']

    @run_before('run')
    def set_env_vars(self):
        '''Set the environment variables.'''

        self.variables = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'OMP_BIND': 'cores',
            'OMP_PROC_BIND': 'spread',
            'OMP_SCHEDULE': 'static'
        }

    @sn.sanity_function
    def get_nodenames(self):
        '''Return a set with the participating node IDs.'''

        return set(sn.extractall(
            r'(?P<hostname>\S+):\s+Time for \d+ DGEMM operations',
            self.stdout, 'hostname'
        ))

    @run_before('sanity')
    def set_sanity_patterns(self):
        '''Assert that all requested nodes have completed.'''

        self.sanity_patterns = sn.assert_eq(
            self.job.num_tasks, sn.count(self.get_nodenames()),
            msg='some nodes did not complete'
        )

    @sn.sanity_function
    def get_node_performance(self, nodeid):
        '''Get the performance data from a specific ``nodeid``.'''

        return sn.extractsingle(
            r'%s:\s+Avg\. performance\s+:\s+(?P<gflops>\S+)\sGflop/s' % nodeid,
            self.stdout, 'gflops', float
        )

    @sn.sanity_function
    def get_min_performance(self):
        '''Get the lowest performance from all nodes.'''

        return sn.min([
            self.get_node_performance(nid) for nid in self.get_nodenames()
        ])

    @run_before('performance')
    def set_perf_patterns(self):
        '''Set the perf patterns to check the min performance reported.'''

        self.perf_patterns = {
            'min_perf': self.get_min_performance(),
        }
