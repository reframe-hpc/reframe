# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


__all__ = ['Stream']


class Stream(rfm.RegressionTest):
    '''Stream benchmark.

    For info on the executable, see the executable sources.

    Derived tests must set the variables ``num_tasks`` and
    ``num_cpus_per_task``.
    '''

    # Required variables
    num_tasks = required
    num_cpus_per_task = required

    descr = 'STREAM Benchmark'
    exclusive_access = True
    use_multithreading = False
    prebuild_cmds = [
        'wget http://www.cs.virginia.edu/stream/FTP/Code/stream.c',
    ]
    sourcepath = 'stream.c'
    build_system = 'SingleSource'
    num_tasks_per_node = 1
    variables = {
        'OMP_PLACES': 'threads',
        'OMP_PROC_BIND': 'spread'
    }
    reference = {
        '*': {
            'triad': (None, None, None, 'MB/s'),
            'add': (None, None, None, 'MB/s'),
            'copy': (None, None, None, 'MB/s'),
            'scale': (None, None, None, 'MB/s')
        }
    }
    maintainers = ['RS', 'SK']

    @rfm.run_before('run')
    def set_omp_num_threads(self):
        '''Set the number of OMP threads to ``num_cpus_per_task``.'''
        self.variables['OMP_NUM_THREADS'] = f'{self.num_cpus_per_task}'

    @rfm.run_before('sanity')
    def set_sanity_patterns(self):
        '''Set sanity patterns to check the error threshold.'''

        self.sanity_patterns = sn.assert_found(
            r'Solution Validates: avg error less than', self.stdout
        )

    @rfm.run_before('performance')
    def set_performance_patterns(self):
        '''Set performance to track the triad bandwidth.'''

        self.perf_patterns = {
            'triad': sn.min(sn.extractall(
                r'Triad:\s+(?P<triad>\S+)\s+\S+', self.stdout, 'triad', float
            )),
            'add': sn.min(sn.extractall(
                r'Add:\s+(?P<add>\S+)\s+\S+', self.stdout, 'add', float
            )),
            'copy': sn.min(sn.extractall(
                r'Copy:\s+(?P<copy>\S+)\s+\S+', self.stdout, 'copy', float
            )),
            'scale': sn.min(sn.extractall(
                r'Scale:\s+(?P<scale>\S+)\s+\S+', self.stdout, 'scale', float
            )),
        }
