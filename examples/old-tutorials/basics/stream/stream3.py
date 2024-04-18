# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class StreamWithRefTest(rfm.RegressionTest):
    valid_systems = ['*']
    valid_prog_environs = ['gnu']
    prebuild_cmds = [
        'wget https://raw.githubusercontent.com/jeffhammond/STREAM/master/stream.c'  # noqa: E501
    ]
    build_system = 'SingleSource'
    sourcepath = 'stream.c'
    env_vars = {
        'OMP_NUM_THREADS': '4',
        'OMP_PLACES': 'cores'
    }
    reference = {
        'catalina': {
            'Copy':  (25200, -0.05, 0.05, 'MB/s'),
            'Scale': (16800, -0.05, 0.05, 'MB/s'),
            'Add':   (18500, -0.05, 0.05, 'MB/s'),
            'Triad': (18800, -0.05, 0.05, 'MB/s')
        }
    }

    @run_before('compile')
    def set_compiler_flags(self):
        self.build_system.cppflags = ['-DSTREAM_ARRAY_SIZE=$((1 << 25))']
        self.build_system.cflags = ['-fopenmp', '-O3', '-Wall']

    @sanity_function
    def validate_solution(self):
        return sn.assert_found(r'Solution Validates', self.stdout)

    @performance_function('MB/s')
    def extract_bw(self, kind='Copy'):
        '''Generic performance extraction function.'''

        if kind not in ('Copy', 'Scale', 'Add', 'Triad'):
            raise ValueError(f'illegal value in argument kind ({kind!r})')

        return sn.extractsingle(rf'{kind}:\s+(\S+)\s+.*',
                                self.stdout, 1, float)

    @run_before('performance')
    def set_perf_variables(self):
        '''Build the dictionary with all the performance variables.'''

        self.perf_variables = {
            'Copy': self.extract_bw(),
            'Scale': self.extract_bw('Scale'),
            'Add': self.extract_bw('Add'),
            'Triad': self.extract_bw('Triad'),
        }
