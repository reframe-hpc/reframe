# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
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
        'wget http://www.cs.virginia.edu/stream/FTP/Code/stream.c',
    ]
    build_system = 'SingleSource'
    sourcepath = 'stream.c'
    variables = {
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

    @run_before('sanity')
    def set_sanity_patterns(self):
        self.sanity_patterns = sn.assert_found(r'Solution Validates',
                                               self.stdout)

    @run_before('performance')
    def set_perf_patterns(self):
        self.perf_patterns = {
            'Copy': sn.extractsingle(r'Copy:\s+(\S+)\s+.*',
                                     self.stdout, 1, float),
            'Scale': sn.extractsingle(r'Scale:\s+(\S+)\s+.*',
                                      self.stdout, 1, float),
            'Add': sn.extractsingle(r'Add:\s+(\S+)\s+.*',
                                    self.stdout, 1, float),
            'Triad': sn.extractsingle(r'Triad:\s+(\S+)\s+.*',
                                      self.stdout, 1, float)
        }
