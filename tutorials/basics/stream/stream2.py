# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class StreamWithRefTest(rfm.RegressionTest):
    def __init__(self):
        self.valid_systems = ['*']
        self.valid_prog_environs = ['gnu']
        self.prebuild_cmds = [
            'wget http://www.cs.virginia.edu/stream/FTP/Code/stream.c',
        ]
        self.build_system = 'SingleSource'
        self.sourcepath = 'stream.c'
        self.build_system.cppflags = ['-DSTREAM_ARRAY_SIZE=$((1 << 25))']
        self.build_system.cflags = ['-fopenmp', '-O3', '-Wall']
        self.variables = {
            'OMP_NUM_THREADS': '4',
            'OMP_PLACES': 'cores'
        }
        self.sanity_patterns = sn.assert_found(r'Solution Validates',
                                               self.stdout)
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
        self.reference = {
            'catalina': {
                'Copy':  (25200, -0.05, 0.05, 'MB/s'),
                'Scale': (16800, -0.05, 0.05, 'MB/s'),
                'Add':   (18500, -0.05, 0.05, 'MB/s'),
                'Triad': (18800, -0.05, 0.05, 'MB/s')
            }
        }
