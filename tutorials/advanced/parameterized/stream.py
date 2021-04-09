# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class StreamMultiSysTest(rfm.RegressionTest):
    num_bytes = parameter(1 << pow for pow in range(19, 30))

    def __init__(self):
        array_size = (self.num_bytes >> 3) // 3
        ntimes = 100*1024*1024 // array_size
        self.descr = f'STREAM test (array size: {array_size}, ntimes: {ntimes})'    # noqa: E501
        self.valid_systems = ['*']
        self.valid_prog_environs = ['cray', 'gnu', 'intel', 'pgi']
        self.prebuild_cmds = [
            'wget http://www.cs.virginia.edu/stream/FTP/Code/stream.c',
        ]
        self.sourcepath = 'stream.c'
        self.build_system = 'SingleSource'
        self.build_system.cppflags = [f'-DSTREAM_ARRAY_SIZE={array_size}',
                                      f'-DNTIMES={ntimes}']
        self.sanity_patterns = sn.assert_found(r'Solution Validates',
                                               self.stdout)
        self.perf_patterns = {
            'Triad': sn.extractsingle(r'Triad:\s+(\S+)\s+.*',
                                      self.stdout, 1, float),
        }

        # Flags per programming environment
        self.flags = {
            'cray':  ['-fopenmp', '-O3', '-Wall'],
            'gnu':   ['-fopenmp', '-O3', '-Wall'],
            'intel': ['-qopenmp', '-O3', '-Wall'],
            'pgi':   ['-mp', '-O3']
        }

        # Number of cores for each system
        self.cores = {
            'catalina:default': 4,
            'daint:gpu': 12,
            'daint:mc': 36,
            'daint:login': 10
        }
        self.reference = {
            '*': {
                'Triad': (0, None, None, 'MB/s'),
            }
        }

    @rfm.run_before('compile')
    def setflags(self):
        environ = self.current_environ.name
        self.build_system.cflags = self.flags.get(environ, [])

    @rfm.run_before('run')
    def set_num_threads(self):
        num_threads = self.cores.get(self.current_partition.fullname, 1)
        self.num_cpus_per_task = num_threads
        self.variables = {
            'OMP_NUM_THREADS': str(num_threads),
            'OMP_PLACES': 'cores'
        }
