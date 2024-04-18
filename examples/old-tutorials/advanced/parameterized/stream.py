# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class StreamMultiSysTest(rfm.RegressionTest):
    num_bytes = parameter(1 << pow for pow in range(19, 30))
    array_size = variable(int)
    ntimes = variable(int)

    valid_systems = ['*']
    valid_prog_environs = ['cray', 'gnu', 'intel', 'nvidia']
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
        '*': {
            'Triad': (0, None, None, 'MB/s'),
        }
    }

    # Flags per programming environment
    flags = variable(dict, value={
        'cray':  ['-fopenmp', '-O3', '-Wall'],
        'gnu':   ['-fopenmp', '-O3', '-Wall'],
        'intel': ['-qopenmp', '-O3', '-Wall'],
        'nvidia':   ['-mp', '-O3']
    })

    # Number of cores for each system
    cores = variable(dict, value={
        'catalina:default': 4,
        'daint:gpu': 12,
        'daint:mc': 36,
        'daint:login': 10
    })

    @run_after('init')
    def setup_build(self):
        self.array_size = (self.num_bytes >> 3) // 3
        self.ntimes = 100*1024*1024 // self.array_size
        self.descr = (
            f'STREAM test (array size: {self.array_size}, '
            f'ntimes: {self.ntimes})'
        )

    @run_before('compile')
    def set_compiler_flags(self):
        self.build_system.cppflags = [f'-DSTREAM_ARRAY_SIZE={self.array_size}',
                                      f'-DNTIMES={self.ntimes}']
        environ = self.current_environ.name
        self.build_system.cflags = self.flags.get(environ, [])

    @run_before('run')
    def set_num_threads(self):
        num_threads = self.cores.get(self.current_partition.fullname, 1)
        self.num_cpus_per_task = num_threads
        self.env_vars = {
            'OMP_NUM_THREADS': num_threads,
            'OMP_PLACES': 'cores'
        }

    @sanity_function
    def validate_solution(self):
        return sn.assert_found(r'Solution Validates', self.stdout)

    @performance_function('MB/s', perf_key='Triad')
    def extract_triad_bw(self):
        return sn.extractsingle(r'Triad:\s+(\S+)\s+.*', self.stdout, 1, float)
