# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
import os
import reframe as rfm
import reframe.utility.sanity as sn


class build_stream(rfm.CompileOnlyRegressionTest):
    build_system = 'SingleSource'
    sourcepath = 'stream.c'
    executable = './stream.x'
    array_size = variable(int, value=0)
    elem_type = parameter(['float', 'double'])

    @run_before('compile')
    def prepare_build(self):
        omp_flag = self.current_environ.extras.get('omp_flag')
        self.build_system.cflags = ['-O3', omp_flag]
        if self.array_size:
            self.build_system.cppflags = [f'-DARRAY_SIZE={self.array_size}',
                                          f'-DELEM_TYPE={self.elem_type}']


@rfm.simple_test
class stream_test(rfm.RunOnlyRegressionTest):
    valid_systems = ['*']
    valid_prog_environs = ['+openmp']
    stream_binary = fixture(build_stream, scope='environment')
    num_threads = parameter([1, 2, 4, 8])
    thread_placement = parameter(['true', 'close', 'spread'])

    @run_after('setup')
    def set_executable(self):
        self.executable = os.path.join(self.stream_binary.stagedir, 'stream.x')

    @run_before('run')
    def setup_threading(self):
        self.env_vars['OMP_NUM_THREADS'] = self.num_threads
        self.env_vars['OMP_PROC_BIND'] = self.thread_placement

    @sanity_function
    def validate(self):
        return sn.assert_found(r'Solution Validates', self.stdout)

    @performance_function('MB/s')
    def copy_bw(self):
        return sn.extractsingle(r'Copy:\s+(\S+)', self.stdout, 1, float)

    @performance_function('MB/s')
    def triad_bw(self):
        return sn.extractsingle(r'Triad:\s+(\S+)', self.stdout, 1, float)
