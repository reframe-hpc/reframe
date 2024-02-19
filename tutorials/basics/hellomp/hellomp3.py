# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class HelloThreadedExtended2Test(rfm.RegressionTest):
    valid_systems = ['*']
    valid_prog_environs = ['*']
    sourcepath = 'hello_threads.cpp'
    build_system = 'SingleSource'
    executable_opts = ['16']

    @run_before('compile')
    def set_compilation_flags(self):
        self.build_system.cppflags = ['-DSYNC_MESSAGES']
        self.build_system.cxxflags = ['-std=c++11', '-Wall']
        environ = self.current_environ.name
        if environ in {'clang', 'gnu'}:
            self.build_system.cxxflags += ['-pthread']

    @sanity_function
    def assert_num_messages(self):
        num_messages = sn.len(sn.findall(r'\[\s?\d+\] Hello, World\!',
                                         self.stdout))
        return sn.assert_eq(num_messages, 16)
