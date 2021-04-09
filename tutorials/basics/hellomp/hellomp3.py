# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class HelloThreadedExtended2Test(rfm.RegressionTest):
    def __init__(self):
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.sourcepath = 'hello_threads.cpp'
        self.executable_opts = ['16']
        self.build_system = 'SingleSource'
        self.build_system.cppflags = ['-DSYNC_MESSAGES']
        self.build_system.cxxflags = ['-std=c++11', '-Wall']
        num_messages = sn.len(sn.findall(r'\[\s?\d+\] Hello, World\!',
                                         self.stdout))
        self.sanity_patterns = sn.assert_eq(num_messages, 16)

    @rfm.run_before('compile')
    def set_threading_flags(self):
        environ = self.current_environ.name
        if environ in {'clang', 'gnu'}:
            self.build_system.cxxflags += ['-pthread']
