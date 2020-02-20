# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class HelloMakeTest(rfm.RegressionTest):
    def __init__(self):
        self.descr = 'C++ Hello World test'

        # All available systems are supported
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.build_system = 'Make'
        self.build_system.cflags = ['-O3']
        self.build_system.cxxflags = ['-O3']
        self.build_system.makefile = 'Makefile.nofort'
        self.executable = './hello_cpp'
        self.keep_files = ['hello_cpp']
        self.tags = {'foo', 'bar'}
        self.sanity_patterns = sn.assert_found(r'Hello, World\!', self.stdout)
        self.maintainers = ['VK']
