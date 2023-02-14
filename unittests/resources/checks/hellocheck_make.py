# Copyright 2016-2023 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class HelloMakeTest(rfm.RegressionTest):
    descr = 'Makefile test'

    # All available systems are supported
    valid_systems = ['*']
    valid_prog_environs = ['*']
    build_system = 'Make'
    executable = './hello_c'
    keep_files = ['hello_c']
    tags = {'foo', 'bar'}
    maintainers = ['VK']

    @run_before('compile')
    def setflags(self):
        self.build_system.cflags = ['-O2']

    @sanity_function
    def validate(self):
        return sn.assert_found(r'Hello, World\!', self.stdout)
