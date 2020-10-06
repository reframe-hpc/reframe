# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class HelloTest(rfm.RegressionTest):
    def __init__(self):
        self.name = 'hellocheck'
        self.descr = 'C Hello World test'

        # All available systems are supported
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.sourcepath = 'hello.c'
        self.tags = {'foo', 'bar'}
        self.sanity_patterns = sn.assert_found(r'Hello, World\!', self.stdout)
        self.maintainers = ['VK']


@rfm.simple_test
class CompileOnlyHelloTest(rfm.CompileOnlyRegressionTest):
    def __init__(self):
        self.descr = 'Compile-only C Hello World test'

        # All available systems are supported
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.sourcepath = 'hello.c'
        self.sanity_patterns = sn.assert_not_found(r'(?i)error', self.stdout)
