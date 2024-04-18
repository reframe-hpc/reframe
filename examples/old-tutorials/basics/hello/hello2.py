# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class HelloMultiLangTest(rfm.RegressionTest):
    lang = parameter(['c', 'cpp'])

    valid_systems = ['*']
    valid_prog_environs = ['*']

    @run_before('compile')
    def set_sourcepath(self):
        self.sourcepath = f'hello.{self.lang}'

    @sanity_function
    def assert_hello(self):
        return sn.assert_found(r'Hello, World\!', self.stdout)
