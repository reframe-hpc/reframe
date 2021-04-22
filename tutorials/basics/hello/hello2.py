# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
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
    executable_opts = ['> hello.out']
    sanity_patterns = sn.assert_found(r'Hello, World\!', 'hello.out')

    @rfm.run_after('init')
    def set_sourcepath(self):
        self.sourcepath = f'hello.{self.lang}'
