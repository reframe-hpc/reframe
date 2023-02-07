# Copyright 2016-2023 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn

from testlib.simple import simple_check


@rfm.simple_test
class HelloFoo(simple_check):
    executable_opts = ['Foo']

    @sanity_function
    def assert_output(self):
        return sn.assert_found(r'Hello Foo', self.stdout)
