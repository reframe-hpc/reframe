# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
from testlib.simple import simple_echo_check


@rfm.simple_test
class HelloFoo(simple_echo_check):
    message = 'Foo'
