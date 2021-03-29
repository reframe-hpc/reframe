# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# New-style checks for testing the registration decorators
#

import reframe as rfm
import reframe.utility.sanity as sn

# We just import this individually for testing purposes
from reframe.core.pipeline import RegressionTest


class _Base(RegressionTest):
    def __init__(self):
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.sanity_patterns = sn.assert_true(1)


@rfm.parameterized_test(*((x, y) for x in range(3) for y in range(2)))
class MyBaseTest(_Base):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    def __eq__(self, other):
        '''This is just for unit tests for convenience.'''
        if not isinstance(other, MyBaseTest):
            return NotImplemented

        return self.a == other.a and self.b == other.b

    def __repr__(self):
        return 'MyBaseTest(%s, %s)' % (self.a, self.b)


@rfm.parameterized_test(
    *({'a': x, 'b': y} for x in range(3) for y in range(2))
)
class AnotherBaseTest(_Base):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    def __eq__(self, other):
        '''This is just for unit tests for convenience.'''
        if not isinstance(other, AnotherBaseTest):
            return NotImplemented

        return self.a == other.a and self.b == other.b

    def __repr__(self):
        return 'AnotherBaseTest(%s, %s)' % (self.a, self.b)


@rfm.required_version('>=2.13.0-dev.1')
@rfm.simple_test
class MyTest(MyBaseTest):
    def __init__(self):
        super().__init__(10, 20)


# We intentionally have swapped the order of the two decorators here.
# The order should not play any role.
@rfm.simple_test
@rfm.required_version('<=2.12.0')
class InvalidTest(MyBaseTest):
    pass
