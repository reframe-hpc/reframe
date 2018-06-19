#
# New-style checks for testing the registration decorators
#

import reframe as rfm

# We just import this individually for testing purposes
from reframe.core.pipeline import RegressionTest


@rfm.parameterized_test(*((x, y) for x in range(3) for y in range(2)))
class MyBaseTest(RegressionTest):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    def __eq__(self, other):
        """This is just for unit tests for convenience."""
        if not isinstance(other, MyBaseTest):
            return NotImplemented

        return self.a == other.a and self.b == other.b

    def __repr__(self):
        return 'MyBaseTest(%s, %s)' % (self.a, self.b)


@rfm.parameterized_test(*({'a': x, 'b': y} for x in range(3) for y in range(2)))
class AnotherBaseTest(RegressionTest):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    def __eq__(self, other):
        """This is just for unit tests for convenience."""
        if not isinstance(other, AnotherBaseTest):
            return NotImplemented

        return self.a == other.a and self.b == other.b

    def __repr__(self):
        return 'AnotherBaseTest(%s, %s)' % (self.a, self.b)


@rfm.required_version('>=2.13-dev1')
@rfm.simple_test
class MyTest(MyBaseTest):
    def __init__(self):
        super().__init__(10, 20)


# We intentionally have swapped the order of the two decorators here.
# The order should not play any role.
@rfm.simple_test
@rfm.required_version('<=2.12')
class InvalidTest(MyBaseTest):
    pass
