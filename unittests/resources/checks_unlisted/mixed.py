import os

import reframe.core.decorators as deco
from reframe.core.pipeline import RegressionTest


@deco.parameterized_test(*((x, y) for x in range(3) for y in range(2)))
class MyBaseTest(RegressionTest):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b


@deco.simple_test
class MyTest(MyBaseTest):
    def __init__(self):
        super().__init__(10, 20)


class OldStyleTest(MyBaseTest):
    def __init__(self, **kwargs):
        super().__init__('old_style_test',
                         os.path.dirname(__file__), **kwargs)


def _get_checks(**kwargs):
    return [OldStyleTest(**kwargs)]
