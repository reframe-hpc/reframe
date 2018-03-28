import os

from reframe.core.pipeline import RegressionTest


class SomeTest(RegressionTest):
    def __init__(self, **kwargs):
        super().__init__('somecheck', os.path.dirname(__file__), **kwargs)


class InvalidTest:
    def __init__(self, **kwargs):
        pass


def _get_checks(**kwargs):
    return [SomeTest(**kwargs), InvalidTest(**kwargs)]
