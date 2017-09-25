import re

from reframe.core.pipeline import RegressionTest
from reframe.core.environments import *


class HelloTest(RegressionTest):
    def __init__(self, **kwargs):
        super().__init__('hellocheck', os.path.dirname(__file__), **kwargs)
        self.descr = 'C Hello World test'

        # All available systems are supported
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.sourcepath = 'hello.c'
        self.tags = {'foo', 'bar'}
        self.sanity_patterns = {
            '-' : {'Hello, World\!' : []}
        }
        self.maintainers = ['VK']


def _get_checks(**kwargs):
    return [HelloTest(**kwargs)]
