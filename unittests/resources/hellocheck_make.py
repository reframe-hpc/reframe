import os

import reframe.utility.sanity as sn
from reframe.core.pipeline import RegressionTest


class HelloMakeTest(RegressionTest):
    def __init__(self, **kwargs):
        super().__init__('hellocheck_make',
                         os.path.dirname(__file__), **kwargs)
        self.descr = 'C++ Hello World test'

        # All available systems are supported
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.sourcepath = ''
        self.executable = './hello_cpp'
        self.keep_files = ['hello_cpp']
        self.tags = {'foo', 'bar'}
        self.sanity_patterns = sn.assert_found(r'Hello, World\!', self.stdout)
        self.maintainers = ['VK']

    def compile(self):
        self.current_environ.cflags = '-O3'
        self.current_environ.cxxflags = '-O3'
        super().compile(makefile='Makefile.nofort')


def _get_checks(**kwargs):
    return [HelloMakeTest(**kwargs)]
