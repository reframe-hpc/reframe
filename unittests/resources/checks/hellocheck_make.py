#
# We purposely use the old syntax here
#

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
        self.build_system = 'Make'
        self.build_system.cflags = ['-O3']
        self.build_system.cxxflags = ['-O3']
        self.build_system.makefile = 'Makefile.nofort'
        self.executable = './hello_cpp'
        self.keep_files = ['hello_cpp']
        self.tags = {'foo', 'bar'}
        self.sanity_patterns = sn.assert_found(r'Hello, World\!', self.stdout)
        self.maintainers = ['VK']


def _get_checks(**kwargs):
    return [HelloMakeTest(**kwargs)]
