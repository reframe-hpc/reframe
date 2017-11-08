import os

import reframe.utility.sanity as sn
from reframe.core.pipeline import RegressionTest


class MakefileTest(RegressionTest):
    def __init__(self, **kwargs):
        super().__init__('preprocessor_check', os.path.dirname(__file__),
                         **kwargs)

        self.descr = ('ReFrame tutorial demonstrating the use of Makefiles '
                      'and compile options')
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.executable = './advanced_example1'
        self.sanity_patterns = sn.assert_found('SUCCESS', self.stdout)
        self.maintainers = ['put-your-name-here']
        self.tags = {'tutorial'}

    def compile(self):
        self.current_environ.cppflags = '-DMESSAGE'
        super().compile()


def _get_checks(**kwargs):
    return [MakefileTest(**kwargs)]
