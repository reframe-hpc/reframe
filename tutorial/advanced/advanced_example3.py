import os

import reframe.utility.sanity as sn
from reframe.core.pipeline import CompileOnlyRegressionTest


class CompileOnlyTest(CompileOnlyRegressionTest):
    def __init__(self, **kwargs):
        super().__init__('compile_only_check', os.path.dirname(__file__),
                         **kwargs)

        self.descr = ('ReFrame tutorial demonstrating the class'
                      'CompileOnlyRegressionTest')
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.sanity_patterns = sn.assert_not_found('warning', self.stderr)
        self.maintainers = ['put-your-name-here']
        self.tags = {'tutorial'}


def _get_checks(**kwargs):
    return [CompileOnlyTest(**kwargs)]
