import os

import reframe.utility.sanity as sn
from reframe.core.pipeline import RunOnlyRegressionTest


class TimeLimitTest(RunOnlyRegressionTest):
    def __init__(self, **kwargs):
        super().__init__('time_limit_check', os.path.dirname(__file__),
                         **kwargs)

        self.descr = ('ReFrame tutorial demonstrating the use'
                      'of a user-defined time limit')
        self.valid_systems = ['daint:gpu', 'daint:mc']
        self.valid_prog_environs = ['*']
        self.time_limit = (0, 1, 0)
        self.executable = 'sleep'
        self.executable_opts = ['100']
        self.sanity_patterns = sn.assert_found(
            r'CANCELLED.*DUE TO TIME LIMIT', self.stderr)
        self.maintainers = ['put-your-name-here']
        self.tags = {'tutorial'}


def _get_checks(**kwargs):
    return [TimeLimitTest(**kwargs)]
