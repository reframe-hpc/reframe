import os

import reframe.utility.sanity as sn
from reframe.core.pipeline import RunOnlyRegressionTest


class RunOnlyTest(RunOnlyRegressionTest):
    def __init__(self, **kwargs):
        super().__init__('run_only_check', os.path.dirname(__file__),
                         **kwargs)

        self.descr = ('ReFrame tutorial demonstrating the class'
                      'RunOnlyRegressionTest')
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.sourcesdir = None

        lower = 90
        upper = 100
        self.executable = 'echo "Random: $((RANDOM%({1}+1-{0})+{0}))"'.format(
            lower, upper)
        self.sanity_patterns = sn.assert_bounded(sn.extractsingle(
            r'Random: (?P<number>\S+)', self.stdout, 'number', float),
            lower, upper)
        self.maintainers = ['put-your-name-here']
        self.tags = {'tutorial'}


def _get_checks(**kwargs):
    return [RunOnlyTest(**kwargs)]
