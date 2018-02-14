import os

import reframe.utility.sanity as sn
from reframe.core.pipeline import RunOnlyRegressionTest


class DeferredIterationTest(RunOnlyRegressionTest):
    def __init__(self, **kwargs):
        super().__init__('deferred_iteration_check',
                         os.path.dirname(__file__), **kwargs)

        self.descr = ('ReFrame tutorial demonstrating the use of deferred '
                      'iteration via the `map` sanity function.')

        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']

        self.pre_run = ['source limits.sh']
        self.executable = './advanced_example6.sh'
        numbers = sn.extractall(r'Random: (?P<number>\S+)', self.stdout,
                                'number', float)

        self.sanity_patterns = sn.and_(
            sn.assert_eq(sn.count(numbers), 100),
            sn.all(sn.map(lambda x: sn.assert_bounded(x, 90, 100), numbers)))

        self.maintainers = ['put-your-name-here']
        self.tags = {'tutorial'}


def _get_checks(**kwargs):
    return [DeferredIterationTest(**kwargs)]
