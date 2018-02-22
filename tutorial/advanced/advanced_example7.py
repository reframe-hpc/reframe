import os

import reframe.utility.sanity as sn
from reframe.core.pipeline import RunOnlyRegressionTest


class PrerunDemoTest(RunOnlyRegressionTest):
    def __init__(self, **kwargs):
        super().__init__('prerun_demo_check',
                         os.path.dirname(__file__), **kwargs)
        self.descr = ('ReFrame tutorial demonstrating the use of '
                      'pre- and post-run commands')
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.pre_run  = ['source scripts/limits.sh']
        self.post_run = ['echo FINISHED']
        self.executable = './random_numbers.sh'
        numbers = sn.extractall(
            r'Random: (?P<number>\S+)', self.stdout, 'number', float)
        self.sanity_patterns = sn.all([
            sn.assert_eq(sn.count(numbers), 100),
            sn.all(sn.map(lambda x: sn.assert_bounded(x, 50, 80), numbers)),
            sn.assert_found('FINISHED', self.stdout)
        ])

        self.maintainers = ['put-your-name-here']
        self.tags = {'tutorial'}


def _get_checks(**kwargs):
    return [PrerunDemoTest(**kwargs)]
