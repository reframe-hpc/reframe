import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class PrerunDemoTest(rfm.RunOnlyRegressionTest):
    def __init__(self):
        super().__init__()
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
