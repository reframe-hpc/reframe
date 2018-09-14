import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class ExampleRunOnlyTest(rfm.RunOnlyRegressionTest):
    def __init__(self):
        super().__init__()
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
