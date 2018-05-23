import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class TimeLimitTest(rfm.RunOnlyRegressionTest):
    def __init__(self):
        super().__init__()
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
