import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class ExampleCompileOnlyTest(rfm.CompileOnlyRegressionTest):
    def __init__(self):
        super().__init__()
        self.descr = ('ReFrame tutorial demonstrating the class'
                      'CompileOnlyRegressionTest')
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.sanity_patterns = sn.assert_not_found('warning', self.stderr)
        self.maintainers = ['put-your-name-here']
        self.tags = {'tutorial'}
