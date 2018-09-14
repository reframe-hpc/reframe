import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class MakefileTest(rfm.RegressionTest):
    def __init__(self):
        super().__init__()
        self.descr = ('ReFrame tutorial demonstrating the use of Makefiles '
                      'and compile options')
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.executable = './advanced_example1'
        self.build_system = 'Make'
        self.build_system.cppflags = ['-DMESSAGE']
        self.sanity_patterns = sn.assert_found('SUCCESS', self.stdout)
        self.maintainers = ['put-your-name-here']
        self.tags = {'tutorial'}
