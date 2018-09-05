import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class EnvironmentVariableTest(rfm.RegressionTest):
    def __init__(self):
        super().__init__()
        self.descr = ('ReFrame tutorial demonstrating the use'
                      'of environment variables provided by loaded modules')
        self.valid_systems = ['daint:gpu']
        self.valid_prog_environs = ['*']
        self.modules = ['cudatoolkit']
        self.variables = {'CUDA_HOME': '$CUDATOOLKIT_HOME'}
        self.executable = './advanced_example4'
        self.build_system = 'Make'
        self.build_system.makefile = 'Makefile_example4'
        self.sanity_patterns = sn.assert_found(r'SUCCESS', self.stdout)
        self.maintainers = ['put-your-name-here']
        self.tags = {'tutorial'}
