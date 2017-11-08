import os

import reframe.utility.sanity as sn
from reframe.core.pipeline import RegressionTest


class EnvironmentVariableTest(RegressionTest):
    def __init__(self, **kwargs):
        super().__init__('env_variable_check', os.path.dirname(__file__),
                         **kwargs)

        self.descr = ('ReFrame tutorial demonstrating the use'
                      'of environment variables provided by loaded modules')
        self.valid_systems = ['daint:gpu']
        self.valid_prog_environs = ['*']
        self.modules = ['cudatoolkit']
        self.variables = {'CUDA_HOME': '$CUDATOOLKIT_HOME'}
        self.executable = './advanced_example4'
        self.sanity_patterns = sn.assert_found(r'SUCCESS', self.stdout)
        self.maintainers = ['put-your-name-here']
        self.tags = {'tutorial'}

    def compile(self):
        super().compile(makefile='Makefile_example4')


def _get_checks(**kwargs):
    return [EnvironmentVariableTest(**kwargs)]
