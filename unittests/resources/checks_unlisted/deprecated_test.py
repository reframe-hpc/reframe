import reframe as rfm
import reframe.utility.sanity as sn

from reframe.core.warnings import user_deprecation_warning


@rfm.simple_test
class DeprecatedTest(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.local = True
        self.executable = 'echo hello'
        self.sanity_patterns = sn.assert_found('hello', self.stdout)

    @run_before('setup')
    def deprecation_warning(self):
        user_deprecation_warning('feature foo is deprecated')
