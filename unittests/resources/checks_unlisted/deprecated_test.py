import reframe as rfm
import reframe.utility.sanity as sn

from reframe.core.warnings import user_deprecation_warning


@rfm.simple_test
class DeprecatedTest(rfm.RunOnlyRegressionTest):
    valid_systems = ['*']
    valid_prog_environs = ['*']
    local = True
    executable = 'echo'
    sanity_patterns = sn.assert_true(1)

    @run_before('setup')
    def deprecation_warning(self):
        user_deprecation_warning('feature foo is deprecated')
