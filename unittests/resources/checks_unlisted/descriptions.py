import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class BaseTest(rfm.RunOnlyRegressionTest):
    _descr = parameter(['foo', '', '__required__'])
    descr = required
    executable = 'true'
    sanity_patterns = sn.assert_true(1)
    valid_systems = ['*']
    valid_prog_environs = ['*']

    @run_after('init')
    def set_descr(self):
        if self._descr != '__required__':
            self.descr = self._descr

@rfm.simple_test
class DerivedTest(BaseTest):
    test_with_descr = fixture(BaseTest)
