import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.builtins import sanity_function, fixture
from reframe.core.runtime import runtime


_FIXT_PASS_THRES = 1
_TEST_PASS_THRES = 2


class FixtA(rfm.RunOnlyRegressionTest):
    executable = 'true'

    @sanity_function
    def validate(self):
        return runtime().current_run >= _FIXT_PASS_THRES


@rfm.simple_test
class TestA(rfm.RunOnlyRegressionTest):
    fixtA = fixture(FixtA)
    executable = 'true'
    valid_systems = ['*']
    valid_prog_environs = ['*']

    @sanity_function
    def validate(self):
        if runtime().current_run >= _TEST_PASS_THRES:
            return sn.assert_true(self.fixtA.stagedir.endswith('_retry1'),
                                  msg=f'wrong stagedir: {self.fixtA.stagedir}')

        return False
