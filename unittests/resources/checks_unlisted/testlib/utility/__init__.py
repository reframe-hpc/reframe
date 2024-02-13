import reframe as rfm
import reframe.utility.sanity as sn


class dummy_fixture(rfm.RunOnlyRegressionTest):
    executable = 'echo'
    sanity_patterns = sn.assert_true(1)
