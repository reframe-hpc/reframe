import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class TestBuggyStdin(rfm.RunOnlyRegressionTest):
    descr = 'Test that stdin is not a tty'
    valid_systems = ['*']
    valid_prog_environs = ['*']
    executable = './test.py'

    @sanity_function
    def assert_stdin_not_a_tty(self):
        return sn.assert_found('stdin is not a tty', self.stdout)
