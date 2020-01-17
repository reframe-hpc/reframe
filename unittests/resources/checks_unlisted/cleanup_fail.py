import os
import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class CleanupFailTest(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.sourcesdir = None
        self.executable = 'echo foo'
        self.sanity_patterns = sn.assert_found(r'foo', self.stdout)

    @rfm.run_before('cleanup')
    def fail(self):
        # Make this test fail on purpose
        raise Exception


@rfm.simple_test
class SleepTest(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.sourcesdir = None
        self.executable = 'echo foo && sleep 1'
        self.sanity_patterns = sn.assert_found(r'foo', self.stdout)
