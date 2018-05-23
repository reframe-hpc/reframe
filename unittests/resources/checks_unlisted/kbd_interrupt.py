#
# A special check to simulate a keyboard interrupt
#
# The reason this test is in a different file is just for being loaded by the
# CLI unit tests exclusively.
#

import reframe as rfm


@rfm.simple_test
class KeyboardInterruptCheck(rfm.RunOnlyRegressionTest):
    def __init__(self, **kwargs):
        super().__init__()
        self.local = True
        self.executable = 'sleep 1'
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.tags = {self.name}

    def setup(self, system, environ, **job_opts):
        raise KeyboardInterrupt
