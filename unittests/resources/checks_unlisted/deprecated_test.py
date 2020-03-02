import reframe as rfm


@rfm.simple_test
class DeprecatedTest(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']

        # Use deprecated syntax to raise a ReframeDeprecationWarning
        self.time_limit = (0, 1, 0)
