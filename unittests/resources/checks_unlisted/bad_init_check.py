import reframe as rfm


@rfm.simple_test
class BadInitTest(rfm.RegressionTest):
    def __init__(self):
        foo
