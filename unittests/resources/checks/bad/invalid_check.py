import reframe as rfm


@rfm.simple_test
class SomeTest(rfm.RegressionTest):
    def __init__(self):
        super().__init__()


class NotATest:
    def __init__(self):
        pass
