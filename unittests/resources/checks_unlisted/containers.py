import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class ContainerCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.descr = 'Run commands inside a container'
