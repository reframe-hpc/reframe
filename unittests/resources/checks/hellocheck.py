import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class HelloTest(rfm.RegressionTest):
    def __init__(self):
        super().__init__()
        self.name = 'hellocheck'
        self.descr = 'C Hello World test'

        # All available systems are supported
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.sourcepath = 'hello.c'
        self.tags = {'foo', 'bar'}
        self.sanity_patterns = sn.assert_found(r'Hello, World\!', self.stdout)
        self.maintainers = ['VK']
