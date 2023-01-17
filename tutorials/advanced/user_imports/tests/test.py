import reframe as rfm
import reframe.utility as util
import reframe.utility.sanity as sn
from testutil import greetings_from_test

commonutil = util.import_module('..commonutil')


@rfm.simple_test
class MyTest(rfm.RunOnlyRegressionTest):
    valid_systems = ['*']
    valid_prog_environs = ['*']
    executable = f'echo {commonutil.greetings("friends")}'

    @sanity_function
    def validate(self):
        return sn.assert_found('Hello, friends', self.stdout)


@rfm.simple_test
class MyTest2(MyTest):
    @run_before('run')
    def set_exec(self):
        self.executable = f'echo {greetings_from_test(self)}'

    @sanity_function
    def validate(self):
        return sn.assert_found(f'Hello from {self.name}', self.stdout)
