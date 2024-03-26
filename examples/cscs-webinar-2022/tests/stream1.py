import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class stream_test(rfm.RegressionTest):
    valid_systems = ['*']
    valid_prog_environs = ['*']
    build_system = 'SingleSource'
    sourcepath = 'stream.c'

    @sanity_function
    def validate_solution(self):
        return sn.assert_found(r'Solution Validates', self.stdout)
