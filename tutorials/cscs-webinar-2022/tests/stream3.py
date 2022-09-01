import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class stream_test(rfm.RegressionTest):
    valid_systems = ['*']
    valid_prog_environs = ['+openmp']
    build_system = 'SingleSource'
    sourcepath = 'stream.c'

    @run_before('compile')
    def setup_build(self):
        try:
            omp_flag = self.current_environ.extras['ompflag']
        except KeyError:
            envname = self.current_environ.name
            self.skip(f'"ompflag" not defined for enviornment {envname!r}')

        self.build_system.cflags = [omp_flag, '-O3']
        self.build_system.cppflags = [f'-DSTREAM_ARRAY_SIZE={1 << 25}']

    @sanity_function
    def validate_solution(self):
        return sn.assert_found(r'Solution Validates', self.stdout)

    @performance_function('MB/s')
    def copy_bandwidth(self):
        return sn.extractsingle(r'Copy:\s+(\S+)\s+.*', self.stdout, 1, float)
