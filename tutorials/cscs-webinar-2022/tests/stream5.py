import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class stream_test(rfm.RegressionTest):
    valid_systems = ['*']
    valid_prog_environs = ['+openmp']
    build_system = 'SingleSource'
    sourcepath = 'stream.c'
    reference = {
        'tresa': {
            'copy_bandwidth': (23000, -0.05, None, 'MB/s')
        }
    }
    array_size = variable(int, value=(1 << 25))
    num_iters = variable(int, value=10)

    @run_before('compile')
    def setup_build(self):
        try:
            omp_flag = self.current_environ.extras['ompflag']
        except KeyError:
            envname = self.current_environ.name
            self.skip(f'"ompflag" not defined for enviornment {envname!r}')

        self.build_system.cflags = [omp_flag, '-O3']
        self.build_system.cppflags = [f'-DSTREAM_ARRAY_SIZE={self.array_size}',
                                      f'-DNTIMES={self.num_iters}']

    @run_before('run')
    def setup_omp_env(self):
        procinfo = self.current_partition.processor
        self.num_cpus_per_task = procinfo.num_cores
        self.env_vars = {
            'OMP_NUM_THREADS': self.num_cpus_per_task,
            'OMP_PLACES': 'cores'
        }

    @sanity_function
    def validate_solution(self):
        return sn.assert_found(r'Solution Validates', self.stdout)

    @performance_function('MB/s')
    def copy_bandwidth(self):
        return sn.extractsingle(r'Copy:\s+(\S+)\s+.*', self.stdout, 1, float)
