import os
import reframe as rfm
import reframe.utility.sanity as sn


class stream_build(rfm.CompileOnlyRegressionTest):
    build_system = 'SingleSource'
    sourcepath = 'stream.c'
    array_size = variable(int, value=(1 << 25))
    num_iters = variable(int, value=10)
    elem_type = parameter(['double', 'float'])
    executable = 'stream'

    @run_before('compile')
    def setup_build(self):
        try:
            omp_flag = self.current_environ.extras['ompflag']
        except KeyError:
            envname = self.current_environ.name
            self.skip(f'"ompflag" not defined for enviornment {envname!r}')

        self.build_system.cflags = [omp_flag, '-O3']
        self.build_system.cppflags = [f'-DSTREAM_ARRAY_SIZE={self.array_size}',
                                      f'-DNTIMES={self.num_iters}',
                                      f'-DSTREAM_TYPE={self.elem_type}']

    @sanity_function
    def validate_build(self):
        return True


@rfm.simple_test
class stream_test(rfm.RunOnlyRegressionTest):
    stream_binaries = fixture(stream_build, scope='environment')
    valid_systems = ['*']
    valid_prog_environs = ['+openmp']
    reference = {
        'tresa': {
            'copy_bandwidth': (23000, -0.05, None, 'MB/s')
        }
    }

    @run_before('run')
    def setup_omp_env(self):
        self.executable = os.path.join(self.stream_binaries.stagedir, 'stream')
        procinfo = self.current_partition.processor
        self.num_cpus_per_task = procinfo.num_cores
        self.env_vars = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'OMP_PLACES': 'cores'
        }

    @sanity_function
    def validate_solution(self):
        return sn.assert_found(r'Solution Validates', self.stdout)

    @performance_function('MB/s')
    def copy_bandwidth(self):
        return sn.extractsingle(r'Copy:\s+(\S+)\s+.*', self.stdout, 1, float)


@rfm.simple_test
class stream_scale_test(stream_test):
    num_threads = parameter([1, 2, 4, 8, 16, 32])
    reference = {}

    @run_before('run')
    def set_cpus_per_task(self):
        self.num_cpus_per_task = self.num_threads
        self.env_vars['OMP_NUM_THREADS'] = str(self.num_cpus_per_task)

    @run_after('setup')
    def skip_if_too_large(self):
        procinfo = self.current_partition.processor
        self.skip_if(self.num_threads > procinfo.num_cores, 'not enough cores')
