import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.16-dev0')
@rfm.parameterized_test(['nompi'], ['mpi'])
class FFTWTest(rfm.RegressionTest):
    def __init__(self, exec_mode):
        super().__init__()
        self.sourcepath = 'fftw_benchmark.c'
        self.build_system = 'SingleSource'
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'kesch:cn']
        self.modules = ['cray-fftw']
        self.num_tasks_per_node = 12
        self.num_gpus_per_node = 0
        self.sanity_patterns = sn.assert_eq(
            sn.count(sn.findall(r'execution time', self.stdout)), 1)
        self.build_system.cflags = ['-O2']
        if self.current_system.name == 'kesch':
            self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-pgi']
            self.build_system.cflags += ['-I$FFTW_INC', '-L$FFTW_DIR',
                                         '-lfftw3']
        else:
            self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-pgi',
                                        'PrgEnv-gnu']

        self.perf_patterns = {
            'fftw_exec_time': sn.extractsingle(
                r'execution time:\s+(?P<exec_time>\S+)', self.stdout,
                'exec_time', float),
        }

        if exec_mode == 'nompi':
            self.num_tasks = 12
            self.executable_opts = ['72 12 1000 0']
            self.reference = {
                'dom:gpu': {
                    'fftw_exec_time': (0.55, None, 0.05, 's'),
                },
                'daint:gpu': {
                    'fftw_exec_time': (0.55, None, 0.05, 's'),
                },
                'kesch:cn': {
                    'fftw_exec_time': (0.61, None, 0.05, 's'),
                },
                '*': {
                    'fftw_exec_time': (0, None, None, 's'),
                }
            }
        else:
            self.num_tasks = 72
            self.executable_opts = ['144 72 200 1']
            self.reference = {
                'dom:gpu': {
                    'fftw_exec_time': (0.47, None, 0.50, 's'),
                },
                'daint:gpu': {
                    'fftw_exec_time': (0.47, None, 0.50, 's'),
                },
                'kesch:cn': {
                    'fftw_exec_time': (1.58, None, 0.50, 's'),
                },
                '*': {
                    'fftw_exec_time': (0, None, None, 's'),
                }
            }

        self.maintainers = ['AJ']
        self.tags = {'benchmark', 'scs'}
