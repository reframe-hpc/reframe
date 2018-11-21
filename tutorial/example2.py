import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class Example2aTest(rfm.RegressionTest):
    def __init__(self):
        super().__init__()
        self.descr = 'Matrix-vector multiplication example with OpenMP'
        self.valid_systems = ['*']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi']
        self.sourcepath = 'example_matrix_vector_multiplication_openmp.c'
        self.build_system = 'SingleSource'
        self.executable_opts = ['1024', '100']
        self.variables = {
            'OMP_NUM_THREADS': '4'
        }
        self.sanity_patterns = sn.assert_found(
            r'time for single matrix vector multiplication', self.stdout)
        self.maintainers = ['you-can-type-your-email-here']
        self.tags = {'tutorial'}

    def setup(self, partition, environ, **job_opts):
        if environ.name == 'PrgEnv-cray':
            self.build_system.cflags = ['-homp']
        elif environ.name == 'PrgEnv-gnu':
            self.build_system.cflags = ['-fopenmp']
        elif environ.name == 'PrgEnv-intel':
            self.build_system.cflags = ['-openmp']
        elif environ.name == 'PrgEnv-pgi':
            self.build_system.cflags = ['-mp']

        super().setup(partition, environ, **job_opts)


@rfm.simple_test
class Example2bTest(rfm.RegressionTest):
    def __init__(self, **kwargs):
        super().__init__()
        self.descr = 'Matrix-vector multiplication example with OpenMP'
        self.valid_systems = ['*']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi']
        self.sourcepath = 'example_matrix_vector_multiplication_openmp.c'
        self.build_system = 'SingleSource'
        self.executable_opts = ['1024', '100']
        self.prgenv_flags = {
            'PrgEnv-cray':  ['-homp'],
            'PrgEnv-gnu':   ['-fopenmp'],
            'PrgEnv-intel': ['-openmp'],
            'PrgEnv-pgi':   ['-mp']
        }
        self.variables = {
            'OMP_NUM_THREADS': '4'
        }
        self.sanity_patterns = sn.assert_found(
            r'time for single matrix vector multiplication', self.stdout)
        self.maintainers = ['you-can-type-your-email-here']
        self.tags = {'tutorial'}

    def setup(self, partition, environ, **job_opts):
        self.build_system.cflags = self.prgenv_flags[environ.name]
        super().setup(partition, environ, **job_opts)
