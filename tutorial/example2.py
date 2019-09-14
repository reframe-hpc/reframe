import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class Example2aTest(rfm.RegressionTest):
    def __init__(self):
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

    @rfm.run_before('compile')
    def setflags(self):
        env = self.current_environ.name
        if env == 'PrgEnv-cray':
            self.build_system.cflags = ['-homp']
        elif env == 'PrgEnv-gnu':
            self.build_system.cflags = ['-fopenmp']
        elif env == 'PrgEnv-intel':
            self.build_system.cflags = ['-openmp']
        elif env == 'PrgEnv-pgi':
            self.build_system.cflags = ['-mp']


@rfm.simple_test
class Example2bTest(rfm.RegressionTest):
    def __init__(self):
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

    @rfm.run_before('compile')
    def setflags(self):
        self.build_system.cflags = self.prgenv_flags[self.current_environ.name]
