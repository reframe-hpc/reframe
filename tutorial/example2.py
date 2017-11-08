import os
import reframe.utility.sanity as sn

from reframe.core.pipeline import RegressionTest


class OpenMPTestIfElse(RegressionTest):
    def __init__(self, **kwargs):
        super().__init__('example2a_check',
                         os.path.dirname(__file__), **kwargs)
        self.descr = 'Matrix-vector multiplication example with OpenMP'
        self.valid_systems = ['*']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi']
        self.sourcepath = 'example_matrix_vector_multiplication_openmp.c'
        self.executable_opts = ['1024', '100']
        self.variables = {
            'OMP_NUM_THREADS': '4'
        }
        self.sanity_patterns = sn.assert_found(
            r'time for single matrix vector multiplication', self.stdout)
        self.maintainers = ['you-can-type-your-email-here']
        self.tags = {'tutorial'}

    def compile(self):
        env_name = self.current_environ.name
        if env_name == 'PrgEnv-cray':
            self.current_environ.cflags = '-homp'
        elif env_name == 'PrgEnv-gnu':
            self.current_environ.cflags = '-fopenmp'
        elif env_name == 'PrgEnv-intel':
            self.current_environ.cflags = '-openmp'
        elif env_name == 'PrgEnv-pgi':
            self.current_environ.cflags = '-mp'

        super().compile()


class OpenMPTestDict(RegressionTest):
    def __init__(self, **kwargs):
        super().__init__('example2b_check',
                         os.path.dirname(__file__), **kwargs)
        self.descr = 'Matrix-vector multiplication example with OpenMP'
        self.valid_systems = ['*']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi']
        self.sourcepath = 'example_matrix_vector_multiplication_openmp.c'
        self.executable_opts = ['1024', '100']
        self.prgenv_flags = {
            'PrgEnv-cray':  '-homp',
            'PrgEnv-gnu':   '-fopenmp',
            'PrgEnv-intel': '-openmp',
            'PrgEnv-pgi':   '-mp'
        }
        self.variables = {
            'OMP_NUM_THREADS': '4'
        }
        self.sanity_patterns = sn.assert_found(
            r'time for single matrix vector multiplication', self.stdout)
        self.maintainers = ['you-can-type-your-email-here']
        self.tags = {'tutorial'}

    def compile(self):
        prgenv_flags = self.prgenv_flags[self.current_environ.name]
        self.current_environ.cflags = prgenv_flags
        super().compile()


def _get_checks(**kwargs):
    return [OpenMPTestIfElse(**kwargs), OpenMPTestDict(**kwargs)]
