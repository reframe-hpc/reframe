import os
import reframe.utility.sanity as sn

from reframe.core.pipeline import RegressionTest


class BaseMatrixVectorTest(RegressionTest):
    def __init__(self, test_version, **kwargs):
        super().__init__('example8_' + test_version.lower() + '_check',
                         os.path.dirname(__file__), **kwargs)
        self.descr = '%s matrix-vector multiplication' % test_version
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.prgenv_flags = None

        matrix_dim = 1024
        iterations = 100
        self.executable_opts = [str(matrix_dim), str(iterations)]

        expected_norm = matrix_dim
        found_norm = sn.extractsingle(
            r'The L2 norm of the resulting vector is:\s+(?P<norm>\S+)',
            self.stdout, 'norm', float)
        self.sanity_patterns = sn.all([
            sn.assert_found(
                r'time for single matrix vector multiplication', self.stdout),
            sn.assert_lt(sn.abs(expected_norm - found_norm), 1.0e-6)
        ])
        self.maintainers = ['you-can-type-your-email-here']
        self.tags = {'tutorial'}

    def compile(self):
        if self.prgenv_flags is not None:
            self.current_environ.cflags = self.prgenv_flags[self.current_environ.name]

        super().compile()


class SerialTest(BaseMatrixVectorTest):
    def __init__(self, **kwargs):
        super().__init__('Serial', **kwargs)
        self.sourcepath = 'example_matrix_vector_multiplication.c'


class OpenMPTest(BaseMatrixVectorTest):
    def __init__(self, **kwargs):
        super().__init__('OpenMP', **kwargs)
        self.sourcepath = 'example_matrix_vector_multiplication_openmp.c'
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi']
        self.prgenv_flags = {
            'PrgEnv-cray':  '-homp',
            'PrgEnv-gnu':   '-fopenmp',
            'PrgEnv-intel': '-openmp',
            'PrgEnv-pgi':   '-mp'
        }
        self.variables = {
            'OMP_NUM_THREADS': '4'
        }


class MPITest(BaseMatrixVectorTest):
    def __init__(self, **kwargs):
        super().__init__('MPI', **kwargs)
        self.valid_systems = ['daint:gpu', 'daint:mc']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi']
        self.sourcepath = 'example_matrix_vector_multiplication_mpi_openmp.c'
        self.prgenv_flags = {
            'PrgEnv-cray':  '-homp',
            'PrgEnv-gnu':   '-fopenmp',
            'PrgEnv-intel': '-openmp',
            'PrgEnv-pgi':   '-mp'
        }
        self.num_tasks = 8
        self.num_tasks_per_node = 2
        self.num_cpus_per_task = 4
        self.variables = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task)
        }


class OpenACCTest(BaseMatrixVectorTest):
    def __init__(self, **kwargs):
        super().__init__('OpenACC', **kwargs)
        self.valid_systems = ['daint:gpu']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-pgi']
        self.sourcepath = 'example_matrix_vector_multiplication_openacc.c'
        self.modules = ['craype-accel-nvidia60']
        self.num_gpus_per_node = 1
        self.prgenv_flags = {
            'PrgEnv-cray': '-hacc -hnoomp',
            'PrgEnv-pgi':  '-acc -ta=tesla:cc60'
        }


class CudaTest(BaseMatrixVectorTest):
    def __init__(self, **kwargs):
        super().__init__('CUDA', **kwargs)
        self.valid_systems = ['daint:gpu']
        self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-cray', 'PrgEnv-pgi']
        self.sourcepath = 'example_matrix_vector_multiplication_cuda.cu'
        self.modules = ['cudatoolkit']
        self.num_gpus_per_node = 1


def _get_checks(**kwargs):
    return [SerialTest(**kwargs), OpenMPTest(**kwargs), MPITest(**kwargs),
            OpenACCTest(**kwargs), CudaTest(**kwargs)]
