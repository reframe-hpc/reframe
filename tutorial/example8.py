import reframe as rfm
import reframe.utility.sanity as sn


class BaseMatrixVectorTest(rfm.RegressionTest):
    def __init__(self, test_version):
        super().__init__()
        self.descr = '%s matrix-vector multiplication' % test_version
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.build_system = 'SingleSource'
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

    def setup(self, partition, environ, **job_opts):
        if self.prgenv_flags is not None:
            self.build_system.cflags = self.prgenv_flags[environ.name]

        super().setup(partition, environ, **job_opts)


@rfm.simple_test
class SerialTest(BaseMatrixVectorTest):
    def __init__(self):
        super().__init__('Serial')
        self.sourcepath = 'example_matrix_vector_multiplication.c'


@rfm.simple_test
class OpenMPTest(BaseMatrixVectorTest):
    def __init__(self):
        super().__init__('OpenMP')
        self.sourcepath = 'example_matrix_vector_multiplication_openmp.c'
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi']
        self.prgenv_flags = {
            'PrgEnv-cray':  ['-homp'],
            'PrgEnv-gnu':   ['-fopenmp'],
            'PrgEnv-intel': ['-openmp'],
            'PrgEnv-pgi':   ['-mp']
        }
        self.variables = {
            'OMP_NUM_THREADS': '4'
        }


@rfm.simple_test
class MPITest(BaseMatrixVectorTest):
    def __init__(self):
        super().__init__('MPI')
        self.valid_systems = ['daint:gpu', 'daint:mc']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi']
        self.sourcepath = 'example_matrix_vector_multiplication_mpi_openmp.c'
        self.prgenv_flags = {
            'PrgEnv-cray':  ['-homp'],
            'PrgEnv-gnu':   ['-fopenmp'],
            'PrgEnv-intel': ['-openmp'],
            'PrgEnv-pgi':   ['-mp']
        }
        self.num_tasks = 8
        self.num_tasks_per_node = 2
        self.num_cpus_per_task = 4
        self.variables = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task)
        }


@rfm.simple_test
class OpenACCTest(BaseMatrixVectorTest):
    def __init__(self):
        super().__init__('OpenACC')
        self.valid_systems = ['daint:gpu']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-pgi']
        self.sourcepath = 'example_matrix_vector_multiplication_openacc.c'
        self.modules = ['craype-accel-nvidia60']
        self.num_gpus_per_node = 1
        self.prgenv_flags = {
            'PrgEnv-cray': ['-hacc', '-hnoomp'],
            'PrgEnv-pgi':  ['-acc', '-ta=tesla:cc60']
        }


@rfm.simple_test
class CudaTest(BaseMatrixVectorTest):
    def __init__(self):
        super().__init__('CUDA')
        self.valid_systems = ['daint:gpu']
        self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-cray', 'PrgEnv-pgi']
        self.sourcepath = 'example_matrix_vector_multiplication_cuda.cu'
        self.modules = ['cudatoolkit']
        self.num_gpus_per_node = 1
