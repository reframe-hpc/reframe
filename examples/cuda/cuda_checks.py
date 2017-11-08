import os

from reframe.core.pipeline import RegressionTest


class CudaCheck(RegressionTest):
    def __init__(self, name, **kwargs):
        super().__init__('cuda_%s_check' % name,
                         os.path.dirname(__file__), **kwargs)

        # Uncomment and adjust for your site
        # self.valid_systems = ['sys1', 'sys2']

        # Uncomment and set the valid prog. environments for your site
        # self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu']

        # Uncomment and adjust to load CUDA
        # self.modules = ['cudatoolkit']

        self.num_gpus_per_node = 1

        # Uncomment and set the maintainers and/or tags
        # self.maintainers = ['me']
        # self.tags = {'example'}

    def compile(self):
        # Set nvcc flags
        self.current_environ.cxxflags = '-ccbin g++ -m64 -lcublas'
        super().compile()


class MatrixmulCublasCheck(CudaCheck):
    def __init__(self, **kwargs):
        super().__init__('matrixmulcublas', **kwargs)
        self.descr = 'Implements matrix multiplication using CUBLAS'
        self.sourcepath = 'matrixmulcublas.cu'
        self.sanity_patterns = {
            '-': {
                'Comparing CUBLAS Matrix Multiply with CPU results: PASS': []
            }
        }


class BandwidthCheck(CudaCheck):
    def __init__(self, **kwargs):
        super().__init__('bandwidth', **kwargs)
        self.descr = 'CUDA bandwidthTest compile and run'
        self.sourcepath = 'bandwidthTest.cu'
        self.sanity_patterns = {
            '-': {'Result = PASS': []}
        }


class DeviceQueryCheck(CudaCheck):
    def __init__(self, **kwargs):
        super().__init__('devicequery', **kwargs)
        self.descr = 'Queries the properties of the CUDA devices'
        self.sourcepath = 'devicequery.cu'
        self.sanity_patterns = {
            '-': {'Result = PASS': []}
        }


class ConcurrentKernelsCheck(CudaCheck):
    def __init__(self, **kwargs):
        super().__init__('concurrentkernels', **kwargs)
        self.descr = 'Use of streams for concurrent execution'
        self.sourcepath = 'concurrentkernels.cu'
        self.sanity_patterns = {
            '-': {'Test passed': []}
        }


class SimpleMPICheck(CudaCheck):
    def __init__(self, **kwargs):
        super().__init__('simplempi', **kwargs)
        self.descr = 'Simple example demonstrating how to use MPI with CUDA'
        self.sourcepath = 'simplempi/'
        self.executable = 'simplempi'
        self.num_tasks = 2
        self.num_tasks_per_node = 2
        self.sanity_patterns = {
            '-': {'Result = PASS': []}
        }

        # Uncomment for Cray systems
        # self.variables = {'CRAY_CUDA_MPS': '1'}


def _get_checks(**kwargs):
    return [BandwidthCheck(**kwargs),
            ConcurrentKernelsCheck(**kwargs),
            DeviceQueryCheck(**kwargs),
            MatrixmulCublasCheck(**kwargs),
            SimpleMPICheck(**kwargs)]
