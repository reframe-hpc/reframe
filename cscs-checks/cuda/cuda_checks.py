import os

import reframe.utility.sanity as sn
from reframe.core.pipeline import RegressionTest


class CudaCheck(RegressionTest):
    def __init__(self, name, **kwargs):
        super().__init__('cuda_%s_check' % name,
                         os.path.dirname(__file__), **kwargs)
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'kesch:cn']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu']
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'CUDA', 'essentials')
        self.modules = ['cudatoolkit']
        self.maintainers = ['AJ', 'VK']
        self.num_gpus_per_node = 1
        self.tags = {'production'}

    def compile(self):
        # Set nvcc flags
        nvidia_sm = '60'
        if self.current_system.name == 'kesch':
            nvidia_sm = '37'
        self.current_environ.cxxflags = ('-ccbin g++ -m64 -lcublas '
                                         '-arch=sm_%s' % nvidia_sm)
        super().compile()


class MatrixmulCublasCheck(CudaCheck):
    def __init__(self, **kwargs):
        super().__init__('matrixmulcublas', **kwargs)
        self.descr = 'Implements matrix multiplication using CUBLAS'
        self.sourcepath = 'matrixmulcublas.cu'
        self.sanity_patterns = sn.assert_found(
            r'Comparing CUBLAS Matrix Multiply with CPU results: PASS',
            self.stdout)


class DeviceQueryCheck(CudaCheck):
    def __init__(self, **kwargs):
        super().__init__('devicequery', **kwargs)
        self.descr = 'Queries the properties of the CUDA devices'
        self.sourcepath = 'devicequery.cu'
        self.sanity_patterns = sn.assert_found(r'Result = PASS', self.stdout)


class ConcurrentKernelsCheck(CudaCheck):
    def __init__(self, **kwargs):
        super().__init__('concurrentkernels', **kwargs)
        self.descr = 'Use of streams for concurrent execution'
        self.sourcepath = 'concurrentkernels.cu'
        self.sanity_patterns = sn.assert_found(r'Test passed', self.stdout)


class SimpleMPICheck(CudaCheck):
    def __init__(self, **kwargs):
        super().__init__('simplempi', **kwargs)
        self.descr = 'Simple example demonstrating how to use MPI with CUDA'
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'CUDA', 'simplempi')
        self.executable = 'simplempi'
        self.num_tasks = 2
        self.num_tasks_per_node = 2
        self.sanity_patterns = sn.assert_found(r'Result = PASS', self.stdout)
        if self.current_system.name == 'kesch':
            self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu']
            self.variables = {'G2G': '0'}
            self.num_gpus_per_node = 2
        else:
            self.variables = {'CRAY_CUDA_MPS': '1'}

    def setup(self, partition, environ, **job_opts):
        if (self.current_system.name == 'kesch' and
            environ.name == 'PrgEnv-gnu'):
            self.modules = ['mvapich2gdr_gnu/2.2_cuda_8.0']
        super().setup(partition, environ, **job_opts)


def _get_checks(**kwargs):
    return [ConcurrentKernelsCheck(**kwargs),
            DeviceQueryCheck(**kwargs),
            MatrixmulCublasCheck(**kwargs),
            SimpleMPICheck(**kwargs)]
