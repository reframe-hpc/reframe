import os

import reframe as rfm
import reframe.utility.sanity as sn


class CudaCheck(rfm.RegressionTest):
    def __init__(self, name):
        super().__init__()
        self.name = 'cuda_%s_check' % name
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'kesch:cn']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu']
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'CUDA', 'essentials')
        self.modules = ['craype-accel-nvidia60']
        self.num_gpus_per_node = 1
        self.nvidia_sm = '60'
        if self.current_system.name == 'kesch':
            self.nvidia_sm = '37'

        self.maintainers = ['AJ', 'VK']
        self.tags = {'production'}


@rfm.simple_test
class MatrixmulCublasCheck(CudaCheck):
    def __init__(self):
        super().__init__('matrixmulcublas')
        self.descr = 'Implements matrix multiplication using CUBLAS'
        self.sourcepath = 'matrixmulcublas.cu'
        self.build_system = 'SingleSource'
        self.build_system.cxxflags = ['-I.', '-ccbin g++ -m64 -lcublas',
                                      '-arch=sm_%s' % self.nvidia_sm]
        self.sanity_patterns = sn.assert_found(
            r'Comparing CUBLAS Matrix Multiply with CPU results: PASS',
            self.stdout)


@rfm.simple_test
class DeviceQueryCheck(CudaCheck):
    def __init__(self):
        super().__init__('devicequery')
        self.descr = 'Queries the properties of the CUDA devices'
        self.sourcepath = 'devicequery.cu'
        self.build_system = 'SingleSource'
        self.build_system.cxxflags = ['-I.', '-ccbin g++ -m64 -lcublas',
                                      '-arch=sm_%s' % self.nvidia_sm]
        self.sanity_patterns = sn.assert_found(r'Result = PASS', self.stdout)


@rfm.simple_test
class ConcurrentKernelsCheck(CudaCheck):
    def __init__(self):
        super().__init__('concurrentkernels')
        self.descr = 'Use of streams for concurrent execution'
        self.sourcepath = 'concurrentkernels.cu'
        self.build_system = 'SingleSource'
        self.build_system.cxxflags = ['-I.', '-ccbin g++ -m64 -lcublas',
                                      '-arch=sm_%s' % self.nvidia_sm]
        self.sanity_patterns = sn.assert_found(r'Test passed', self.stdout)


@rfm.simple_test
class SimpleMPICheck(CudaCheck):
    def __init__(self):
        super().__init__('simplempi')
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

        self.build_system = 'Make'
        self.build_system.cxxflags = ['-I.', '-ccbin g++ -m64 -lcublas',
                                      '-arch=sm_%s' % self.nvidia_sm]

    def setup(self, partition, environ, **job_opts):
        if (self.current_system.name == 'kesch' and
            environ.name == 'PrgEnv-gnu'):
            self.modules = ['mvapich2gdr_gnu/2.2_cuda_8.0']
        super().setup(partition, environ, **job_opts)
