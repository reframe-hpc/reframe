import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class GpuDirectCudaCheck(rfm.RegressionTest):
    def __init__(self, **kwargs):
        super().__init__()
        self.descr = 'tests gpu-direct for CUDA'
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'kesch:cn']
        self.valid_prog_environs = ['PrgEnv-gnu']
        if self.current_system.name in ['daint', 'dom']:
            self.variables = {'MPICH_RDMA_ENABLED_CUDA': '1'}
        elif self.current_system.name in ['kesch']:
            self.valid_prog_environs = ['PrgEnv-gnu-gdr']
            self.variables = {
                'MPICH_RDMA_ENABLED_CUDA': '1',
                'MV2_USE_CUDA': '1',
                'G2G': '1',
            }

        self.num_tasks = 2
        self.num_gpus_per_node = 1
        self.sourcepath = 'gpu_direct_cuda.cu'
        self.num_tasks_per_node = 1
        self.modules = ['cudatoolkit']

        result = sn.extractsingle(r'Result :\s+(?P<result>\d+\.?\d*)',
                                  self.stdout, 'result', float)
        self.sanity_patterns = sn.assert_reference(result, 1., -1e-5, 1e-5)
        self.maintainers = ['AJ', 'VK']
        self.tags = {'production'}
        self.pre_run = [
            'export LD_PRELOAD='
            '$(pkg-config --variable=libdir mvapich2-gdr)/libmpi.so'
        ]

    def compile(self):
        # Set nvcc flags
        if self.current_system.name == 'kesch':
            nvidia_sm = '37'
            cpp_compiler = 'mpicxx'
        else:
            nvidia_sm = '60'
            cpp_compiler = 'CC'

        self.current_environ.cxxflags = (
            '-ccbin %s -lcublas -lcudart -arch=sm_%s' %
            (cpp_compiler, nvidia_sm))
        super().compile()
