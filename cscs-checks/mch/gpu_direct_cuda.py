import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.14')
@rfm.simple_test
class GpuDirectCudaCheck(rfm.RegressionTest):
    def __init__(self):
        super().__init__()
        self.descr = 'tests gpu-direct for CUDA'
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'kesch:cn']
        # FIXME: temporary workaround until the mvapich module is fixed
        #        'PrgEnv-gnu-c2sm-gpu' will be added later
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.sourcepath = 'gpu_direct_cuda.cu'
        self.build_system = 'SingleSource'
        self.build_system.ldflags = ['-lcublas', '-lcudart']
        if self.current_system.name in ['daint', 'dom']:
            self.modules = ['craype-accel-nvidia60']
            self.variables = {'MPICH_RDMA_ENABLED_CUDA': '1'}
            self.build_system.cxxflags = ['-ccbin CC', '-arch=sm_60']
        elif self.current_system.name == 'kesch':
            self.exclusive_access = True
            self.valid_prog_environs = ['PrgEnv-gnu']
            self.modules = ['craype-accel-nvidia35']
            self.variables = {
                'MV2_USE_CUDA': '1',
                'G2G': '1',
            }
            self.build_system.cxxflags = ['-ccbin', 'mpicxx', '-arch=sm_37']

        self.num_tasks = 2
        self.num_gpus_per_node = 1
        self.num_tasks_per_node = 1
        result = sn.extractsingle(r'Result :\s+(?P<result>\d+\.?\d*)',
                                  self.stdout, 'result', float)
        self.sanity_patterns = sn.assert_reference(result, 1., -1e-5, 1e-5)
        self.maintainers = ['AJ', 'VK']
        self.tags = {'production', 'mch'}
