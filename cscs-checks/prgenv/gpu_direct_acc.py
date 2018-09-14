import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class GpuDirectAccCheck(rfm.RegressionTest):
    def __init__(self, **kwargs):
        super().__init__()
        self.descr = 'tests gpu-direct for Fortran OpenACC'
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'kesch:cn']
        self.valid_prog_environs = ['PrgEnv-cray*', 'PrgEnv-pgi*']
        if self.current_system.name in ['daint', 'dom']:
            self.modules = ['craype-accel-nvidia60']
            self._pgi_flags = '-acc -ta=tesla:cc60 -Mnorpath'
            self.variables = {'MPICH_RDMA_ENABLED_CUDA': '1'}
            self.num_tasks = 2
            self.num_gpus_per_node = 1
            self.num_tasks_per_node = 1
        elif self.current_system.name in ['kesch']:
            self.modules = ['craype-accel-nvidia35']
            self._pgi_flags = '-acc -ta=tesla:cc35'
            self.variables = {
                'MPICH_RDMA_ENABLED_CUDA': '1',
                'MV2_USE_CUDA': '1',
                'G2G': '1'
            }
            self.num_tasks = 8
            self.num_gpus_per_node = 8
            self.num_tasks_per_node = 8

        self.sourcepath = 'gpu_direct_acc.F90'
        self.sanity_patterns = sn.all([
            sn.assert_found(r'GPU with OpenACC', self.stdout),
            sn.assert_found(r'Result :\s+OK', self.stdout)
        ])
        self.launch_options = []
        self.maintainers = ['AJ', 'VK']
        self.tags = {'production'}

    def setup(self, partition, environ, **job_opts):
        if environ.name.startswith('PrgEnv-cray'):
            environ.fflags = '-hacc -hnoomp'
        elif environ.name.startswith('PrgEnv-pgi'):
            environ.fflags = self._pgi_flags

        super().setup(partition, environ, **job_opts)
