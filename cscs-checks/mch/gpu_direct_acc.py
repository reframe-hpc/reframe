import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.14')
@rfm.simple_test
class GpuDirectAccCheck(rfm.RegressionTest):
    def __init__(self):
        super().__init__()
        self.descr = 'tests gpu-direct for Fortran OpenACC'
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'kesch:cn']

        # FIXME: temporary workaround until the mvapich module is fixed;
        #        'PrgEnv-pgi-c2sm-gpu' will be added later
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-cray-c2sm-gpu',
                                    'PrgEnv-pgi']
        if self.current_system.name in ['daint', 'dom']:
            self.modules = ['craype-accel-nvidia60']
            self.variables = {'MPICH_RDMA_ENABLED_CUDA': '1'}
            self.num_tasks = 2
            self.num_gpus_per_node = 1
            self.num_tasks_per_node = 1
        elif self.current_system.name == 'kesch':
            self.exclusive_access = True
            self.modules = ['craype-accel-nvidia35']
            self.variables = {
                'MV2_USE_CUDA': '1',
                'G2G': '1'
            }
            self.num_tasks = 8
            self.num_gpus_per_node = 8
            self.num_tasks_per_node = 8

        self.sourcepath = 'gpu_direct_acc.F90'
        self.build_system = 'SingleSource'
        self.sanity_patterns = sn.all([
            sn.assert_found(r'GPU with OpenACC', self.stdout),
            sn.assert_found(r'Result :\s+OK', self.stdout)
        ])
        self.launch_options = []
        self.maintainers = ['AJ', 'VK']
        self.tags = {'production', 'mch'}

    def setup(self, partition, environ, **job_opts):
        if environ.name.startswith('PrgEnv-cray'):
            self.build_system.fflags = ['-hacc', '-hnoomp']
        elif environ.name.startswith('PrgEnv-pgi'):
            self.build_system.fflags = ['-acc']
            if self.current_system.name in ['daint', 'dom']:
                self.build_system.fflags += ['-ta=tesla:cc60', '-Mnorpath']
            elif self.current_system.name == 'kesch':
                self.build_system.fflags += ['-ta=tesla:cc35']

        super().setup(partition, environ, **job_opts)
