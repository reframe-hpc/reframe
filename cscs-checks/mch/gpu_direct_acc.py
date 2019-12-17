import os
import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.14')
@rfm.simple_test
class GpuDirectAccCheck(rfm.RegressionTest):
    def __init__(self):
        super().__init__()
        self.descr = 'tests gpu-direct for Fortran OpenACC'
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'kesch:cn', 'tiger:gpu',
                              'tsa:cn']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-pgi']
        if self.current_system.name in ['daint', 'dom', 'tiger']:
            self.modules = ['craype-accel-nvidia60']
            self.variables = {
                'MPICH_RDMA_ENABLED_CUDA': '1',
            }

            if self.current_system.name in ['tiger']:
                craypath = '%s:$PATH' % os.environ['CRAY_BINUTILS_BIN']
                self.variables['PATH'] = craypath

            self.num_tasks = 2
            self.num_gpus_per_node = 1
            self.num_tasks_per_node = 1
        elif self.current_system.name == 'kesch':
            self.exclusive_access = True
            self.modules = ['cudatoolkit/8.0.61']
            self.variables = {
                'CRAY_ACCEL_TARGET': 'nvidia35',
                'MV2_USE_CUDA': '1',
                'G2G': '1'
            }
            self.num_tasks = 8
            self.num_gpus_per_node = 8
            self.num_tasks_per_node = 8
        elif self.current_system.name == 'tsa':
            self.exclusive_access = True
            self.variables = {
                'G2G': '1'
            }
            self.num_tasks = 8
            self.num_gpus_per_node = 8
            self.num_tasks_per_node = 8

        self.sourcepath = 'gpu_direct_acc.F90'
        self.build_system = 'SingleSource'
        self.prebuild_cmd = ['module list -l']
        self.sanity_patterns = sn.all([
            sn.assert_found(r'GPU with OpenACC', self.stdout),
            sn.assert_found(r'Result :\s+OK', self.stdout)
        ])
        self.launch_options = []
        self.maintainers = ['AJ', 'VK']
        self.tags = {'production', 'mch', 'craype'}

    def setup(self, partition, environ, **job_opts):
        if environ.name.startswith('PrgEnv-cray'):
            self.build_system.fflags = ['-hacc', '-hnoomp']
        elif environ.name.startswith('PrgEnv-pgi'):
            self.build_system.fflags = ['-acc']
            if self.current_system.name in ['daint', 'dom']:
                self.build_system.fflags += ['-ta=tesla:cc60', '-Mnorpath']
            elif self.current_system.name == 'kesch':
                self.build_system.fflags += ['-ta=tesla:cc35']
            elif self.current_system.name == 'tsa':
                self.build_system.fflags += ['-ta=tesla:cc70']

        super().setup(partition, environ, **job_opts)
