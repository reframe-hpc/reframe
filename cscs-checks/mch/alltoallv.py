import reframe as rfm
import reframe.utility.sanity as sn


@rfm.parameterized_test(['default'], ['nocomm'], ['nocomp'])
class Alltoallv(rfm.RegressionTest):
    def __init__(self, variant):
        super().__init__()
        self.valid_systems = ['dom:gpu', 'daint:gpu', 'kesch:cn']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.variables = {'G2G': '1'}
        self.executable = 'build/src/comm_overlap_benchmark'
        if variant != 'default':
            self.executable_opts = ['--' + variant]

        self.sourcesdir = 'https://github.com/cosunae/comm_overlap_bench'
        # Checkout to the branch alltoallv
        self.prebuild_cmd = ['git checkout alltoallv']
        self.build_system = 'CMake'
        self.build_system.builddir = 'build'
        self.build_system.config_opts = ['-DCMAKE_BUILD_TYPE=Release',
                                         '-DENABLE_MPI_TIMER=ON']

        if self.current_system.name == 'kesch':
            self.num_tasks = 144
            self.num_gpus_per_node = 16
            self.num_tasks_per_node = 16
            self.num_tasks_per_socket = 8
            self.modules = ['craype-accel-nvidia35', 'cmake']
            self.variables['MV2_USE_CUDA'] = '1'
            self.build_system.config_opts += [
                '-DMPI_VENDOR=mvapich2',
                '-DCUDA_COMPUTE_CAPABILITY="sm_37"'
            ]
            self.build_system.max_concurrency = 1
        else:
            self.num_tasks = 4
            self.num_gpus_per_node = 1
            self.num_tasks_per_node = 1
            self.modules = ['craype-accel-nvidia60', 'CMake']
            self.variables['MPICH_RDMA_ENABLED_CUDA'] = '1'
            self.build_system.config_opts += [
                '-DCUDA_COMPUTE_CAPABILITY="sm_60"'
            ]
            self.build_system.max_concurrency = 8

        self.sanity_patterns = sn.assert_found(r'ELAPSED TIME:', self.stdout)
        self.perf_patterns = {
            'elapsed_time': sn.extractsingle(r'ELAPSED TIME:\s+(\S+)',
                                             self.stdout, 1, float, 1)
        }
        ref_values = {
            'kesch': {
                'nocomm':  5.7878,
                'nocomp':  5.62155,
                'default': 5.53777
            },
            'daint': {
                'nocomm':  0.0171947,
                'nocomp':  0.0137893,
                'default': 0.0138493
            }
        }
        if self.current_system.name == 'dom':
            sysname = 'daint'
        else:
            sysname = self.current_system.name

        ref = ref_values[sysname][variant]
        self.reference = {
            'kesch:cn': {
                'elapsed_time': (ref, None, 0.15)
            },
            '*': {
                'elapsed_time': (ref, None, 0.15)
            }
        }

        self.maintainers = ['AJ', 'VK']
        self.tags = {'production'}

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)
        if self.current_system.name == 'kesch':
            self.job.launcher.options = ['--distribution=block:block',
                                         '--cpu_bind=q']
