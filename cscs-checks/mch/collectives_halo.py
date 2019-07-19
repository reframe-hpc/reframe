import reframe as rfm
import reframe.utility.sanity as sn


class CommunicationTestBase(rfm.RegressionTest):
    def __init__(self, variant, bench_reference):
        super().__init__()
        self.valid_systems = ['dom:gpu', 'daint:gpu', 'kesch:cn', 'arolla:cn', 'tsa:cn']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.variables = {'G2G': '1'}
        self.executable = 'build/src/comm_overlap_benchmark'
        if variant != 'default':
            self.executable_opts = ['--' + variant]

        self.build_system = 'CMake'
        self.build_system.builddir = 'build'
        self.build_system.config_opts = ['-DCMAKE_BUILD_TYPE=Release',
                                         '-DENABLE_MPI_TIMER=ON']

        if self.current_system.name == 'kesch':
            self.exclusive_access = True
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
        elif self.current_system.name in {'daint', 'dom'}:
            self.num_tasks = 4
            self.num_gpus_per_node = 1
            self.num_tasks_per_node = 1
            self.modules = ['craype-accel-nvidia60', 'CMake']
            self.variables['MPICH_RDMA_ENABLED_CUDA'] = '1'
            self.build_system.config_opts += [
                '-DCUDA_COMPUTE_CAPABILITY="sm_60"'
            ]
            self.build_system.max_concurrency = 8
        elif self.current_system.name == 'arolla':
            self.exclusive_access = True
            self.num_tasks = 16
            self.num_gpus_per_node = 8
            self.modules = [
                'cmake',
                'cuda92/toolkit/9.2.88',
            ]
            self.variables['MV2_USE_CUDA'] = '1'
            self.variables['MPICH_RDMA_ENABLED_CUDA'] = '1'
            self.build_system.config_opts += [
                '-DMPI_VENDOR=openmpi',
                '-DCUDA_COMPUTE_CAPABILITY="sm_70"'
            ]
        elif self.current_system.name == 'tsa':
            self.exclusive_access = True
            self.num_tasks = 16
            self.num_gpus_per_node = 8
            self.modules = [
                'cmake',
                'cuda10.0/toolkit/10.0.130',
            ]
            self.variables['MV2_USE_CUDA'] = '1'
            self.variables['MPICH_RDMA_ENABLED_CUDA'] = '1'
            self.build_system.config_opts += [
                #'-DMPI_VENDOR=mvapich2',
                '-DCUDA_COMPUTE_CAPABILITY="sm_70"'
            ]
        else:
            self.num_tasks = 4
            self.num_gpus_per_node = 1
            self.num_tasks_per_node = 1
            self.build_system.max_concurrency = 1

        self.sanity_patterns = sn.assert_found(r'ELAPSED TIME:', self.stdout)
        self.perf_patterns = {
            'elapsed_time': sn.extractsingle(r'ELAPSED TIME:\s+(\S+)',
                                             self.stdout, 1, float, -1)
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

        try:
            ref = bench_reference[sysname][variant]
        except KeyError:
            ref = 0.0

        self.reference = {
            'kesch:cn': {
                'elapsed_time': (ref, None, 0.15)
            },
            'arolla:cn': {
                'elapsed_time': (ref, None, 0.15)
            },
            'tsa:cn': {
                'elapsed_time': (ref, None, 0.15)
            },
            'daint': {
                'elapsed_time': (ref, None, 0.15)
            },
            'dom': {
                'elapsed_time': (ref, None, 0.15)
            },
            '*': {
                'elapsed_time': (ref, None, None)
            }
        }

        self.maintainers = ['AJ', 'VK']
        self.tags = {'production', 'mch'}

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)
        #if self.current_system.name == 'kesch':
        if self.current_system.name in {'kesch', 'tsa', 'arolla'}:
            self.job.launcher.options = ['--distribution=block:block',
                                         '--cpu_bind=q']


# the values default, nocomm and nocomp refer to the different parts
# of the check where the time is measured; default == all
# nocomm == no communication  nocomp == no computation
@rfm.parameterized_test(['default'], ['nocomm'], ['nocomp'])
class AlltoallvTest(CommunicationTestBase):
    def __init__(self, variant):
        super().__init__(variant,
                         {
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
                         })
        self.descr = 'Alltoall communication test'
        self.strict_check = False
        self.sourcesdir = 'https://github.com/cosunae/comm_overlap_bench'
        self.prebuild_cmd = ['git checkout alltoallv']


@rfm.parameterized_test(['default'], ['nocomm'], ['nocomp'])
class HaloExchangeTest(CommunicationTestBase):
    def __init__(self, variant):
        super().__init__(variant,
                         {
                             'kesch': {
                                 'nocomm':  5.7878,
                                 'nocomp':  5.62155,
                                 'default': 5.53777
                             },
                             'daint': {
                                 'nocomm':  0.978306,
                                 'nocomp':  1.36716,
                                 'default': 2.53509
                             }
                         })
        self.descr = 'Halo-cell exchange test'
        self.sourcesdir = 'https://github.com/MeteoSwiss-APN/comm_overlap_bench.git'
        self.prebuild_cmd = ['git checkout barebones']
