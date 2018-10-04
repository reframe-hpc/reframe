import reframe as rfm
import reframe.utility.sanity as sn


@rfm.parameterized_test([''], ['--nocomm'], ['--nocomp'])
class Alltoallv(rfm.RegressionTest):
    def __init__(self, exec_parameter):
        super().__init__()
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'kesch:cn']
        self.valid_prog_environs = ['PrgEnv-gnu']
        if self.current_system.name in ['daint', 'dom']:
            self.modules = ['craype-accel-nvidia60']
            self._pgi_flags = ['-acc', '-ta=tesla:cc60', '-Mnorpath']
        elif self.current_system.name in ['kesch']:
            self.modules = ['craype-accel-nvidia35']
            self._pgi_flags = ['-O2', '-ta=tesla,cc35,cuda8.0']

        self.num_tasks = 144
        self.num_gpus_per_node = 16
        self.num_tasks_per_node = 16
        self.num_tasks_per_socket = 8
        self.build_system = 'CMake'
        self.build_system.builddir = 'build'
        self.build_system.config_opts = ['-DMPI_VENDOR=mvapich2',
                                         '-DCUDA_COMPUTE_CAPABILITY="sm_37"',
                                         '-DCMAKE_BUILD_TYPE=Release',
                                         '-DENABLE_MPI_TIMER=ON']
        self.executable = 'build/src/comm_overlap_benchmark'
        self.executable_opts = [exec_parameter]
        self.sourcesdir = 'https://github.com/cosunae/comm_overlap_bench'
        self.prebuild_cmd = ['git checkout alltoallv']
        self.sourcepath = 'src'
        self.sanity_patterns = sn.assert_found(r'ELAPSED TIME:', self.stdout)
        self.perf_patterns = {
            'perf': sn.extractsingle(r'ELAPSED TIME:\s+(?P<perf>\S+)',
                                     self.stdout, 'perf', float, 1)
        }

        if exec_parameter == '':
            self.reference = {
                'kesch:cn': {
                    'perf': (5.53777, None, 0.15)
                },
            }
        elif exec_parameter == '--nocomm':
            self.reference = {
                'kesch:cn': {
                    'perf': (5.7878, None, 0.15)
                },
            }
        elif exec_parameter == '--nocomp':
            self.reference = {
                'kesch:cn': {
                    'perf': (5.62155, None, 0.15)
                },
            }

        self.modules += [
            'craype-haswell', 'craype-network-infiniband',
            'mvapich2gdr_gnu/2.2_cuda_8.0', 'cray-libsci_acc/17.03.1', 'cmake'
        ]

        self.variables = {
            'G2G': '1',
            'jobs': '144',
            'RDMA_FAST_PATH': '0',
            'MV2_USE_CUDA': '1'
        }

        self.pre_run = [
            'export BOOST_LIBRARY_PATH=/apps/escha/UES/PrgEnv-gnu-17.02/modulefiles/boost/1.63.0-gmvolf-17.02-python-2.7.13/lib',
            'export LD_LIBRARY_PATH=$BOOST_LIBRARY_PATH:$LD_LIBRARY_PATH',
            'export XXX_LIBRARY_PATH=/apps/escha/UES/RH7.3_experimental/pgi/17.10/linux86-64/17.10/REDIST',
            'export LD_LIBRARY_PATH=$XXX_LIBRARY_PATH:$LD_LIBRARY_PATH',
            'export LD_PRELOAD=/opt/mvapich2/gdr/2.3a/mcast/no-openacc/cuda8.0/mofed3.4/mpirun/pgi17.10/lib64/libmpi.so',
        ]

        self.extra_resources = {
            'distribution': {},
            'cpu_bind': {}
        }

        self.maintainers = ['AJ', 'VK']
        self.tags = {'production'}

    def setup(self, partition, environ, **job_opts):
        if environ.name.startswith('PrgEnv-cray'):
            self.build_system.fflags = ['-O2', '-hacc', '-hnoomp']
        elif environ.name.startswith('PrgEnv-pgi'):
            self.build_system.fflags = self._pgi_flags
        elif environ.name.startswith('PrgEnv-gnu'):
            self.build_system.fflags = ['-O2']

        super().setup(partition, environ, **job_opts)
