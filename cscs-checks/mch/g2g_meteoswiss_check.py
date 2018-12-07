import reframe as rfm
import reframe.utility.sanity as sn


@rfm.parameterized_test([1], [2])
class G2GMeteoswissTest(rfm.RegressionTest):
    def __init__(self, g2g):
        super().__init__()
        self.descr = 'G2G Meteoswiss check with G2G=%s' % g2g
        self.strict_check = False
        self.valid_systems = ['kesch:cn']

        # FIXME: temporary workaround until the mvapich module is fixed;
        #        'PrgEnv-gnu-c2sm-gpu' will be added later
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.exclusive_access = True
        self.modules = ['cmake', 'craype-accel-nvidia35']
        self.pre_run = ["export EXECUTABLE=$(ls src/ | "
                        "grep 'GNU.*MVAPICH.*CUDA.*kesch.*')"]
        self.executable = 'build/src/comm_overlap_benchmark'
        self.sourcesdir = ('https://github.com/MeteoSwiss-APN/'
                           'comm_overlap_bench.git')
        self.prebuild_cmd = ['git checkout barebones']
        self.build_system = 'CMake'
        self.build_system.builddir = 'build'
        self.build_system.config_opts = ['-DMPI_VENDOR=mvapich2',
                                         '-DCUDA_COMPUTE_CAPABILITY="sm_37"',
                                         '-DCMAKE_BUILD_TYPE=Release',
                                         '-DENABLE_MPI_TIMER=ON']
        self.build_system.max_concurrency = 1
        self.maintainers = ['TM', 'JG']
        self.tags = {'production', 'mch'}
        self.num_tasks = 2
        self.num_gpus_per_node  = 2
        cuda_visible_devices = {1: r'CUDA_VISIBLE_DEVICES: '
                                   r'\[0: \d\] \[1: \d\]',
                                2: r'CUDA_VISIBLE_DEVICES: '
                                   r'\[0: \d,\d\] \[1: \d,\d\]'}
        self.sanity_patterns = sn.all([
            sn.assert_found('ELAPSED TIME:', self.stdout),
            sn.assert_found(cuda_visible_devices[g2g], self.stdout)
        ])
        self.perf_patterns = {
            'time': sn.extractsingle(r'ELAPSED TIME:\s+(?P<time>\S+)',
                                     self.stdout, 'time', float)
        }
        self.reference = {
            'kesch:cn': {'time': (3.00, None, 0.2)}
        }
        self.variables = {'G2G': str(g2g)}
