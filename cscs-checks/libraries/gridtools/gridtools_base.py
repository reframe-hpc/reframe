import reframe as rfm
import reframe.utility.sanity as sn


class GridToolsCheckBase(rfm.RegressionTest):
    def __init__(self):
        super().__init__()
        self.descr = 'GridTools test base'

        self.valid_prog_environs = ['PrgEnv-gnu']
        self.modules = ['CMake', 'Boost']
        self.sourcesdir = '/scratch/snx3000/bignamic/gridtools/' #'git@github.com:eth-cscs/gridtools'
#        self.prebuild_cmd = ['git checkout tags/1.07.00'] # More recent versions require CMake >= 3.12
        self.build_system = 'CMake'
        self.build_system.config_opts = ['-DCMAKE_INSTALL_PREFIX=/scratch/snx3000/bignamic/reframe/cscs-checks/libraries/gridtools/install',
                                         '-DBoost_NO_BOOST_CMAKE=ON',
                                         '-DGT_ENABLE_TARGET_CUDA=OFF',
                                         '-DGT_ENABLE_TARGET_MC=ON',
                                         '-DCUDA_ARCH:STRING=NONE',
                                         '-DCMAKE_BUILD_TYPE:STRING="Release"',
                                         '-DBUILD_TESTING=OFF',
                                         '-DBUILD_SHARED_LIBS=OFF',
                                         '-DGT_USE_MPI=ON',
                                         '-DBOOST_ROOT=/apps/daint/UES/jenkins/6.0.UP07/gpu/easybuild/software/Boost/1.67.0-CrayGNU-18.08-python3',
                                         '-DBOOST_INCLUDEDIR=/apps/daint/UES/jenkins/6.0.UP07/gpu/easybuild/software/Boost/1.67.0-CrayGNU-18.08-python3/include/boost'
                                         '-DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON']

        self.sanity_patterns = sn.assert_found(r'PASSED', self.stdout)
        self.perf_patterns = {
            'perf': sn.extractsingle(r'(?P<timer>\w+) ms total',
                                     self.stdout, 'timer', int)
        }
        self.build_system.max_concurrency = 2
        self.tags = {'production'}
        self.maintainers = ['CB']

@rfm.parameterized_test(['VerticalAdvectionHostNaive'], ['VerticalAdvectionHostBlock'], 
                        ['SimpleHoriDiffHostNaive'], ['SimpleHoriDiffHostBlock'])
class GridToolsCheckCPU(GridToolsCheckBase):
    def __init__(self, test_name):
        super().__init__()
        self.descr = 'GridTools host test'

        self.valid_systems = ['daint:mc', 'daint:gpu']
        self.num_gpus_per_node = 0
        self.num_tasks = 1

        if test_name == 'VerticalAdvectionHostNaive':
            self.build_system.make_opts = ['vertical_advection_dycore_host_naive']
            self.executable = 'examples/vertical_advection_dycore_host_naive'
            self.executable_opts = ['200', '200', '200']
            self.reference = {
                'daint:mc': {
                    'perf': (4800, None, 0.1)
                },
                'daint:gpu': {
                    'perf': (4500, None, 0.1)
                }
            }
        elif test_name == 'VerticalAdvectionHostBlock':
            self.build_system.make_opts = ['vertical_advection_dycore_host_block']
            self.executable = 'examples/vertical_advection_dycore_host_block'
            self.executable_opts = ['200', '200', '200']
            self.reference = {
                'daint:mc': {
                    'perf': (4800, None, 0.1)
                },
                'daint:gpu': {
                    'perf': (4500, None, 0.1)
                }
        }
        elif test_name == 'SimpleHoriDiffHostNaive':
            self.build_system.make_opts = ['simple_hori_diff_host_naive']
            self.executable = 'examples/simple_hori_diff_host_naive'
            self.executable_opts = ['400', '400', '400']
            self.reference = {
                'daint:mc': {
                    'perf': (12000, None, 0.1)
                },
                'daint:gpu': {
                    'perf': (12000, None, 0.1)
                }
        }
        elif test_name == 'SimpleHoriDiffHostBlock':
            self.build_system.make_opts = ['simple_hori_diff_host_block']
            self.executable = 'examples/simple_hori_diff_host_block'
            self.executable_opts = ['400', '400', '400']
            self.reference = {
                'daint:mc': {
                    'perf': (9000, None, 0.1)
                },
                'daint:gpu': {
                    'perf': (9000, None, 0.1)
                }
        }

