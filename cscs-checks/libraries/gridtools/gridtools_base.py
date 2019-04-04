import reframe as rfm
import reframe.utility.sanity as sn


class GridToolsCheckBase(rfm.RegressionTest):
    def __init__(self):
        super().__init__()
        self.descr = 'GridTools test base'

        self.valid_prog_environs = ['PrgEnv-gnu']
        self.prebuild_cmd = ['module switch gcc/6.2.0 gcc/5.3.0']# TODO: fix this
        self.modules = ['/scratch/snx3000/bignamic/easybuild_install/modules/all/CMake/3.12.4', 'Boost']
        self.sourcesdir = '/scratch/snx3000/bignamic/gridtools/' #'git@github.com:eth-cscs/gridtools'
#        self.prebuild_cmd = ['git checkout tags/1.07.00'] # More recent versions require CMake >= 3.12
        self.build_system = 'CMake'
        self.build_system.config_opts = ['-DCMAKE_INSTALL_PREFIX=/scratch/snx3000/bignamic/reframe/cscs-checks/libraries/gridtools/install',
                                         '-DBoost_NO_BOOST_CMAKE="true"',
                                         '-DCMAKE_BUILD_TYPE:STRING=Release',
                                         '-DBUILD_SHARED_LIBS:BOOL=ON',
                                         '-DGT_ENABLE_TARGET_X86:BOOL=ON',
                                         '-DGT_ENABLE_TARGET_NAIVE:BOOL=ON',
                                         '-DGT_ENABLE_TARGET_CUDA:BOOL=OFF',
                                         '-DGT_ENABLE_TARGET_MC=ON',
                                         '-DGT_GCL_ONLY:BOOL=OFF',
                                         '-DCMAKE_CXX_COMPILER=CC',
                                         '-DGT_USE_MPI:BOOL=OFF', 
                                         '-DGT_SINGLE_PRECISION:BOOL=OFF', 
                                         '-DGT_ENABLE_PERFORMANCE_METERS:BOOL=ON',
                                         '-DGT_TESTS_ICOSAHEDRAL_GRID:BOOL=OFF',
                                         '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
                                         '-DBOOST_ROOT=$BOOST_ROOT',
                                         '-DGT_ENABLE_PYUTILS=OFF',
                                         '-DGT_TESTS_REQUIRE_FORTRAN_COMPILER=ON',
                                         '-DGT_TESTS_REQUIRE_C_COMPILER=ON',
                                         '-DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON']
#        self.build_system.config_opts = ['-DCMAKE_INSTALL_PREFIX=/scratch/snx3000/bignamic/reframe/cscs-checks/libraries/gridtools/install',
#                                         '-DBoost_NO_BOOST_CMAKE=ON',
#                                         '-DGT_ENABLE_TARGET_CUDA=OFF',
#                                         '-DGT_ENABLE_TARGET_MC=ON',
#                                         '-DCUDA_ARCH:STRING=NONE',
#                                         '-DCMAKE_BUILD_TYPE:STRING="Release"',
#                                         '-DBUILD_TESTING=OFF',
#                                         '-DBUILD_SHARED_LIBS=OFF',
#                                         '-DGT_USE_MPI=ON',
#                                         '-DBOOST_ROOT=/apps/daint/UES/jenkins/6.0.UP07/gpu/easybuild/software/Boost/1.67.0-CrayGNU-18.08-python3',
#                                         '-DBOOST_INCLUDEDIR=/apps/daint/UES/jenkins/6.0.UP07/gpu/easybuild/software/Boost/1.67.0-CrayGNU-18.08-python3/include/boost'
#                                         '-DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON']

        self.sanity_patterns = sn.assert_found(r'PASSED', self.stdout)
        self.perf_patterns = {
            'perf': sn.extractsingle(r'(?P<timer>\w+) ms total',
                                     self.stdout, 'timer', int)
        }
        self.build_system.max_concurrency = 2
        self.tags = {'production'}
        self.maintainers = ['CB']

@rfm.parameterized_test(['VerticalAdvectionNaive'], ['VerticalAdvectionMC'], 
                        ['SimpleHoriDiffNaive'], ['SimpleHoriDiffMC'])
class GridToolsCheckCPU(GridToolsCheckBase):
    def __init__(self, test_name):
        super().__init__()
        self.descr = 'GridTools host test'

        self.valid_systems = ['daint:mc', 'daint:gpu']
        self.num_gpus_per_node = 0
        self.num_tasks = 1

        if test_name == 'VerticalAdvectionNaive':
            self.build_system.make_opts = ['vertical_advection_dycore_naive']
            self.executable = 'regression/vertical_advection_dycore_naive'
            self.executable_opts = ['150', '150', '150']
            self.reference = {
                'daint:mc': {
                    'perf': (5000, None, 0.1)
                },
                'daint:gpu': {
                    'perf': (5000, None, 0.1)
                }
            }
        elif test_name == 'VerticalAdvectionMC':
            self.build_system.make_opts = ['vertical_advection_dycore_mc']
            self.executable = 'regression/vertical_advection_dycore_mc'
            self.executable_opts = ['150', '150', '150']
            self.reference = {
                'daint:mc': {
                    'perf': (5000, None, 0.1)
                },
                'daint:gpu': {
                    'perf': (5000, None, 0.1)
                }
        }
        elif test_name == 'SimpleHoriDiffNaive':
            self.build_system.make_opts = ['simple_hori_diff_naive']
            self.executable = 'regression/simple_hori_diff_naive'
            self.executable_opts = ['150', '150', '150']
            self.reference = {
                'daint:mc': {
                    'perf': (5000, None, 0.1)
                },
                'daint:gpu': {
                    'perf': (5000, None, 0.1)
                }
        }
        elif test_name == 'SimpleHoriDiffMC':
            self.build_system.make_opts = ['simple_hori_diff_mc']
            self.executable = 'regression/simple_hori_diff_mc'
            self.executable_opts = ['150', '150', '150']
            self.reference = {
                'daint:mc': {
                    'perf': (5000, None, 0.1)
                },
                'daint:gpu': {
                    'perf': (5000, None, 0.1)
                }
        }

@rfm.parameterized_test(['VerticalAdvectionCuda'], ['SimpleHoriDiffCuda'])
class GridToolsCheckGPU(GridToolsCheckBase):
    def __init__(self, test_name):
        super().__init__()
        self.descr = 'GridTools device test'

        self.modules = ['/scratch/snx3000/bignamic/easybuild_install/modules/all/CMake/3.12.4', 'Boost', 'cray-libsci_acc/18.07.1']
        self.build_system.config_opts = ['-DCMAKE_INSTALL_PREFIX=/scratch/snx3000/bignamic/reframe/cscs-checks/libraries/gridtools/install',
                                         '-DBoost_NO_BOOST_CMAKE="true"',
                                         '-DCMAKE_BUILD_TYPE:STRING=Release',
                                         '-DBUILD_SHARED_LIBS:BOOL=ON',
                                         '-DGT_ENABLE_TARGET_X86:BOOL=OFF',
                                         '-DGT_ENABLE_TARGET_NAIVE:BOOL=OFF',
                                         '-DGT_ENABLE_TARGET_CUDA:BOOL=ON',
                                         '-DCUDA_ARCH:STRING=sm_60',
                                         '-DGT_GCL_ONLY:BOOL=OFF',
                                         '-DCMAKE_CXX_COMPILER=CC',
                                         '-DCMAKE_CUDA_HOST_COMPILER:STRING=CC',
                                         '-DGT_USE_MPI:BOOL=OFF', 
                                         '-DGT_SINGLE_PRECISION:BOOL=OFF', 
                                         '-DGT_ENABLE_PERFORMANCE_METERS:BOOL=ON',
                                         '-DGT_TESTS_ICOSAHEDRAL_GRID:BOOL=OFF',
                                         '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
                                         '-DBOOST_ROOT=$BOOST_ROOT',
                                         '-DGT_ENABLE_PYUTILS=OFF',
                                         '-DGT_TESTS_REQUIRE_FORTRAN_COMPILER=ON',
                                         '-DGT_TESTS_REQUIRE_C_COMPILER=ON',
                                         '-DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON']

        self.valid_systems = ['daint:gpu']
        self.num_gpus_per_node = 1
        self.num_tasks = 1

        if test_name == 'VerticalAdvectionCuda':
            self.build_system.make_opts = ['vertical_advection_dycore_cuda']
            self.executable = 'regression/vertical_advection_dycore_cuda'
            self.executable_opts = ['200', '200', '200']
            self.reference = {
                'daint:gpu': {
                    'perf': (4500, None, 0.1)
                }
            }
        elif test_name == 'SimpleHoriDiffCuda':
            self.build_system.make_opts = ['simple_hori_diff_cuda']
            self.executable = 'regression/simple_hori_diff_cuda'
            self.executable_opts = ['200', '200', '200']
            self.reference = {
                'daint:mc': {
                    'perf': (12000, None, 0.1)
                },
                'daint:gpu': {
                    'perf': (12000, None, 0.1)
                }
        }
