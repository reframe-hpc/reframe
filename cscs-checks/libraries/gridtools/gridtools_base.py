import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class BaseGridToolsCheck(rfm.RegressionTest):
    def __init__(self):
        super().__init__()
        self.descr = 'GridTools base test'
#        self.strict_check = False # TODO: check this
        self.valid_systems = ['kesch:cn', 'dom:mc']

        self.valid_prog_environs = ['PrgEnv-gnu']
        self.modules = ['CMake', 'Boost']
        self.executable = 'examples/copy_stencil_host_naive'
        self.executable_opts = ['10', '10', '10']
        self.sourcesdir = 'git@github.com:eth-cscs/gridtools'
        self.prebuild_cmd = ['git checkout tags/1.07.00'] # More recent versions require CMake >= 3.12
        self.build_system = 'CMake'
#        self.build_system.builddir = 'build/examples/'
        self.build_system.config_opts = ['-DCMAKE_INSTALL_PREFIX=/scratch/snx1600tds/bignamic/reframe/cscs-checks/libraries/gridtools/install',
                                         '-DBoost_NO_BOOST_CMAKE=ON',
                                         '-DGT_ENABLE_TARGET_CUDA=OFF',
                                         '-DGT_ENABLE_TARGET_MC=ON',
                                         '-DCUDA_ARCH:STRING=NONE',
                                         '-DCMAKE_BUILD_TYPE:STRING="Release"',
                                         '-DBUILD_TESTING=OFF',
                                         '-DBUILD_SHARED_LIBS=OFF',
                                         '-DGT_USE_MPI=ON',
                                         '-DBOOST_ROOT=/apps/dom/UES/jenkins/6.0.UP07/mc/easybuild/software/Boost/1.67.0-CrayGNU-18.08-python3',
                                         '-DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON']

        self.sanity_patterns = sn.assert_found(r'[  PASSED  ] 1 test.', self.stdout)
        self.tags = {'production'}
        self.maintainers = ['CB']
