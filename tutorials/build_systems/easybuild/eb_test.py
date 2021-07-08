import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class BZip2EBCheck(rfm.RegressionTest):
    descr = 'Demo test using EasyBuild to build the test code'
    valid_systems = ['*']
    valid_prog_environs = ['builtin']
    executable = 'bzip2'
    executable_opts = ['--help']

    @run_before('compile')
    def setup_build_system(self):
        self.build_system = 'EasyBuild'
        self.build_system.easyconfigs = ['bzip2-1.0.6.eb']
        self.build_system.options = ['-f']

    @run_before('run')
    def prepare_run(self):
        self.modules = self.build_system.generated_modules

    @sanity_function
    def assert_version(self):
        return sn.assert_found(r'Version 1.0.6', self.stderr)
