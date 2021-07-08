import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class BZip2SpackCheck(rfm.RegressionTest):
    descr = 'Demo test using Spack to build the test code'
    valid_systems = ['daint:gpu']
    valid_prog_environs = ['builtin']
    executable = 'bzip2'
    executable_opts = ['--help']

    @run_before('compile')
    def setup_build_system(self):
        self.build_system = 'Spack'
        self.build_system.environment = '.'
        self.build_system.specs = ['bzip2@1.0.6']

    @sanity_function
    def assert_version(self):
        return sn.assert_found(r'Version 1.0.6', self.stderr)
