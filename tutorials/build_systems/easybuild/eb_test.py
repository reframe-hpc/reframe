import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class BZip2Check(rfm.RegressionTest):
    descr = 'This demonstrates the EasyBuild build system.'
    valid_systems = ['daint:gpu']
    valid_prog_environs = ['gnu']
    executable = 'bzip2'
    executable_opts = ['--help']

    @run_before('compile')
    def set_makefile(self):
        self.build_system = 'EasyBuild'
        self.build_system.easyconfigs = ['bzip2-1.0.6.eb']
        self.build_system.options = ['-f']

    @run_before('run')
    def prepare_run(self):
        self.modules = self.build_system.generated_modules

    @run_before('sanity')
    def set_sanity(self):
        self.sanity_patterns = sn.assert_found(r'Version 1.0.6',
                                               self.stderr)
