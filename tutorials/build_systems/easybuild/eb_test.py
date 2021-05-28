import reframe as rfm
import reframe.utility.sanity as sn


class EasybuildMixin(rfm.RegressionTest):
    @rfm.run_before('run')
    def prepare_run(self):
        self.modules = self.build_system.generated_modules


@rfm.simple_test
class BZip2Check(EasybuildMixin):
    def __init__(self):
        self.descr = 'This demonstrates the EasyBuild build system.'
        self.valid_systems = ['daint:gpu']
        self.valid_prog_environs = ['gnu']
        self.modules = ['daint-gpu',
                        'EasyBuild-custom']
        self.build_system = 'EasyBuild'
        self.build_system.easyconfigs = ['bzip2-1.0.6.eb']
        self.build_system.options = ['-f']
        self.sanity_patterns = sn.assert_found(r'Version 1.0.6',
                                               self.stderr)
        self.executable = 'bzip2'
        self.executable_opts = ['--help']
