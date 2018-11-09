import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class RubyNArray(rfm.RunOnlyRegressionTest):
    def __init__(self):
        super().__init__()
        self.descr = ('Check NArray for Ruby version 2.2.2')
        self.valid_systems = ['kesch:cn']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-pgi',
                                    'PrgEnv-gnu-nompi']
        self.modules = ['ruby/2.2.2-gmvolf-17.02']
        self.executable = 'ruby NArray.rb'
        self.executable_opts = [self.sourcepath]
        self.sanity_patterns = sn.assert_found(r'NArray\.float\(4\):\s*'
                                               '\[ 1.0, 2.0, 3.0, 4.0 \]',
                                               self.stdout)
        self.maintainers = ['MKr']
        self.tags = {'production'}
