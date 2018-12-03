import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class RubyNArray(rfm.RunOnlyRegressionTest):
    def __init__(self):
        super().__init__()
        self.descr = ('Check NArray for Ruby version 2.2.2')
        self.valid_systems = ['kesch:cn']
        self.valid_prog_environs = ['PrgEnv-gnu-nompi']
        self.exclusive_access = True
        self.modules = ['ruby/2.2.2-gmvolf-17.02']
        self.executable = 'ruby'
        self.executable_opts = ['NArray.rb']
        self.sanity_patterns = sn.assert_found(r'NArray\.float\(4\):\s*'
                                               r'\[ 1.0, 2.0, 3.0, 4.0 \]',
                                               self.stdout)
        self.maintainers = ['MKr']
        self.tags = {'production', 'mch'}
