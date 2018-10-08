import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class UlimitCheck(rfm.RegressionTest):
    def __init__(self):
        super().__init__()
        self.descr = 'Checking the output of ulimit -s in node.'
        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu',   'dom:mc']
        self.valid_prog_environs = ['PrgEnv-cray',  'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi']
        self.sourcepath = 'ulimit.c'
        self.sanity_patterns = sn.all([
            sn.assert_found(r'The soft limit is unlimited', self.stdout),
            sn.assert_found(r'The hard limit is unlimited', self.stdout),
        ])

        self.maintainers = ['RS', 'VK']
        self.tags = {'production', 'scs'}
