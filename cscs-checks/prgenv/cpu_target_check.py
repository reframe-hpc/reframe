import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class CrayCPUTargetTest(rfm.RunOnlyRegressionTest):
    def __init__(self):
        super().__init__()
        self.descr = 'Checks whether CRAY_CPU_TARGET is set'
        self.valid_systems = ['daint:login', 'dom:login']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi']
        self.executable = 'echo CRAY_CPU_TARGET=$CRAY_CPU_TARGET'
        self.sanity_patterns = sn.assert_found(r'CRAY_CPU_TARGET=\S+',
                                               self.stdout)

        self.maintainers = ['TM']
        self.tags = {'production', 'maintenance'}
