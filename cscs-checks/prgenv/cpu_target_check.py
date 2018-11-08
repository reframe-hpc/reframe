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
        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.executable = 'echo ${CRAY_CPU_TARGET:-Unknown}'
        self.sanity_patterns = sn.assert_not_found('Unknown', self.stdout)

        self.maintainers = ['TM']
        self.tags = {'maintenance'}
