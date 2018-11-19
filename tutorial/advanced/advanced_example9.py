import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class HostnameCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['daint:gpu', 'daint:mc']
        self.valid_prog_environs = ['PrgEnv-cray']
        self.executable = 'hostname'
        self.sourcesdir = None
        self.num_tasks = 0
        self.num_tasks_per_node = 1
        self.sanity_patterns = sn.assert_eq(
            self.num_tasks_assigned,
            sn.count(sn.findall(r'nid\d+', self.stdout))
        )
        self.maintainers = ['you-can-type-your-email-here']
        self.tags = {'tutorial'}

    @property
    @sn.sanity_function
    def num_tasks_assigned(self):
        return self.job.num_tasks
