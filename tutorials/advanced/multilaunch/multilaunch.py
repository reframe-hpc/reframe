import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class MultiLaunchTest(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.valid_systems = ['daint:gpu', 'daint:mc']
        self.valid_prog_environs = ['cray']
        self.executable = 'hostname'
        self.num_tasks = 4
        self.num_tasks_per_node = 1
        self.sanity_patterns = sn.assert_eq(
            sn.count(sn.extractall(r'^nid\d+', self.stdout)), 10
        )

    @rfm.run_before('run')
    def pre_launch(self):
        cmd = self.job.launcher.run_command(self.job)
        self.prerun_cmds = [
            f'{cmd} -n {n} {self.executable}'
            for n in range(1, self.num_tasks)
        ]
