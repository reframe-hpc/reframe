import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class SSHLoginEnvCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        super().__init__()
        self.descr = 'Test the environment when accessed remotely over SSH'
        self.valid_systems = ['daint:login', 'dom:login']
        self.valid_prog_environs = ['PrgEnv-cray']
        self.executable = "ssh"
        self.executable_opts = ["%s" % self.current_system.name,
                                "echo", "'$SCRATCH'"]
        self.sanity_patterns = sn.assert_found(
            r'/scratch/[\S+]', self.stdout)
        self.maintainers = ['RS', 'VK']
        self.tags = {'maintenance'}
