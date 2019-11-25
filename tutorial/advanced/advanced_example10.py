import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.20-dev2')
@rfm.simple_test
class Example10Test(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.descr = 'Run commands inside a container'
        self.valid_systems = ['daint:gpu']
        self.valid_prog_environs = ['PrgEnv-cray']
        self.container_platform = 'Singularity'
        self.container_platform.image = 'docker://ubuntu:18.04'
        self.container_platform.commands = [
            'pwd', 'ls', 'cat /etc/os-release'
        ]
        self.container_platform.workdir = '/workdir'
        self.sanity_patterns = sn.all([
            sn.assert_found(r'^' + self.container_platform.workdir,
                            self.stdout),
            sn.assert_found(r'^advanced_example1.c', self.stdout),
            sn.assert_found(r'18.04.\d+ LTS \(Bionic Beaver\)', self.stdout),
        ])
        self.maintainers = ['put-your-name-here']
        self.tags = {'tutorial'}
