import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class ContainerCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.descr = 'Run commands inside a container'
        self.valid_systems = ['dom:gpu', 'daint:gpu']
        self.valid_prog_environs = ['*']
        self.sanity_patterns = sn.all([
            #sn.assert_found(r'^' + self.container_platform.workdir,
            #                self.stdout),
            sn.assert_found(r'^container_test.txt', self.stdout),
            sn.assert_found(r'18.04.3 LTS \(Bionic Beaver\)', self.stdout),
        ])
