import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class JupyterHubSubmitTest(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.valid_systems = ['daint:jupyter_gpu', 'daint:jupyter_mc',
                              'dom:jupyter_gpu', 'dom:jupyter_mc']
        self.valid_prog_environs = ['*']
        self.sourcesdir = None
        self.executable = 'hostname'
        self.time_limit = (0, 1, 0)
        self.sanity_patterns = sn.assert_found(r'nid\d+', self.stdout)
        self.tags = {'production', 'maintenance'}
