import reframe as rfm
import reframe.utility.sanity as sn

@rfm.simple_test
class ParaViewCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.num_tasks = 12
        self.num_tasks_per_node = 12
        self.modules = ['ParaView']

        self.time_limit = (0, 1, 0)
        self.executable = 'pvbatch'
        self.executable_opts = ['newcoloredSphere.py']

        self.maintainers = ['JF']
        self.tags = {'scs', 'production'}

    def setup(self, partition, environ, **job_opts):
        if partition.fullname == 'daint:mc':
            self.sanity_patterns = sn.assert_found('Vendor:   VMware, Inc.',
                                                   self.stdout)
            self.sanity_patterns = sn.assert_found('Renderer: llvmpipe',
                                                   self.stdout)
        elif partition.fullname == 'daint:gpu':
            self.sanity_patterns = sn.assert_found(
                'Vendor:   NVIDIA Corporation', self.stdout)
            self.sanity_patterns = sn.assert_found('Renderer: Tesla P100',
                                                   self.stdout)
        elif partition.fullname == 'dom:gpu':
            self.sanity_patterns = sn.assert_found(
                'Vendor:   NVIDIA Corporation', self.stdout)
            self.sanity_patterns = sn.assert_found('Renderer: Tesla P100',
                                                   self.stdout)

        super().setup(partition, environ, **job_opts)
