import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class ParaViewCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        super().__init__()
        self.name = 'paraview_gpu_check'
        self.descr = 'ParaView GPU check'
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.num_tasks = 12
        self.num_tasks_per_node = 12
        self.modules = ['gcc/7.1.0', 'ParaView']

        self.executable = 'pvbatch'
        self.executable_opts = ['coloredSphere.py']

        self.sanity_patterns = sn.assert_found(
            'Vendor:   NVIDIA Corporation', self.stdout)

        self.maintainers = ['JF']
        self.tags = {'scs', 'production'}
