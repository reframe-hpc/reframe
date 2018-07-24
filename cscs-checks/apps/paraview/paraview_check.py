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

        # Uncomment and adjust to load the ParaView module
        self.modules = ['ParaView']

        self.executable = 'pvbatch'
        self.executable_opts = ['coloredSphere.py']

        # NOTE: This is needed in order to swap from the default
        # version of gcc until issue #59 is fixed. Then it should
        # be moved to the `self.modules` definition.
        self.pre_run = ['module swap gcc/6.2.0 gcc/7.1.0']

        self.sanity_patterns = sn.assert_found(
            'Vendor:   NVIDIA Corporation', self.stdout)

        self.num_tasks = 12
        self.num_tasks_per_node = 12

        self.maintainers = ['JF']
        self.tags = {'scs', 'production'}
