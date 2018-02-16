import os

from reframe.core.pipeline import RegressionTest
import reframe.utility.sanity as sn


class OpenCLCheck(RegressionTest):
    def __init__(self, **kwargs):
        super().__init__('opencl_check',
                         os.path.dirname(__file__), **kwargs)
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-pgi']

        self.modules = ['cudatoolkit']

        self.num_gpus_per_node = 1
        self.executable = 'vecAdd_opencl'

        self.sanity_patterns = sn.assert_found('SUCCESS', self.stdout)

        self.maintainers = ['TM', 'VK']
        self.tags = {'production'}


def _get_checks(**kwargs):
    return [OpenCLCheck(**kwargs)]
