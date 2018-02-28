import os

import reframe.utility.sanity as sn
from reframe.core.pipeline import RunOnlyRegressionTest


class ParaViewCheck(RunOnlyRegressionTest):
    def __init__(self, **kwargs):
        super().__init__('paraview_gpu_check',
                         os.path.dirname(__file__), **kwargs)
        self.descr = 'ParaView GPU check'

        # Uncomment and adjust for your site
        self.valid_systems = ['daint:gpu', 'dom:gpu']

        # Uncomment and set the valid prog. environments for your site
        self.valid_prog_environs = ['PrgEnv-gnu']

        # Uncomment and adjust to load the ParaView module
        self.modules = ['ParaView']

        # Reset sources dir relative to the SCS apps prefix
        self.executable = 'pvbatch'
        self.executable_opts = ['coloredSphere.py']

        self.sanity_patterns = sn.assert_found(
            'Vendor:   NVIDIA Corporation', self.stdout)

        # Uncomment and adjust for your site
        # self.use_multithreading = True
        self.num_tasks = 12
        self.num_tasks_per_node = 12

        # Uncomment and set the maintainers and/or tags
        self.maintainers = ['JF']
        self.tags = {'scs', 'production'}


def _get_checks(**kwargs):
    return [ParaViewCheck(**kwargs)]
