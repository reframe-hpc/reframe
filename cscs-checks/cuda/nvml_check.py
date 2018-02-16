import os
import reframe.utility.sanity as sn

from reframe.core.pipeline import RegressionTest


class NvmlCheck(RegressionTest):
    def __init__(self, **kwargs):
        super().__init__('nvml_check', os.path.dirname(__file__), **kwargs)
        self.descr = 'check GPU compute mode'
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'CUDA', 'nvml')
        self.sourcepath = 'nvml.c'
        self.sanity_patterns = sn.assert_found(
            r"compute\s+mode\s+'Exclusive Process'", self.stdout)
        self.modules = ['cudatoolkit']

        self.maintainers = ['AJ', 'VK']
        self.tags = {'production'}

    def compile(self):
        # Set cc flags
        self.current_environ.cflags = '-lnvidia-ml'
        super().compile()


def _get_checks(**kwargs):
    return [NvmlCheck(**kwargs)]
