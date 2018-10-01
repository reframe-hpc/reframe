import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.14')
@rfm.simple_test
class NvmlCheck(rfm.RegressionTest):
    def __init__(self):
        super().__init__()
        self.descr = 'check GPU compute mode'
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'CUDA', 'nvml')
        self.modules = ['craype-accel-nvidia60']
        self.sourcepath = 'nvml.c'
        self.build_system = 'SingleSource'
        self.build_system.ldflags = ['-lnvidia-ml']
        self.sanity_patterns = sn.assert_found(
            r"compute\s+mode\s+'Exclusive Process'", self.stdout)

        self.maintainers = ['AJ', 'VK']
        self.tags = {'production'}
