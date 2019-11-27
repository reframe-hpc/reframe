import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class OpenCLCheck(rfm.RegressionTest):
    def __init__(self):
        super().__init__()
        self.maintainers = ['TM', 'VK']
        self.tags = {'production', 'craype'}

        self.valid_systems = ['daint:gpu', 'dom:gpu', 'tiger:gpu']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-pgi']
        self.modules = ['craype-accel-nvidia60']
        self.build_system = 'Make'
        self.num_gpus_per_node = 1
        self.executable = 'vecAdd_opencl'

        self.sanity_patterns = sn.assert_found('SUCCESS', self.stdout)

    def setup(self, system, environ, **job_opts):
        super().setup(system, environ, **job_opts)
        if environ.name == 'PrgEnv-pgi':
            self.build_system.cflags = ['-mmmx']
