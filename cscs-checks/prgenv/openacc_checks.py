import os

import reframe.utility.sanity as sn
from reframe.core.pipeline import RegressionTest


class OpenACCFortranCheck(RegressionTest):
    def __init__(self, **kwargs):
        super().__init__('openacc_fortran_check',
                         os.path.dirname(__file__), **kwargs)
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'kesch:cn']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-pgi']
        if self.current_system.name in ['daint', 'dom']:
            self.modules = ['craype-accel-nvidia60']

        self.sourcepath = 'vecAdd_openacc.f90'
        self.num_gpus_per_node = 1
        self.executable = self.name

        result = sn.extractsingle(r'final result:\s+(?P<result>\d+\.?\d*)',
            self.stdout, 'result', float)
        self.sanity_patterns = sn.assert_reference(result, 1., -1e-5, 1e-5)

        self.maintainers = ['TM', 'VK']
        self.tags = {'production'}

    def setup(self, partition, environ, **job_opts):
        if environ.name == 'PrgEnv-cray':
            environ.fflags = '-hacc -hnoomp'
        elif environ.name == 'PrgEnv-pgi':
            environ.fflags = '-acc -ta=tesla:cc60'

        super().setup(partition, environ, **job_opts)


def _get_checks(**kwargs):
    return [OpenACCFortranCheck(**kwargs)]
