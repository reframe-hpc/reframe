import os

import reframe.utility.sanity as sn
from reframe.core.pipeline import RegressionTest

class OpenACCFortranCheck(RegressionTest):
    def __init__(self, num_tasks, **kwargs):
        if num_tasks == 1:
            check_name = 'openacc_fortran_check'
        else:
            check_name = 'openacc_mpi_fortran_check'
        super().__init__(check_name,
                         os.path.dirname(__file__), **kwargs)
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'kesch:cn']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-pgi']
        if self.current_system.name in ['daint', 'dom']:
            self.modules = ['craype-accel-nvidia60']
            self._pgi_flags = '-acc -ta=tesla:cc60'
        elif self.current_system.name in ['kesch']:
            self.modules = ['craype-accel-nvidia35']
            self._pgi_flags = '-acc -ta=tesla:cc35'

        self.num_tasks = num_tasks
        if self.num_tasks == 1:
            self.sourcepath = 'vecAdd_openacc.f90'
        else:
            self.sourcepath = 'vecAdd_openacc_mpi.f90'
        self.num_gpus_per_node = 1
        self.executable = self.name
        self.num_tasks_per_node = 1

        result = sn.extractsingle(r'final result:\s+(?P<result>\d+\.?\d*)',
            self.stdout, 'result', float)
        self.sanity_patterns = sn.assert_reference(result, 1., -1e-5, 1e-5)

        self.maintainers = ['TM', 'VK']
        self.tags = {'production'}

    def setup(self, partition, environ, **job_opts):
        if environ.name == 'PrgEnv-cray':
            environ.fflags = '-hacc -hnoomp'
        elif environ.name == 'PrgEnv-pgi':
            environ.fflags = self._pgi_flags

        super().setup(partition, environ, **job_opts)


def _get_checks(**kwargs):
    return [OpenACCFortranCheck(1, **kwargs),
            OpenACCFortranCheck(2, **kwargs)]
