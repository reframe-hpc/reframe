import os
import itertools
import reframe.utility.sanity as sn

from reframe.core.deferrable import deferrable
from reframe.core.pipeline import RegressionTest


class MPIHelloWorldTest(RegressionTest):
    def __init__(self, **kwargs):
        super().__init__('mpi_helloworld_check',
                         os.path.dirname(__file__), **kwargs)
        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc',
                              'kech:cn', 'kech:pn',
                              'leone:normal', 'monch:compute']
        self.descr = 'MPI Hello World'
        self.sourcepath = 'mpi_helloworld.c'
        self.executable = './mpi_helloworld'
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi']
        self.maintainers = ['RS', 'VK']
        self.num_tasks_per_node = 1
        self.num_tasks = 0
        num_processes = sn.extractsingle(
            r'Received messages from (?P<nprocs>\d+) processes',
            self.stdout, 'nprocs', int)
        self.sanity_patterns = sn.assert_eq(num_processes, self.real_num_tasks)

    @property
    @deferrable
    def real_num_tasks(self):
        return self.job.num_tasks


def _get_checks(**kwargs):
    return [MPIHelloWorldTest(**kwargs)]
