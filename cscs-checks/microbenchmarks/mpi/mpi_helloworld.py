import os
import reframe.utility.sanity as sn

from reframe.core.pipeline import RegressionTest


class MPIHelloWorldTest(RegressionTest):
    def __init__(self, **kwargs):
        super().__init__('mpi_helloworld_check',
                         os.path.dirname(__file__), **kwargs)
        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc',
                              'kesch:cn', 'kesch:pn',
                              'leone:normal', 'monch:compute']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi']
        if self.current_system.name == 'kesch':
            self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                        'PrgEnv-intel']

        self.descr = 'MPI Hello World'
        self.sourcepath = 'mpi_helloworld.c'
        self.maintainers = ['RS', 'VK']
        self.num_tasks_per_node = 1
        self.num_tasks = 0
        num_processes = sn.extractsingle(
            r'Received correct messages from (?P<nprocs>\d+) processes',
            self.stdout, 'nprocs', int)
        self.sanity_patterns = sn.assert_eq(num_processes,
                                            self.num_tasks_assigned-1)

    @property
    @sn.sanity_function
    def num_tasks_assigned(self):
        return self.job.num_tasks


def _get_checks(**kwargs):
    return [MPIHelloWorldTest(**kwargs)]
