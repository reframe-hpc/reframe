import os

import reframe.utility.sanity as sn
from reframe.core.pipeline import RegressionTest


class MPITest(RegressionTest):
    def __init__(self, **kwargs):
        super().__init__('example3_check',
                         os.path.dirname(__file__), **kwargs)
        self.descr = 'Matrix-vector multiplication example with MPI'
        self.valid_systems = ['daint:gpu', 'daint:mc']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi']
        self.sourcepath = 'example_matrix_vector_multiplication_mpi_openmp.c'
        self.executable_opts = ['1024', '10']
        self.prgenv_flags = {
            'PrgEnv-cray':  '-homp',
            'PrgEnv-gnu':   '-fopenmp',
            'PrgEnv-intel': '-openmp',
            'PrgEnv-pgi':   '-mp'
        }
        self.sanity_patterns = sn.assert_found(
            r'time for single matrix vector multiplication', self.stdout)
        self.num_tasks = 8
        self.num_tasks_per_node = 2
        self.num_cpus_per_task = 4
        self.variables = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task)
        }
        self.maintainers = ['you-can-type-your-email-here']
        self.tags = {'tutorial'}

    def compile(self):
        prgenv_flags = self.prgenv_flags[self.current_environ.name]
        self.current_environ.cflags = prgenv_flags
        super().compile()


def _get_checks(**kwargs):
    return [MPITest(**kwargs)]
