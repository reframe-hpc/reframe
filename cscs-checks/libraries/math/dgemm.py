import os
import reframe.utility.sanity as sn

from reframe.core.pipeline import RegressionTest


class DGEMMTest(RegressionTest):
    def __init__(self, name, **kwargs):
        super().__init__('DGEMM_' + name, os.path.dirname(__file__), **kwargs)
        self.descr = 'DGEMM performance test'
        self.sourcepath = 'dgemm.c'
        self.executable_opts = ['5000', '5000', '5000']
        self.cflags = None
        self.ldflags = None
        self.sanity_patterns = sn.assert_found(
            'Time for \d+ DGEMM operations', self.stdout)
        self.maintainers = ['AJ']
        self.tags = {'production'}

    def compile(self):
        self.current_environ.cflags  = self.cflags
        self.current_environ.ldflags = self.ldflags
        super().compile()


class DGEMMTestMonch(DGEMMTest):
    def __init__(self, **kwargs):
        super().__init__('Monch', **kwargs)
        self.tags = {'monch_acceptance'}
        self.valid_systems = ['monch:compute']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.num_tasks_per_core = 1
        self.num_cpus_per_task = 20
        self.num_tasks_per_socket = 10
        self.use_multithreading = False
        self.cflags  = '-O3 -I$EBROOTOPENBLAS/include'
        self.ldflags = '-L$EBROOTOPENBLAS/lib -lopenblas -lpthread -lgfortran'
        self.variables = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'MV2_ENABLE_AFFINITY': '0'
        }
        self.perf_patterns = {
            'perf': sn.max(
                sn.extractall(r'Run\s\d\s+:\s+(?P<gflops>\S+)\s\S+',
                              self.stdout, "gflops", float)
            )
        }
        self.reference = {
            'monch:compute': {
                'perf': (350, -0.1, None)
            }
        }


def _get_checks(**kwargs):
    return [DGEMMTestMonch(**kwargs)]
