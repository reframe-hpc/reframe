import reframe as rfm
import reframe.utility.sanity as sn


class DGEMMTest(rfm.RegressionTest):
    def __init__(self):
        super().__init__()
        self.descr = 'DGEMM performance test'
        self.sourcepath = 'dgemm.c'
        self.executable_opts = ['5000', '5000', '5000']
        self.sanity_patterns = sn.assert_found(
            r'Time for \d+ DGEMM operations', self.stdout)
        self.maintainers = ['AJ']
        self.tags = {'production'}


# FIXME: This test is obsolete; it is kept only for reference.
@rfm.required_version('>=2.14')
@rfm.simple_test
class DGEMMTestMonch(DGEMMTest):
    def __init__(self):
        super().__init__()
        self.tags = {'monch_acceptance'}
        self.valid_systems = ['monch:compute']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.num_tasks_per_core = 1
        self.num_cpus_per_task = 20
        self.num_tasks_per_socket = 10
        self.use_multithreading = False
        self.variables = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'MV2_ENABLE_AFFINITY': '0'
        }
        self.build_system = 'SingleSource'
        self.build_system.cflags = ['-O3', '-I$EBROOTOPENBLAS/include']
        self.build_system.ldflags = ['-L$EBROOTOPENBLAS/lib', '-lopenblas',
                                     '-lpthread', '-lgfortran']
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
