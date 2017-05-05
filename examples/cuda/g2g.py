import os

from reframe.core.pipeline import RegressionTest
from reframe.utility.functions import standard_threshold

class G2GBaseTest(RegressionTest):
    def __init__(self, name, **kwargs):
        super().__init__('g2g_osu_microbenchmark_p2p_%s' % name,
                         os.path.dirname(__file__), **kwargs)

        # Uncomment and adjust for your site
        # self.valid_systems = [ 'sys1', 'sys2' ]

        # Uncomment and adjust to load CUDA
        # self.modules = [ 'cudatoolkit' ]

        self.num_tasks = 2
        self.num_tasks_per_node = 1
        self.num_gpus_per_node = 1

        self.executable = 'g2g_osu_bw'
        self.descr = 'G2G microbenchmark P2P ' + name.upper()
        self.variables = {'MPICH_RDMA_ENABLED_CUDA': '1'}

        # Uncomment and set the valid prog. environments for your site
        self.valid_prog_environs = [ 'PrgEnv-cray', 'PrgEnv-gnu',
                                     'PrgEnv-intel' ]

        self.maintainers = [ 'VH', 'VK' ]
        self.tags = { 'production' }

        self.sanity_patterns = {
            '-': {'^4194304': []}
        }

        self.reference = {
            # Uncomment and adjust the references for your systems/partitions
            # 'sys1' : {
            #     'perf' : (9140.10, -0.1, None)
            # },
            # 'sys2': {
            #     'perf': (9140.10, -0.1, None)
            # },
        }

        self.perf_patterns = {
            '-': {
                '^4194304\s+(?P<perf>\S+)': [
                    ('perf', float, standard_threshold)
                ]
            }
        }

    def compile(self):
        self.current_environ.cflags += ' -D_ENABLE_CUDA_ -DCUDA_ENABLED '
        super().compile(makefile='Makefile_g2g')


class G2GCPUTest(G2GBaseTest):
    def __init__(self, **kwargs):
        super().__init__('cpu', **kwargs)


class G2GCUDATest(G2GBaseTest):
    def __init__(self, **kwargs):
        super().__init__('gpu', **kwargs)
        self.executable_opts = '-d cuda D D'.split()
        self.num_gpus_per_node = 1


def _get_checks(**kwargs):
    return [ G2GCPUTest(**kwargs), G2GCUDATest(**kwargs) ]
