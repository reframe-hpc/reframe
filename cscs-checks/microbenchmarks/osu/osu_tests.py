import os
import itertools
import reframe.utility.sanity as sn

from reframe.core.pipeline import RegressionTest


class AlltoallBaseTest(RegressionTest):
    def __init__(self, name, **kwargs):
        super().__init__(name,
                         os.path.dirname(__file__), **kwargs)
        self.strict_check = False
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.descr = 'Alltoall osu microbenchmark'
        self.executable = './osu_alltoall'
        # The -x option controls the number of warm-up iterations
        # The -i option controls the number of iterations
        self.executable_opts = ['-m', '8', '-x', '1000', '-i', '20000']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel']
        self.maintainers = ['RS', 'VK']
        self.sanity_patterns = sn.assert_found(r'^8', self.stdout)
        self.perf_patterns = {
            'perf': sn.extractsingle(r'^8\s+(?P<perf>\S+)',
                                     self.stdout, 'perf', float)
        }
        self.num_tasks_per_node = 1
        self.num_gpus_per_node  = 1
        if self.current_system.name == 'dom':
            self.num_tasks = 6

        if self.current_system.name == 'daint':
            self.num_tasks = 16

        self.extra_resources = {
            'switches': {
                'num_switches': 1
            }
        }

    def compile(self):
        super().compile(makefile='Makefile_alltoall')


class AlltoallProdTest(AlltoallBaseTest):
    def __init__(self, **kwargs):
        super().__init__('alltoall_osu_microbenchmark', **kwargs)
        self.tags = {'production'}
        self.reference = {
            'dom:gpu': {
                'perf': (8.23, None, 0.1)
            },
            'daint:gpu': {
                'perf': (20.73, None, 2.0)
            },
        }


class AlltoallMonchAcceptanceTest(AlltoallBaseTest):
    def __init__(self, num_tasks, **kwargs):
        super().__init__('alltoall_osu_microbenchmark_monch_%s'
                         % num_tasks, **kwargs)
        self.valid_systems = ['monch:compute']
        self.num_tasks = num_tasks
        reference_by_node = {
            2: {
                'perf': (2.71, None, 0.1)
            },
            4: {
                'perf': (3.75, None, 0.1)
            },
            6: {
                'perf': (6.28, None, 0.1)
            },
            8: {
                'perf': (8.15, None, 0.1)
            },
        }
        self.reference = {
            'monch:compute': reference_by_node[self.num_tasks]
        }
        self.tags = {'monch_acceptance'}


class G2GBaseTest(RegressionTest):
    def __init__(self, name, **kwargs):
        super().__init__('g2g_osu_microbenchmark_p2p_%s' % name,
                         os.path.dirname(__file__), **kwargs)
        self.exclusive_access = True
        self.strict_check = False
        self.num_tasks = 2
        self.num_tasks_per_node = 1
        self.descr = 'G2G microbenchmark P2P ' + name.upper()
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel']
        self.maintainers = ['RS', 'VK']
        self.tags = {'production'}
        self.sanity_patterns = sn.assert_found(r'^4194304', self.stdout)

        self.extra_resources = {
            'switches': {
                'num_switches': 1
            }
        }

    def setup(self, partition, environ, **job_opts):
        if partition.name == 'gpu':
            self.num_gpus_per_node  = 1
            self.modules = ['cudatoolkit']
            self.variables = {'MPICH_RDMA_ENABLED_CUDA': '1'}

        if partition.name == 'cn':
            self.variables = {'MV2_USE_CUDA': '1'}

        super().setup(partition, environ, **job_opts)
        if self.num_gpus_per_node >= 1:
            self.current_environ.cflags = ' -D_ENABLE_CUDA_ -DCUDA_ENABLED '

    def compile(self):
        super().compile(makefile='Makefile_g2g')


class G2GCPUBandwidthTest(G2GBaseTest):
    def __init__(self, **kwargs):
        super().__init__('cpu_bandwidth', **kwargs)
        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc',
                              'monch:compute',
                              'kesch:cn']
        if self.current_system.name == 'kesch':
            self.valid_prog_environs = ['PrgEnv-cray']

        self.executable = './g2g_osu_bw'
        self.executable_opts = ['-x', '100', '-i', '1000']

        self.reference = {
            'daint:gpu': {
                'bw': (9798.29, -0.1, None)
            },
            'daint:mc': {
                'bw': (9865.00, -0.2, None)
            },
            'dom:gpu': {
                'bw': (9815.66, -0.1, None)
            },
            'dom:mc': {
                'bw': (9472.59, -0.20, None)
            },
            'monch:compute': {
                'bw': (6317.84, -0.15, None)
            },
            'kesch:cn': {
                'bw': (6311.48, -0.15, None)
            }
        }
        self.perf_patterns = {
            'bw': sn.extractsingle(r'^4194304\s+(?P<bw>\S+)',
                                   self.stdout, 'bw', float)
        }
        self.tags |= {'monch_acceptance'}


class G2GCPULatencyTest(G2GBaseTest):
    def __init__(self, **kwargs):
        super().__init__('cpu_latency', **kwargs)
        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc',
                              'monch:compute',
                              'kesch:cn']
        if self.current_system.name == 'kesch':
            self.valid_prog_environs = ['PrgEnv-cray']
        self.executable_opts = ['-x', '100', '-i', '1000']

        self.executable = './g2g_osu_latency'
        self.reference = {
            'daint:gpu': {
                'latency': (1.16, None, 1.0)
            },
            'daint:mc': {
                'latency': (1.15, None, 0.6)
            },
            'dom:gpu': {
                'latency': (1.13, None, 0.1)
            },
            'dom:mc': {
                'latency': (1.27, None, 0.2)
            },
            'monch:compute': {
                'latency': (1.27, None, 0.1)
            },
            'kesch:cn': {
                'latency': (1.17, None, 0.1)
            }
        }
        self.perf_patterns = {
            'latency': sn.extractsingle(r'^8\s+(?P<latency>\S+)',
                                        self.stdout, 'latency', float)
        }
        self.tags |= {'monch_acceptance'}


class G2GCUDABandwidthTest(G2GBaseTest):
    def __init__(self, **kwargs):
        super().__init__('gpu_bandwidth', **kwargs)
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'kesch:cn']
        if self.current_system.name == 'kesch':
            self.valid_prog_environs = ['PrgEnv-cray']

        self.num_gpus_per_node = 1
        self.executable = './g2g_osu_bw'
        self.executable_opts = ['-x', '100', '-i', '1000', '-d',
                                'cuda', 'D', 'D']

        self.reference = {
            'dom:gpu': {
                'bw': (8897.86, -0.1, None)
            },
            'daint:gpu': {
                'bw': (8765.65, -0.1, None)
            },
            'kesch:cn': {
                'bw': (6288.98, -0.1, None)
            },
        }
        self.perf_patterns = {
            'bw': sn.extractsingle(r'^4194304\s+(?P<bw>\S+)',
                                   self.stdout, 'bw', float)
        }


class G2GCUDALatencyTest(G2GBaseTest):
    def __init__(self, **kwargs):
        super().__init__('gpu_latency', **kwargs)
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'kesch:cn']
        if self.current_system.name == 'kesch':
            self.valid_prog_environs = ['PrgEnv-cray']

        self.num_gpus_per_node = 1
        self.executable = './g2g_osu_latency'
        self.executable_opts = ['-x', '100', '-i', '1000', '-d',
                                'cuda', 'D', 'D']

        self.reference = {
            'dom:gpu': {
                'latency': (5.49, None, 0.1)
            },
            'daint:gpu': {
                'latency': (5.73, None, 1.0)
            },
            'kesch:cn': {
                'latency': (23.09, None, 0.1)
            },
        }
        self.perf_patterns = {
            'latency': sn.extractsingle(r'^8\s+(?P<latency>\S+)',
                                        self.stdout, 'latency', float)
        }


def _get_checks(**kwargs):
    fixed_tests = [AlltoallProdTest(**kwargs),
                   G2GCPUBandwidthTest(**kwargs),
                   G2GCPULatencyTest(**kwargs),
                   G2GCUDABandwidthTest(**kwargs),
                   G2GCUDALatencyTest(**kwargs)]
    return list(itertools.chain(fixed_tests,
                                map(lambda n: AlltoallMonchAcceptanceTest(
                                    n, **kwargs), range(2, 10, 2))))
