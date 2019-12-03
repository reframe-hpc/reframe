import os

import reframe as rfm
import reframe.utility.sanity as sn


class OSUBenchmarkTestBase(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.valid_systems = ['daint:gpu']
        self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-pgi', 'PrgEnv-intel']
        self.sourcesdir = None
        self.num_tasks = 2
        self.num_tasks_per_node = 1
        self.sanity_patterns = sn.assert_found(r'^8', self.stdout)


@rfm.simple_test
class OSULatencyTest(OSUBenchmarkTestBase):
    def __init__(self):
        super().__init__()
        self.descr = 'OSU latency test'
        self.perf_patterns = {
            'latency': sn.extractsingle(r'^8\s+(\S+)', self.stdout, 1, float)
        }
        self.depends_on('OSUBuildTest')
        self.reference = {
            '*': {'latency': (0, None, None, 'us')}
        }

    @rfm.require_deps
    def set_executable(self, OSUBuildTest):
        self.executable = os.path.join(
            OSUBuildTest().stagedir,
            'osu-micro-benchmarks-5.6.2', 'mpi', 'pt2pt', 'osu_latency'
        )
        self.executable_opts = ['-x', '100', '-i', '1000']


@rfm.simple_test
class OSUBandwidthTest(OSUBenchmarkTestBase):
    def __init__(self):
        super().__init__()
        self.descr = 'OSU bandwidth test'
        self.perf_patterns = {
            'bandwidth': sn.extractsingle(r'^4194304\s+(\S+)',
                                          self.stdout, 1, float)
        }
        self.depends_on('OSUBuildTest')
        self.reference = {
            '*': {'bandwidth': (0, None, None, 'MB/s')}
        }

    @rfm.require_deps
    def set_executable(self, OSUBuildTest):
        self.executable = os.path.join(
            OSUBuildTest().stagedir,
            'osu-micro-benchmarks-5.6.2', 'mpi', 'pt2pt', 'osu_bw'
        )
        self.executable_opts = ['-x', '100', '-i', '1000']


@rfm.parameterized_test(*([1 << i] for i in range(1, 5)))
class OSUAllreduceTest(OSUBenchmarkTestBase):
    def __init__(self, num_tasks):
        super().__init__()
        self.descr = 'OSU Allreduce test'
        self.perf_patterns = {
            'latency': sn.extractsingle(r'^8\s+(\S+)', self.stdout, 1, float)
        }
        self.depends_on('OSUBuildTest')
        self.reference = {
            '*': {'latency': (0, None, None, 'us')}
        }
        self.num_tasks = num_tasks

    @rfm.require_deps
    def set_executable(self, OSUBuildTest):
        self.executable = os.path.join(
            OSUBuildTest().stagedir,
            'osu-micro-benchmarks-5.6.2', 'mpi', 'collective', 'osu_allreduce'
        )
        self.executable_opts = ['-m', '8', '-x', '1000', '-i', '20000']


@rfm.simple_test
class OSUBuildTest(rfm.CompileOnlyRegressionTest):
    def __init__(self):
        self.descr = 'OSU benchmarks build test'
        self.valid_systems = ['daint:gpu']
        self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-pgi', 'PrgEnv-intel']
        self.sourcesdir = None
        self.prebuild_cmd = [
            'wget http://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-5.6.2.tar.gz',
            'tar xzf osu-micro-benchmarks-5.6.2.tar.gz',
            'cd osu-micro-benchmarks-5.6.2'
        ]
        self.build_system = 'Autotools'
        self.build_system.max_concurrency = 8
        self.sanity_patterns = sn.assert_not_found('error', self.stderr)
