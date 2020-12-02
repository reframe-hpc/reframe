# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.udeps as udeps

OSU_BENCH_VERSION = '5.6.3'
NUM_NODES = 10


def tsa_node_pairs():
    def nodeid(n):
        return f'tsa-pp{n+11:03}'

    for u in range(NUM_NODES):
        for v in range(NUM_NODES):
            if u < v:
                yield (nodeid(u), nodeid(v))


def count_hops(u, v):
    switches = {
        's0': {'tsa-pp011', 'tsa-pp012', 'tsa-pp013', 'tsa-pp014'},
        's1': {'tsa-pp015', 'tsa-pp016', 'tsa-pp017'},
        's2': {'tsa-pp018', 'tsa-pp019', 'tsa-pp020'}
    }

    for group in switches.values():
        if u in group and v in group:
            return 1
        elif u in group:
            return 2
        elif v in group:
            return 2

    # This should not happen; u, v must be in a node group
    assert 0


@rfm.simple_test
class OSUDownloadTest(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.descr = 'OSU benchmarks download sources'
        self.valid_systems = ['tsa:login']
        self.valid_prog_environs = ['PrgEnv-gnu-nocuda']
        self.tags = {'production'}
        self.executable = 'wget'
        self.executable_opts = [
            f'http://mvapich.cse.ohio-state.edu/download/mvapich/'
            f'osu-micro-benchmarks-{OSU_BENCH_VERSION}.tar.gz'
        ]
        self.postrun_cmds = [
            f'tar xzf osu-micro-benchmarks-{OSU_BENCH_VERSION}.tar.gz'
        ]
        self.sanity_patterns = sn.assert_not_found('error', self.stderr)


@rfm.simple_test
class OSUBuildTest(rfm.CompileOnlyRegressionTest):
    def __init__(self):
        self.descr = 'OSU benchmarks build test'
        self.valid_systems = ['tsa:login']
        self.valid_prog_environs = ['PrgEnv-gnu-nocuda']
        self.build_system = 'Autotools'
        self.build_system.max_concurrency = 8
        self.sanity_patterns = sn.assert_not_found('error', self.stderr)
        self.depends_on('OSUDownloadTest')
        self.tags = {'production'}

    @rfm.require_deps
    def set_sourcedir(self, OSUDownloadTest):
        self.sourcesdir = os.path.join(
            OSUDownloadTest().stagedir,
            f'osu-micro-benchmarks-{OSU_BENCH_VERSION}'
        )


class OSUBaseRunTest(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.valid_systems = ['tsa:pn']
        self.valid_prog_environs = ['PrgEnv-gnu-nocuda']
        self.sanity_patterns = sn.assert_found(r'^1048576', self.stdout)
        self.maintainers = ['MKr, AJ']
        self.tags = {'production'}
        self.perf_patterns = {
            'latency': sn.extractsingle(r'^8\s+(?P<latency>\S+)',
                                        self.stdout, 'latency', float)
        }
        self.executable_opts = ['-x', '1000', '-i', '5000']
        self.exclusive_access = True
        self.depends_on('OSUBuildTest', udeps.fully)


@rfm.parameterized_test(
    ['tsa-pp011', 'tsa-pp012', 'tsa-pp013', 'tsa-pp014'],
    ['tsa-pp015', 'tsa-pp016', 'tsa-pp017'],
    ['tsa-pp018', 'tsa-pp019', 'tsa-pp020'],
    ['tsa-pp011', 'tsa-pp012', 'tsa-pp013', 'tsa-pp014', 'tsa-pp015',
     'tsa-pp016', 'tsa-pp017', 'tsa-pp018', 'tsa-pp019', 'tsa-pp020']
)
class OSUAlltoallvTest(OSUBaseRunTest):

    def __init__(self, *node_list):
        super().__init__()
        self.num_tasks_per_node = 2
        self.num_tasks = len(node_list)*self.num_tasks_per_node
        self.executable_opts += ['-f']
        self.node_list = node_list

    @rfm.require_deps
    def set_executable(self, OSUBuildTest):
        self.executable = os.path.join(
            OSUBuildTest(part='login').stagedir,
            'mpi', 'collective', 'osu_alltoallv'
        )

    @rfm.run_before('run')
    def prepare_run(self):
        self.job.options += [
            f'--nodelist={",".join(self.node_list)}'
        ]

        # Run the affinity program first
        launcher_cmd = self.job.launcher.run_command(self.job)
        self.prerun_cmds = [f'{launcher_cmd} ./affinity']


@rfm.parameterized_test(
    ['tsa-pp011', 'tsa-pp012', 'tsa-pp013', 'tsa-pp014'],
    ['tsa-pp015', 'tsa-pp016', 'tsa-pp017'],
    ['tsa-pp018', 'tsa-pp019', 'tsa-pp020'],
    ['tsa-pp011', 'tsa-pp012', 'tsa-pp013', 'tsa-pp014', 'tsa-pp015',
     'tsa-pp016', 'tsa-pp017', 'tsa-pp018', 'tsa-pp019', 'tsa-pp020']
)
class OSUAllgathervTest(OSUBaseRunTest):

    def __init__(self, *node_list):
        super().__init__()
        self.num_tasks_per_node = 2
        self.num_tasks = len(node_list)*self.num_tasks_per_node
        self.executable_opts += ['-f']
        self.node_list = node_list

    @rfm.require_deps
    def set_executable(self, OSUBuildTest):
        self.executable = os.path.join(
            OSUBuildTest(part='login').stagedir,
            'mpi', 'collective', 'osu_allgatherv'
        )

    @rfm.run_before('run')
    def prepare_run(self):
        self.job.options += [
            f'--nodelist={",".join(self.node_list)}'
        ]

        # Run the affinity program first
        launcher_cmd = self.job.launcher.run_command(self.job)
        self.prerun_cmds = [f'{launcher_cmd} ./affinity']


@rfm.parameterized_test(
    ['tsa-pp011', 'tsa-pp012', 'tsa-pp013', 'tsa-pp014'],
    ['tsa-pp015', 'tsa-pp016', 'tsa-pp017'],
    ['tsa-pp018', 'tsa-pp019', 'tsa-pp020'],
    ['tsa-pp011', 'tsa-pp012', 'tsa-pp013', 'tsa-pp014', 'tsa-pp015',
     'tsa-pp016', 'tsa-pp017', 'tsa-pp018', 'tsa-pp019', 'tsa-pp020']
)
class OSUIBcastTest(OSUBaseRunTest):

    def __init__(self, *node_list):
        super().__init__()
        self.num_tasks_per_node = 2
        self.num_tasks = len(node_list)*self.num_tasks_per_node
        self.executable_opts += ['-f']
        self.node_list = node_list

    @rfm.require_deps
    def set_executable(self, OSUBuildTest):
        self.executable = os.path.join(
            OSUBuildTest(part='login').stagedir,
            'mpi', 'collective', 'osu_ibcast'
        )

    @rfm.run_before('run')
    def prepare_run(self):
        self.job.options += [
            f'--nodelist={",".join(self.node_list)}'
        ]

        # Run the affinity program first
        launcher_cmd = self.job.launcher.run_command(self.job)
        self.prerun_cmds = [f'{launcher_cmd} ./affinity']


@rfm.parameterized_test(*((np, c)
                          for np in tsa_node_pairs()
                          for c in [0, 20]))
class OSULatencyTest(OSUBaseRunTest):
    def __init__(self, node_pairs, cpu_no):
        super().__init__()
        self.num_tasks = 2
        self.num_tasks_per_node = 1
        self.node_pairs = node_pairs
        self.cpu_no = cpu_no
        self.executable_opts += ['-m', '8']
        cpu_pinned = sn.count(
            sn.extractall(rf'CPU affinity: \[\s+{cpu_no}\]', self.stdout)
        )
        self.sanity_patterns = sn.all([
            sn.assert_eq(cpu_pinned, 2),
            sn.assert_found(r'^8\s', self.stdout),
        ])

        num_hops = count_hops(*node_pairs)
        if cpu_no == 0 and num_hops == 1:
            self.reference = {
                'tsa:pn': {'latency': (1.25, -0.05, 0.05, 'us')}
            }
        elif cpu_no == 20 and num_hops == 1:
            self.reference = {
                'tsa:pn': {'latency': (1.45, -0.05, 0.05, 'us')}
            }
        elif cpu_no == 0 and num_hops == 2:
            self.reference = {
                'tsa:pn': {'latency': (1.95, -0.05, 0.05, 'us')}
            }
        elif cpu_no == 20 and num_hops == 2:
            self.reference = {
                'tsa:pn': {'latency': (2.25, -0.05, 0.05, 'us')}
            }

    @rfm.require_deps
    def set_executable(self, OSUBuildTest):
        self.executable = os.path.join(
            OSUBuildTest(part='login').stagedir,
            'mpi', 'pt2pt', 'osu_latency'
        )

    @rfm.run_before('run')
    def prepare_run(self):
        self.job.options += [
            f'--nodelist={",".join(self.node_pairs)}'
        ]
        self.job.launcher.options = [f'--cpu-bind=map_cpu:{self.cpu_no}']

        # Run the affinity program first
        launcher_cmd = self.job.launcher.run_command(self.job)
        self.prerun_cmds = [f'{launcher_cmd} ./affinity']


@rfm.parameterized_test(*tsa_node_pairs())
class OSUBandwidthTest(OSUBaseRunTest):
    def __init__(self, *node_pairs):
        super().__init__()
        self.num_tasks = 2
        self.num_tasks_per_node = 1
        self.node_pairs = node_pairs
        self.executable_opts = ['-x', '1000', '-i', '1500']
        self.perf_patterns = {
            'bandwidth': sn.extractsingle(r'^4194304\s+(?P<bandwidth>\S+)',
                                          self.stdout, 'bandwidth', float)
        }
        self.reference = {
            'tsa:pn': {'bandwidth': (12000, -0.05, 0.05, 'MB/s')}
        }

    @rfm.require_deps
    def set_executable(self, OSUBuildTest):
        self.executable = os.path.join(
            OSUBuildTest(part='login').stagedir,
            'mpi', 'pt2pt', 'osu_bw'
        )

    @rfm.run_before('run')
    def set_nodelist(self):
        self.job.options += [f'--nodelist={",".join(self.node_pairs)}']
