# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.udeps as udeps
from reframe.core.launchers import JobLauncher

OSU_BENCH_VERSION = '5.6.3'
NUM_NODES = 7


def tsa_node_pairs():
    def nodeid(n):
        return f'tsa-pp{n+8:03}'

    for u in range(NUM_NODES):
        for v in range(NUM_NODES):
            if u < v:
                yield (nodeid(u), nodeid(v))


@rfm.simple_test
class OSUDownloadTest(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.descr = 'OSU benchmarks download sources'
        self.valid_systems = ['tsa:login']
        self.valid_prog_environs = ['PrgEnv-gnu-nocuda']
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
        self.perf_patterns = {
            'latency': sn.extractsingle(r'^8\s+(?P<latency>\S+)',
                                        self.stdout, 'latency', float)
        }
        self.reference = {
            '*': {'latency': (0, None, None, 'us')}
        }
        self.executable_opts = ['-x', '1000', '-i', '20000']
        self.exclusive_access = True
        self.depends_on('OSUBuildTest', udeps.fully)


# @rfm.parameterized_test(*([1 << i] for i in range(1)))
@rfm.parameterized_test([2])
class OSUAlltoallvTest(OSUBaseRunTest):

    def __init__(self, num_tasks_per_node):
        super().__init__()
        self.num_tasks = NUM_NODES*num_tasks_per_node
        self.num_tasks_per_node = num_tasks_per_node
        self.executable_opts += ['-f']

    @rfm.require_deps
    def set_executable(self, OSUBuildTest):
        self.executable = os.path.join(
            OSUBuildTest(part='login').stagedir,
            'mpi', 'collective', 'osu_alltoallv'
        )


# @rfm.parameterized_test(*([1 << i] for i in range(1)))
@rfm.parameterized_test([2])
class OSUAllgathervTest(OSUBaseRunTest):

    def __init__(self, num_tasks_per_node):
        super().__init__()
        self.num_tasks = NUM_NODES*num_tasks_per_node
        self.num_tasks_per_node = num_tasks_per_node
        self.executable_opts += ['-f']

    @rfm.require_deps
    def set_executable(self, OSUBuildTest):
        self.executable = os.path.join(
            OSUBuildTest(part='login').stagedir,
            'mpi', 'collective', 'osu_allgatherv'
        )


# @rfm.parameterized_test(*([1 << i] for i in range(1)))
@rfm.parameterized_test([2])
class OSUIBcastTest(OSUBaseRunTest):

    def __init__(self, num_tasks_per_node):
        super().__init__()
        self.num_tasks = NUM_NODES*num_tasks_per_node
        self.num_tasks_per_node = num_tasks_per_node
        self.executable_opts += ['-f']

    @rfm.require_deps
    def set_executable(self, OSUBuildTest):
        self.executable = os.path.join(
            OSUBuildTest(part='login').stagedir,
            'mpi', 'collective', 'osu_ibcast'
        )


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

    @rfm.require_deps
    def set_executable(self, OSUBuildTest):
        self.executable = os.path.join(
            OSUBuildTest(part='login').stagedir,
            'mpi', 'pt2pt', 'osu_bw'
        )

    @rfm.run_before('run')
    def set_nodelist(self):
        self.job.options += [f'--nodelist={",".join(self.node_pairs)}']
