# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.udeps as udeps

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
            f'http://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-{OSU_BENCH_VERSION}.tar.gz']
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


@rfm.parameterized_test(*([1 << i] for i in range(6)))
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


@rfm.parameterized_test(*([1 << i] for i in range(6)))
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


@rfm.parameterized_test(*([1 << i] for i in range(6)))
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


@rfm.parameterized_test(*tsa_node_pairs())
class OSULatencyTest(OSUBaseRunTest):
    def __init__(self, *node_pairs):
        super().__init__()
        self.num_tasks = 2
        self.num_tasks_per_node = 1
        self.node_pairs = node_pairs

    @rfm.require_deps
    def set_executable(self, OSUBuildTest):
        self.executable = os.path.join(
            OSUBuildTest(part='login').stagedir,
            'mpi', 'pt2pt', 'osu_latency'
        )

    @rfm.run_before('run')
    def set_nodelist(self):
        self.job.options += [f'--nodelist={",".join(self.node_pairs)}']
