# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.udeps as udeps


class OSUBenchmarkTestBase(rfm.RunOnlyRegressionTest):
    '''Base class of OSU benchmarks runtime tests'''

    valid_systems = ['daint:gpu']
    valid_prog_environs = ['gnu', 'pgi', 'intel']
    sourcesdir = None
    num_tasks = 2
    num_tasks_per_node = 1

    @run_after('init')
    def set_dependencies(self):
        self.depends_on('OSUBuildTest', udeps.by_env)

    @run_before('sanity')
    def set_sanity_patterns(self):
        self.sanity_patterns = sn.assert_found(r'^8', self.stdout)


@rfm.simple_test
class OSULatencyTest(OSUBenchmarkTestBase):
    descr = 'OSU latency test'
    reference = {
        '*': {'latency': (0, None, None, 'us')}
    }

    @require_deps
    def set_executable(self, OSUBuildTest):
        self.executable = os.path.join(
            OSUBuildTest().stagedir,
            'mpi', 'pt2pt', 'osu_latency'
        )
        self.executable_opts = ['-x', '100', '-i', '1000']

    @run_before('performance')
    def set_perf_patterns(self):
        self.perf_patterns = {
            'latency': sn.extractsingle(r'^8\s+(\S+)', self.stdout, 1, float)
        }


@rfm.simple_test
class OSUBandwidthTest(OSUBenchmarkTestBase):
    descr = 'OSU bandwidth test'
    reference = {
        '*': {'bandwidth': (0, None, None, 'MB/s')}
    }

    @require_deps
    def set_executable(self, OSUBuildTest):
        self.executable = os.path.join(
            OSUBuildTest().stagedir,
            'mpi', 'pt2pt', 'osu_bw'
        )
        self.executable_opts = ['-x', '100', '-i', '1000']

    @run_before('performance')
    def set_perf_patterns(self):
        self.perf_patterns = {
            'bandwidth': sn.extractsingle(r'^4194304\s+(\S+)',
                                          self.stdout, 1, float)
        }


@rfm.simple_test
class OSUAllreduceTest(OSUBenchmarkTestBase):
    mpi_tasks = parameter(1 << i for i in range(1, 5))
    descr = 'OSU Allreduce test'
    reference = {
        '*': {'latency': (0, None, None, 'us')}
    }

    @run_after('init')
    def set_num_tasks(self):
        self.num_tasks = self.mpi_tasks

    @require_deps
    def set_executable(self, OSUBuildTest):
        self.executable = os.path.join(
            OSUBuildTest().stagedir,
            'mpi', 'collective', 'osu_allreduce'
        )
        self.executable_opts = ['-m', '8', '-x', '1000', '-i', '20000']

    @run_before('performance')
    def set_perf_patterns(self):
        self.perf_patterns = {
            'latency': sn.extractsingle(r'^8\s+(\S+)', self.stdout, 1, float)
        }


@rfm.simple_test
class OSUBuildTest(rfm.CompileOnlyRegressionTest):
    descr = 'OSU benchmarks build test'
    valid_systems = ['daint:gpu']
    valid_prog_environs = ['gnu', 'pgi', 'intel']
    build_system = 'Autotools'

    @run_after('init')
    def inject_dependencies(self):
        self.depends_on('OSUDownloadTest', udeps.fully)

    @require_deps
    def set_sourcedir(self, OSUDownloadTest):
        self.sourcesdir = os.path.join(
            OSUDownloadTest(part='login', environ='builtin').stagedir,
            'osu-micro-benchmarks-5.6.2'
        )

    @run_before('compile')
    def set_build_system_attrs(self):
        self.build_system.max_concurrency = 8

    @run_before('sanity')
    def set_sanity_patterns(self):
        self.sanity_patterns = sn.assert_not_found('error', self.stderr)


@rfm.simple_test
class OSUDownloadTest(rfm.RunOnlyRegressionTest):
    descr = 'OSU benchmarks download sources'
    valid_systems = ['daint:login']
    valid_prog_environs = ['builtin']
    executable = 'wget'
    executable_opts = [
        'http://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-5.6.2.tar.gz'  # noqa: E501
    ]
    postrun_cmds = [
        'tar xzf osu-micro-benchmarks-5.6.2.tar.gz'
    ]

    @run_before('sanity')
    def set_sanity_patterns(self):
        self.sanity_patterns = sn.assert_not_found('error', self.stderr)
