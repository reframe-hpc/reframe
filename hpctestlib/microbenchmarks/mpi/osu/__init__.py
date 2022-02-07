# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import reframe as rfm
import reframe.utility.sanity as sn


__all__ = ['Alltoall', 'FlexAlltoall', 'Allreduce',
           'P2PCPUBandwidth', 'P2PCPULatency',
           'G2GBandwidth', 'G2GLatency']


class fetch_osu_benchmarks(rfm.RunOnlyRegressionTest):
    descr = 'Fetch OSU benchmarks'
    version = variable(str, value='5.6.2')
    executable = 'cp'
    executable_opts = [
        f'http://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-{version}.tar.gz'  # noqa: E501
    ]
    local = True

    @sanity_function
    def validate_download(self):
        return sn.assert_eq(self.job.exitcode, 0)


class build_osu_benchmarks(rfm.CompileOnlyRegressionTest):
    descr = 'Build OSU benchmarks'
    build_system = 'Autotools'
    build_prefix = variable(str)
    osu_benchmarks = fixture(fetch_osu_benchmarks, scope='session')

    @run_before('compile')
    def prepare_build(self):
        tarball = f'osu-micro-benchmarks-{self.osu_benchmarks.version}.tar.gz'
        self.build_prefix = tarball[:-7]  # remove .tar.gz extension

        fullpath = os.path.join(self.osu_benchmarks.stagedir, tarball)
        self.prebuild_cmds = [
            f'cp {fullpath} {self.stagedir}',
            f'tar xzf {tarball}',
            f'cd {self.build_prefix}'
        ]
        self.build_system.make_opts = ['-C', 'mpi']
        self.build_system.max_concurrency = 8

    @sanity_function
    def validate_build(self):
        return True


class OSUBenchmarkTestBase(rfm.RunOnlyRegressionTest):
    '''Base class of OSU benchmarks runtime tests'''
    osu_binaries = fixture(build_osu_benchmarks, scope='environment')

    @sanity_function
    def validate_test(self):
        return sn.assert_found(r'^8', self.stdout)

    @run_after('setup')
    def set_mpi_tests_dir(self):
        self.mpi_tests_dir = os.path.join(
            self.osu_binaries.stagedir,
            self.osu_binaries.build_prefix,
            'mpi'
        )


class Alltoall(OSUBenchmarkTestBase):
    descr = 'Alltoall OSU microbenchmark'
    num_tasks_per_node = 1
    num_gpus_per_node  = 1

    @run_before('run')
    def set_executable(self):
        self.executable = os.path.join(
            self.mpi_tests_dir, 'collective', 'osu_alltoall',
        )
        # The -m option sets the maximum message size
        # The -x option sets the number of warm-up iterations
        # The -i option sets the number of iterations
        self.executable_opts = ['-m', '8', '-x', '1000', '-i', '20000']

    @sanity_function
    def assert_found_8MB_latency(self):
        return sn.assert_found(r'^8', self.stdout)

    @run_before('performance')
    def set_performance_patterns(self):
        self.perf_patterns = {
            'latency': sn.extractsingle(r'^8\s+(?P<latency>\S+)',
                                        self.stdout, 'latency', float)
        }


class FlexAlltoall(OSUBenchmarkTestBase):
    descr = 'Flexible Alltoall OSU test'
    num_tasks_per_node = 1
    num_tasks = 0

    @run_before('run')
    def set_executable(self):
        self.executable = os.path.join(
            self.mpi_tests_dir, 'collective', 'osu_alltoall',
        )

    @sanity_function
    def assert_found_1KB_bw(self):
        return sn.assert_found(r'^1048576', self.stdout)


class Allreduce(OSUBenchmarkTestBase):
    descr = 'Allreduce OSU microbenchmark'
    # The -x option controls the number of warm-up iterations
    # The -i option controls the number of iterations
    executable_opts = ['-m', '8', '-x', '1000', '-i', '20000']
    num_tasks_per_node = 1
    num_gpus_per_node  = 1

    @run_before('run')
    def set_executable(self):
        self.executable = os.path.join(
            self.mpi_tests_dir, 'collective', 'osu_allreduce',
        )

    @sanity_function
    def assert_found_8MB_latency(self):
        return sn.assert_found(r'^8', self.stdout)


class P2PBase(OSUBenchmarkTestBase):
    descr = 'P2P microbenchmark'
    num_tasks = 2
    num_tasks_per_node = 1
    osu_binaries = fixture(build_osu_benchmarks, scope='environment')

    @sanity_function
    def assert_found_4KB_bw(self):
        return sn.assert_found(r'^4194304', self.stdout)


class P2PCPUBandwidth(P2PBase):
    @run_before('run')
    def set_executable(self):
        self.executable = os.path.join(
            self.mpi_tests_dir, 'pt2pt', 'osu_bw'
        )

    @run_before('performance')
    def set_performance_patterns(self):
        self.perf_patterns = {
            'bw': sn.extractsingle(r'^4194304\s+(?P<bw>\S+)',
                                   self.stdout, 'bw', float)
        }


class P2PCPULatency(P2PBase):
    @run_before('run')
    def set_executable(self):
        self.executable = os.path.join(
            self.mpi_tests_dir, 'pt2pt', 'osu_latency'
        )

    @run_before('performance')
    def set_performance_patterns(self):
        self.perf_patterns = {
            'bw': sn.extractsingle(r'^4194304\s+(?P<bw>\S+)',
                                   self.stdout, 'bw', float)
        }
        self.perf_patterns = {
            'latency': sn.extractsingle(r'^8\s+(?P<latency>\S+)',
                                        self.stdout, 'latency', float)
        }


class G2GBandwidth(P2PBase):
    num_gpus_per_node = 1
    executable_opts = ['-x', '100', '-i', '1000', '-d',
                       'cuda', 'D', 'D']

    @run_before('run')
    def set_executable(self):
        self.executable = os.path.join(
            self.mpi_tests_dir, 'pt2pt', 'osu_bw'
        )

    @run_before('performance')
    def set_performance_patterns(self):
        self.perf_patterns = {
            'bw': sn.extractsingle(r'^4194304\s+(?P<bw>\S+)',
                                   self.stdout, 'bw', float)
        }


class G2GLatency(P2PBase):
    num_gpus_per_node = 1
    executable_opts = ['-x', '100', '-i', '1000', '-d',
                       'cuda', 'D', 'D']

    @run_before('run')
    def set_executable(self):
        self.executable = os.path.join(
            self.mpi_tests_dir, 'pt2pt', 'osu_latency'
        )

    @run_before('performance')
    def set_performance_patterns(self):
        self.perf_patterns = {
            'latency': sn.extractsingle(r'^8\s+(?P<latency>\S+)',
                                        self.stdout, 'latency', float)
        }
