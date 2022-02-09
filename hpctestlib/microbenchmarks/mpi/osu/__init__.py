# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import reframe as rfm
import reframe.utility.sanity as sn


__all__ = ['alltoall', 'flex_alltoall', 'allreduce',
           'p2p_bandwidth', 'p2p_latency']


class fetch_osu_benchmarks(rfm.RunOnlyRegressionTest):
    #: The version of OSU benchmarks to use.
    #:
    #: :type: :class:`str`
    #: :default: ``'5.6.2'``
    version = variable(str, value='5.6.2')

    descr = 'Fetch OSU benchmarks'
    local = True
    executable = 'wget'
    executable_opts = [
        f'http://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-{version}.tar.gz'  # noqa: E501
    ]

    @sanity_function
    def validate_download(self):
        return sn.assert_eq(self.job.exitcode, 0)


class build_osu_benchmarks(rfm.CompileOnlyRegressionTest):
    #: The version of OSU benchmarks to use.
    #:
    #: :type: :class:`str`
    #: :default: ``'5.6.2'``
    version = variable(str, value='5.6.2')

    descr = 'Build OSU benchmarks'
    build_system = 'Autotools'
    build_prefix = variable(str)
    osu_benchmarks = fixture(fetch_osu_benchmarks, scope='session',
                             variables={'version': f'{version}'})

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


class osu_benchmark_test_base(rfm.RunOnlyRegressionTest):
    '''Base class of OSU benchmarks runtime tests'''
    #: The version of OSU benchmarks to use.
    #:
    #: :type: :class:`str`
    #: :default: ``'5.6.2'``
    version = variable(str, value='5.6.2')

    #: Maximum message size
    #: This value is passed to the excutable through the -m option
    #:
    #: :type: :class:`int`
    #: :default: ``8``
    max_message_size = variable(int, value=8)

    #: Number of warmup iterations
    #: This value is passed to the excutable through the -x option
    #:
    #: :type: :class:`int`
    #: :default: ``1000``
    num_warmup_iters = variable(int, value=1000)

    #: Number of warmup iterations
    #: This value is passed to the excutable through the -i option
    #:
    #: :type: :class:`int`
    #: :default: ``20000``
    num_iters = variable(int, value=20000)

    osu_binaries = fixture(build_osu_benchmarks, scope='environment',
                           variables={'version': f'{version}'})

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


class alltoall(osu_benchmark_test_base):
    descr = 'Alltoall OSU microbenchmark'
    num_tasks_per_node = 1
    num_gpus_per_node  = 1

    @run_before('run')
    def set_executable(self):
        self.executable = os.path.join(
            self.mpi_tests_dir, 'collective', 'osu_alltoall',
        )
        self.executable_opts = ['-m', f'{self.max_message_size}',
                                '-x', f'{self.num_warmup_iters}',
                                '-i', f'{self.num_iters}']

    @sanity_function
    def assert_found_8MB_latency(self):
        return sn.assert_found(r'^8', self.stdout)

    @run_before('performance')
    def set_performance_patterns(self):
        self.perf_patterns = {
            'latency': sn.extractsingle(r'^8\s+(?P<latency>\S+)',
                                        self.stdout, 'latency', float)
        }


class flex_alltoall(osu_benchmark_test_base):
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


class allreduce(osu_benchmark_test_base):
    descr = 'Allreduce OSU microbenchmark'
    num_tasks_per_node = 1
    num_gpus_per_node  = 1

    @run_before('run')
    def set_executable(self):
        self.executable = os.path.join(
            self.mpi_tests_dir, 'collective', 'osu_allreduce',
        )
        self.executable_opts = ['-m', f'{self.max_message_size}',
                                '-x', f'{self.num_warmup_iters}',
                                '-i', f'{self.num_iters}']

    @sanity_function
    def assert_found_8MB_latency(self):
        return sn.assert_found(r'^8', self.stdout)


class p2p_test_base(osu_benchmark_test_base):
    #: Accelerator device type
    #:
    #: Use accelerator device buffers, i.e cuda, openacc or rocm
    #:
    #: :type: :class:`str`
    #: :default: ``None``
    device = variable(str, type(None), value=None)

    descr = 'P2P microbenchmark'
    num_warmup_iters = 100
    num_iters = 1000
    num_tasks = 2
    num_tasks_per_node = 1

    @run_before('run')
    def set_executable_opts(self):
        self.executable_opts = ['-x', f'{self.num_warmup_iters}',
                                '-i', f'{self.num_iters}']
        if self.device:
            self.executable_opts += ['-d', f'{self.device}', 'D', 'D']

    @sanity_function
    def assert_found_4KB_bw(self):
        return sn.assert_found(r'^4194304', self.stdout)


class p2p_bandwidth(p2p_test_base):
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


class p2p_latency(p2p_test_base):
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
