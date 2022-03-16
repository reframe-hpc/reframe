# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import reframe as rfm
import reframe.utility.sanity as sn


class fetch_osu_benchmarks(rfm.RunOnlyRegressionTest):
    #: The version of OSU benchmarks to use.
    #:
    #: :type: :class:`str`
    #: :default: ``'5.9'``
    version = variable(str, value='5.9')

    local = True
    osu_file_name = f'osu-micro-benchmarks-{version}.tar.gz'
    executable = f'curl -LJO http://mvapich.cse.ohio-state.edu/download/mvapich/{osu_file_name}'  # noqa: E501

    @sanity_function
    def validate_download(self):
        return sn.assert_eq(self.job.exitcode, 0)


class build_osu_benchmarks(rfm.CompileOnlyRegressionTest):
    #: Option to build with support for accelerators
    #:
    #: :type: :class:`str`
    #: :default: ``'cpu'``
    build_type = parameter(['cpu', 'cuda', 'rocm', 'openacc'])

    build_system = 'Autotools'
    build_prefix = variable(str)
    osu_benchmarks = fixture(fetch_osu_benchmarks, scope='session')

    @run_before('compile')
    def prepare_build(self):
        tarball = f'osu-micro-benchmarks-{self.osu_benchmarks.version}.tar.gz'
        self.build_prefix = tarball[:-7]  # remove .tar.gz extension
        fullpath = os.path.join(self.osu_benchmarks.stagedir, tarball)
        self.prebuild_cmds += [
            f'cp {fullpath} {self.stagedir}',
            f'tar xzf {tarball}',
            f'cd {self.build_prefix}'
        ]
        self.build_system.config_opts = [f'--enable-{self.build_type}']
        self.build_system.make_opts = ['-C', 'mpi']
        self.build_system.max_concurrency = 8

    @sanity_function
    def validate_build(self):
        return True


class osu_benchmark_test_base(rfm.RunOnlyRegressionTest):
    '''Base class of OSU benchmarks runtime tests'''

    #: Number of warmup iterations
    #:
    #: This value is passed to the excutable through the -x option.
    #:
    #: :type: :class:`int`
    #: :default: ``1000``
    num_warmup_iters = variable(int, value=1000)

    #: Number of iterations
    #:
    #: This value is passed to the excutable through the -i option.
    #:
    #: :type: :class:`int`
    #: :default: ``20000``
    num_iters = variable(int, value=20000)

    #: Maximum message size
    #:
    #: Both the performance and the sanity checks will be done
    #: for this message size
    #:
    #: :type: :class:`int`
    #: :default: ``8``
    message_size = variable(int, value=8)

    #: Device buffers
    #:
    #: Use accelerator device buffers, i.e cuda, openacc or rocm
    #:
    #: :type: :class:`str`
    #: :default: ``None``
    device_buffers = variable(str, type(None), value='cpu')

    executable = ''
    microbenchmarks = {
        'collective': ['osu_alltoall', 'osu_allreduce'],
        'pt2pt': ['osu_bw', 'osu_latency']
    }
    osu_binaries = fixture(build_osu_benchmarks, scope='environment')

    @run_after('setup')
    def set_executable(self):
        if self.executable in self.microbenchmarks['collective']:
            benchmark_type = 'collective'
        elif self.executable in self.microbenchmarks['pt2pt']:
            benchmark_type = 'pt2pt'

        self.executable = os.path.join(
            self.osu_binaries.stagedir, self.osu_binaries.build_prefix,
            'mpi', benchmark_type, self.executable
        )
        max_message_size = max(self.message_size, self.message_size)
        self.executable_opts = ['-m', f'{max_message_size}',
                                '-x', f'{self.num_warmup_iters}',
                                '-i', f'{self.num_iters}']
        if self.device_buffers  and f'{self.device_buffers}' != 'cpu':
            self.executable_opts += ['-d', f'{self.device_buffers}']

            if benchmark_type == 'pt2pt':
                self.executable_opts += ['D', 'D']

    @sanity_function
    def validate_test(self):
        return sn.assert_found(rf'^{self.message_size}', self.stdout)


class osu_bandwidth(osu_benchmark_test_base):
    @performance_function('MB/s', perf_key='bw')
    def bandwidth(self):
        """Bandwidth for the message size `message_size`."""

        return sn.extractsingle(rf'^{self.message_size}\s+(?P<bw>\S+)',
                                self.stdout, 'bw', float)


class osu_latency(osu_benchmark_test_base):
    @performance_function('us', perf_key='latency')
    def latency(self):
        """Latency for the message size `message_size`."""

        return sn.extractsingle(rf'^{self.message_size}\s+(?P<latency>\S+)',
                                self.stdout, 'latency', float)


@rfm.simple_test
class osu_cpu_latency_pt2pt(osu_latency):
    @run_after('init')
    def set_job_options(self):
        self.executable = 'osu_latency'
        self.num_tasks = 2
        self.num_tasks_per_node = 1


@rfm.simple_test
class osu_cpu_bandwidth_pt2pt(osu_bandwidth):
    @run_after('init')
    def set_job_options(self):
        self.executable = 'osu_latency'
        self.num_tasks = 2
        self.num_tasks_per_node = 1


@rfm.simple_test
class osu_allreduce(osu_latency):
    @run_after('init')
    def set_job_options(self):
        self.executable = 'osu_allreduce'
        self.num_tasks = 4
        self.num_tasks_per_node = 1
