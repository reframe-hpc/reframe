# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import reframe as rfm
import reframe.utility.sanity as sn


class fetch_osu_benchmarks(rfm.RunOnlyRegressionTest):
    '''Fixture for fetching the OSU benchmarks.'''

    #: The version of the benchmarks to fetch.
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
    '''Fixture for building the OSU benchmarks'''

    #: Build variant parameter.
    #:
    #: :type: :class:`str`
    #: :values: ``'cpu', 'cuda', 'rocm', 'openacc'``
    build_type = parameter(['cpu', 'cuda', 'rocm', 'openacc'])

    build_system = 'Autotools'
    build_prefix = variable(str)

    #: The fixture object that retrieves the benchmarks
    #:
    #: :type: :class:`fetch_osu_benchmarks`
    #: :scope: *session*
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
        # If build fails, the test will fail before reaching this point.
        return True


class osu_benchmark(rfm.RunOnlyRegressionTest):
    '''OSU benchmark test base class.'''

    #: Number of warmup iterations.
    #:
    #: This value is passed to the excutable through the -x option.
    #:
    #: :type: :class:`int`
    #: :default: ``10``
    num_warmup_iters = variable(int, value=10)

    #: Number of iterations.
    #:
    #: This value is passed to the excutable through the -i option.
    #:
    #: :type: :class:`int`
    #: :default: ``1000``
    num_iters = variable(int, value=1000)

    #: Maximum message size.
    #:
    #: Both the performance and the sanity checks will be done
    #: for this message size.
    #:
    #: This value is set to ``8`` for latency benchmarks and to ``4194304`` for
    #: bandwidth benchmarks.
    #:
    #: :type: :class:`int`
    message_size = variable(int)

    #: Device buffers.
    #:
    #: Use accelerator device buffers.
    #: Valid values are ``cpu``, ``cuda``, ``openacc`` or ``rocm``.
    #:
    #: :type: :class:`str`
    #: :default: ``'cpu'``
    device_buffers = variable(str, value='cpu')

    #: Number of tasks to use.
    #:
    #: This variable is required.
    #: It is set to ``2`` for point to point benchmarks, but it is undefined
    #: for collective benchmarks
    #:
    #: :required: Yes
    num_tasks = required
    num_tasks_per_node = 1

    #: Parameter indicating the available benchmark to execute.
    #:
    #: :type: 2-element tuple containing the benchmark name and whether latency
    #:   or bandwidth is to be measured.
    #:
    #: :values:
    #:   ``mpi.collective.osu_alltoall``,
    #:   ``mpi.collective.osu_allreduce``,
    #:   ``mpi.pt2pt.osu_bw``,
    #:   ``mpi.pt2pt.osu_latency``
    benchmark_info = parameter([
        ('mpi.collective.osu_alltoall', 'latency'),
        ('mpi.collective.osu_allreduce', 'latency'),
        ('mpi.pt2pt.osu_bw', 'bandwidth'),
        ('mpi.pt2pt.osu_latency', 'latency')
    ], fmt=lambda x: x[0], loggable=True)

    @run_before('setup')
    def setup_per_benchmark(self):
        bench, bench_metric = self.benchmark_info
        if bench_metric == 'latency':
            self.message_size = 8
            unit = 'us'
        elif bench_metric == 'bandwidth':
            self.message_size = 4194304
            unit = 'MB/s'
        else:
            raise ValueError(f'unknown benchmark metric: {bench_metric}')

        self.executable = bench.split('.')[-1]
        self.executable_opts = ['-m', f'{self.message_size}',
                                '-x', f'{self.num_warmup_iters}',
                                '-i', f'{self.num_iters}', '-c']

        if self.device_buffers != 'cpu':
            self.executable_opts += ['-d', self.device_buffers]

        if bench.startswith('mpi.pt2pt'):
            self.executable_opts += ['D', 'D']
            self.num_tasks = 2

        self.perf_variables = {
            bench_metric: sn.make_performance_function(
                self._extract_metric, unit
            )
        }

    @sanity_function
    def validate_test(self):
        return sn.assert_found(rf'^{self.message_size}.*Pass', self.stdout)

    @deferrable
    def _extract_metric(self):
        return sn.extractsingle(rf'^{self.message_size}\s+(\S+)',
                                self.stdout, 1, float)


@rfm.simple_test
class osu_run(osu_benchmark):
    '''Run-only OSU benchmark test'''


@rfm.simple_test
class osu_build_run(osu_benchmark):
    '''OSU benchmark test (build and run)'''

    #: The fixture object that builds the OSU binaries
    #:
    #: :type: :class:`build_osu_benchmarks`
    #: :scope: *environment*
    osu_binaries = fixture(build_osu_benchmarks, scope='environment')

    @run_before('run')
    def prepend_build_prefix(self):
        bench_path = self.benchmark_info[0].replace('.', '/')
        self.executable = os.path.join(self.osu_binaries.stagedir,
                                       self.osu_binaries.build_prefix,
                                       bench_path)
