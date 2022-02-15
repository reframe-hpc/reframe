# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import reframe as rfm
import reframe.utility.sanity as sn


__all__ = ['fetch_osu_benchmarks', 'build_osu_benchmarks',
           'osu_benchmark_test_base', 'osu_bandwidth', 'osu_latency']


class fetch_osu_benchmarks(rfm.RunOnlyRegressionTest):
    #: The version of OSU benchmarks to use.
    #:
    #: :type: :class:`str`
    #: :default: ``'5.6.2'``
    version = variable(str, value='5.6.2')

    descr = 'Fetch OSU benchmarks'
    local = True
    # executable = 'wget'
    executable = 'cp'

    @run_after('init')
    def set_executable_opts(self):
        self.executable_opts = [
            # f'http://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-{self.version}.tar.gz'  # noqa: E501
            f'/users/sarafael/git/reframe/osu-micro-benchmarks-{self.version}.tar.gz .'
        ]

    @sanity_function
    def validate_download(self):
        return sn.assert_eq(self.job.exitcode, 0)


class build_osu_benchmarks(rfm.CompileOnlyRegressionTest):
    #: Option to build with support for accelerators
    #:
    #: :type: :class:`str`
    #: :default: ``'cpu'``
    build_type = parameter(['cpu', 'cuda', 'rocm', 'openacc'])

    #: Install directory of the accelerator libraries
    #:
    #: :type: :class:`str`
    #: :default: ``None``
    gpu_lib_dir = variable(str, type(None), value=None)

    descr = 'Build OSU benchmarks'
    build_system = 'Autotools'
    build_prefix = variable(str)
    osu_benchmarks = fixture(fetch_osu_benchmarks, scope='session')
    build_flags = {
        'cpu': [],
        'cuda': ['--enable-cuda'],
        'rocm': ['--enable-rocm'],
        'openacc': ['--enable-openacc'],
    }

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
        self.build_system.config_opts = self.build_flags[self.build_type]
        self.build_system.make_opts = ['-C', 'mpi']
        self.build_system.max_concurrency = 8

    @sanity_function
    def validate_build(self):
        return True


class osu_benchmark_test_base(rfm.RunOnlyRegressionTest):
    '''Base class of OSU benchmarks runtime tests'''
    #: Number of warmup iterations
    #:
    #: This value is passed to the excutable through the -x option
    #:
    #: :type: :class:`int`
    #: :default: ``1000``
    num_warmup_iters = variable(int, value=1000)

    #: Number of iterations
    #: This value is passed to the excutable through the -i option
    #:
    #: :type: :class:`int`
    #: :default: ``20000``
    num_iters = variable(int, value=20000)

    #: Control message size
    #:
    #: When this value is present on the stdout, the check can be considered successful
    #:
    #: :type: :class:`int`
    #: :default: ``8``
    ctrl_msg_size = variable(int, value=8)

    #: Performance message size
    #:
    #: This value is used for the performance checks
    #:
    #: :type: :class:`int`
    #: :default: ``8``
    perf_msg_size = variable(int, value=8)

    #: Accelerator device type
    #:
    #: Use accelerator device buffers, i.e cuda, openacc or rocm
    #:
    #: :type: :class:`str`
    #: :default: ``None``
    device = variable(str, type(None), value=None)

    osu_binaries = fixture(build_osu_benchmarks, scope='environment')
    executables = {
        'collective': ['osu_alltoall', 'osu_allreduce'],
        'pt2pt': ['osu_bw', 'osu_latency']
    }

    @run_after('setup')
    def set_executable(self):
        if self.executable in self.executables['collective']:
            benchmark_type = 'collective'
        elif self.executable in self.executables['pt2pt']:
            benchmark_type = 'pt2pt'

        self.executable = os.path.join(
            self.osu_binaries.stagedir, self.osu_binaries.build_prefix,
            'mpi', benchmark_type, self.executable
        )
        max_message_size = max(self.ctrl_msg_size, self.perf_msg_size)
        self.executable_opts = ['-m', f'{max_message_size}',
                                '-x', f'{self.num_warmup_iters}',
                                '-i', f'{self.num_iters}']
        if self.device:
            self.executable_opts += ['-d', f'{self.device}']

        if benchmark_type == 'pt2pt':
            self.executable_opts += ['D', 'D']

    @sanity_function
    def validate_test(self):
        return sn.assert_found(rf'^{self.ctrl_msg_size}', self.stdout)


class osu_bandwidth(osu_benchmark_test_base):
    @run_before('performance')
    def set_performance_patterns(self):
        self.perf_patterns = {
            'bw': sn.extractsingle(rf'^{self.perf_msg_size}\s+(?P<bw>\S+)',
                                   self.stdout, 'bw', float)
        }


class osu_latency(osu_benchmark_test_base):
    @run_before('performance')
    def set_performance_patterns(self):
        self.perf_patterns = {
            'latency': sn.extractsingle(
                rf'^{self.perf_msg_size}\s+(?P<latency>\S+)',
                self.stdout, 'latency', float
            )
        }
