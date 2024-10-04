# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.typecheck as typ
import reframe.utility.sanity as sn


class gpu_burn_build(rfm.CompileOnlyRegressionTest, pin_prefix=True):
    '''Fixture for building the GPU burn benchmark.

    .. list-table:: Summary
       :widths: 20 40 40
       :header-rows: 1

       * - Variables
         - Parameters
         - Fixtures
       * - - :attr:`gpu_arch`
           - :attr:`gpu_build`
         - *None*
         - *None*
    '''

    #: Set the build option to either ``'cuda'`` or ``'hip'``.
    #:
    #: :type: :class:`str`
    #: :default: ``'cuda'``
    gpu_build = variable(str, type(None), value=None)

    #: Set the GPU architecture.
    #:
    #: This variable will be passed to the compiler to generate the
    #: arch-specific code.
    #:
    #: :type: :class:`str` or :obj:`None`
    #: :default: ``None``
    gpu_arch = variable(str, type(None), value=None)

    descr = 'GPU burn test build fixture'
    sourcesdir = 'src/gpu_burn'
    build_system = 'Make'

    @run_before('compile')
    def setup_build(self):
        curr_part = self.current_partition
        curr_env = self.current_environ

        if self.gpu_build is None:
            # Try to set the build type from the partition features
            if 'cuda' in curr_env.features:
                self.gpu_build = 'cuda'
            elif 'hip' in curr_env.features:
                self.gpu_build = 'hip'

        gpu_devices = curr_part.select_devices('gpu')
        if self.gpu_arch is None and gpu_devices:
            # Try to set the gpu arch from the partition's devices; we assume
            # all devices are of the same architecture
            self.gpu_arch = gpu_devices[0].arch

        if self.gpu_build == 'cuda':
            self.build_system.makefile = 'makefile.cuda'
            if self.gpu_arch:
                cc = self.gpu_arch.replace('sm_', 'compute_')
                self.build_system.cxxflags = [f'-arch={cc}',
                                              f'-code={self.gpu_arch}']
        elif self.gpu_build == 'hip':
            self.build_system.makefile = 'makefile.hip'
            if self.gpu_arch:
                self.build_system.cxxflags = [
                    f'--amdgpu-target={self.gpu_arch}'
                ]
        else:
            raise ValueError(f'unknown build variant: {self.gpu_build!r}')

    @sanity_function
    def valid_build(self):
        return True


@rfm.simple_test
class gpu_burn_check(rfm.RunOnlyRegressionTest):
    '''GPU burn benchmark.

    This benchmark runs continuously GEMM, either single or double precision,
    on a selected set of GPUs on the node where the benchmark runs.

    The floating point precision of the computations, the duration of the
    benchmark as well as the list of GPU devices that the benchmark will run
    on can be controlled through test variables.

    This benchmark tries to build the benchmark code through the
    :class:`gpu_burn_build` fixture.

    This benchmark sets the
    :attr:`~reframe.core.pipeline.RegressionTest.num_gpus_per_node` test
    attribute, if not already set, based on the number of devices with ``type
    == 'gpu'`` defined in the corresponding partition configuration.
    Similarly, this benchmark will use the ``arch`` device configuration
    attribute to set the :attr:`gpu_arch` variable, if this is not already set
    by the user.

    .. list-table:: Summary
       :widths: 10 10 20 20 20 20
       :header-rows: 1

       * - Variables
         - Parameters
         - Metrics
         - Fixtures
         - System features
         - Environment features
       * - - :attr:`use_dp`
           - :attr:`duration`
           - :attr:`devices`
         - *None*
         - - :obj:`gpu_perf_min`
           - :obj:`gpu_temp_max`
         - - :class:`gpu_burn_build` :obj:`[E]`
         - ``+gpu``
         - ``+cuda`` OR ``+hip``

    '''

    #: Use double-precision arithmetic when running the benchmark.
    #:
    #: :type: :class:`bool`
    #: :default: ``True``
    use_dp = variable(typ.Bool, value=True)

    #: Duration of the benchmark in seconds.
    #:
    #: :type: :class:`int`
    #: :default: ``10``
    duration = variable(int, value=10)

    #: List of device IDs to run the benchmark on.
    #:
    #: If empty, the benchmark will run on all the available devices.
    #:
    #: :type: :class:`List[int]`
    #: :default: ``[]``
    devices = variable(typ.List[int], value=[])

    num_tasks = 1
    num_tasks_per_node = 1

    descr = 'GPU burn test'
    build_system = 'Make'
    executable = 'gpu_burn.x'

    # The fixture to build the benchmark
    #
    # :type: :class:`gpu_burn_build`
    # :scope: *environment*
    gpu_burn_binaries = fixture(gpu_burn_build, scope='environment')

    valid_systems = ['+gpu']
    valid_prog_environs = ['+cuda', '+hip']

    @run_before('run')
    def set_exec_opts(self):
        if self.use_dp:
            self.executable_opts += ['-d']

        if self.devices:
            self.executable_opts += ['-D',
                                     ','.join(str(x) for x in self.devices)]

        self.executable_opts += [str(self.duration)]

    @run_before('run')
    def add_exec_prefix(self):
        self.executable = os.path.join(self.gpu_burn_binaries.stagedir,
                                       self.executable)

    @run_before('run')
    def set_num_gpus_per_node(self):
        if self.num_gpus_per_node is not None:
            return

        gpu_devices = self.current_partition.select_devices('gpu')
        if gpu_devices:
            self.num_gpus_per_node = gpu_devices[0].num_devices

    @sanity_function
    def assert_sanity(self):
        num_gpus_detected = sn.extractsingle(
            r'==> devices selected \((\d+)\)', self.stdout, 1, int
        )
        return sn.assert_eq(
            sn.count(sn.findall(r'GPU\s+\d+\(OK\)', self.stdout)),
            num_gpus_detected
        )

    def _extract_metric(self, metric):
        return sn.extractall(
            r'GPU\s+\d+\(OK\):\s+(?P<perf>\S+)\s+GF/s\s+'
            r'(?P<temp>\S+)\s+Celsius', self.stdout, metric, float
        )

    @performance_function('Gflop/s')
    def gpu_perf_min(self):
        '''Lowest performance recorded among all the selected devices.'''
        return sn.min(self._extract_metric('perf'))

    @performance_function('degC')
    def gpu_temp_max(self):
        '''Maximum temperature recorded among all the selected devices.'''
        return sn.max(self._extract_metric('temp'))
