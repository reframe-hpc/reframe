# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.typecheck as typ
import reframe.utility.sanity as sn


class gpu_burn_build(rfm.CompileOnlyRegressionTest, pin_prefix=True):
    # FIXME: We set a default value to the following variable due to:
    # https://github.com/eth-cscs/reframe/issues/2477

    #: Set the build option to either ``'cuda'`` or ``'hip'``.
    #:
    #: :default: ``required``
    gpu_build = variable(str, value='cuda')

    #: Set the GPU architecture.
    #:
    #: This variable will be passed to the compiler to generate the
    #: arch-specific code.
    #:
    #: :default: ``None``
    gpu_arch = variable(str, type(None), value=None)

    descr = 'GPU burn test build fixture'
    sourcesdir = 'src/gpu_burn'
    build_system = 'Make'

    @run_after('init')
    def setup_build(self):
        if self.gpu_build == 'cuda':
            self.build_system.makefile = 'makefile.cuda'
            if self.gpu_arch:
                self.build_system.cxxflags = [f'-arch=compute_{self.gpu_arch}',
                                              f'-code=sm_{self.gpu_arch}']
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
    '''Base class for the GPU Burn test.

    The test sources can be compiled for both CUDA and HIP. This is set with
    the ``gpu_build`` variable, which must be set by a derived class to either
    ``'cuda'`` or ``'hip'``. This source code can also be compiled for a
    specific device architecture by setting the ``gpu_arch`` variable to an
    AMD or NVIDIA supported architecture code. For the run stage, this test
    requires that derived classes set the variables, num_tasks and
    num_gpus_per_node.

    The duration of the run can be changed by passing the value (in seconds) of
    the desired run length. If this value is prepended with ``-d``, the matrix
    operations will take place using double precision. By default, the code
    will run for 10s in single precision mode.

    The performance stage of this test assesses the Gflops/s and the
    temperatures recorded for each device after the burn.
    '''

    use_dp = variable(typ.Bool, value=True)
    duration = variable(int, value=10)
    devices = variable(typ.List[int], value=[])

    num_tasks = 1
    num_tasks_per_node = 1

    descr = 'GPU burn test'
    build_system = 'Make'
    executable = 'gpu_burn.x'
    gpu_burn_binaries = fixture(gpu_burn_build, scope='environment')

    @run_after('init')
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
        '''Lowest performance recorded.'''
        return sn.min(self._extract_metric('perf'))

    @performance_function('degC')
    def gpu_temp_max(self, nid=None):
        '''Maximum temperature recorded.'''
        return sn.max(self._extract_metric('temp'))
