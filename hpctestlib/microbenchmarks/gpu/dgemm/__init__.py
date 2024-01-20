# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


__all__ = ['DgemmGpu']


class DgemmGpu(rfm.RegressionTest, pin_prefix=True):
    '''Base class for the DGEMM GPU test.

    The test sources can be compiled for both CUDA and HIP. This is set with
    the ``gpu_build`` variable, which must be set by a derived class to either
    ``'cuda'`` or ``'hip'``. This source code can also be compiled for a
    specific device architecture by setting the ``gpu_arch`` variable to an
    AMD or NVIDIA supported architecture code. For the run stage, this test
    requires that derived classes set the variables, num_tasks and
    num_gpus_per_node.

    The performance stage of this test assesses the Tflops/s.
    '''

    #: Set the build option to either ``'cuda'`` or ``'hip'``.
    #:
    #: :default: ``required``
    gpu_build = variable(str)

    #: Set the GPU architecture.
    #: This variable will be passed to the compiler to generate the
    #: arch-specific code.
    #:
    #: :default: ``None``
    gpu_arch = variable(str, type(None), value=None)

    # Required variables
    num_tasks = required
    num_gpus_per_node = required

    descr = 'DGEMM GPU test'
    num_tasks_per_node = 1
    build_system = 'Make'
    executable = 'dgemm.x'
    reference = {
        '*': {
            'perf': (None, None, None, 'TF/s')
        }
    }

    @run_before('compile')
    def set_gpu_build(self):
        '''Set the build options [pre-compile hook].

        This hook requires the ``gpu_build`` variable to be set.
        The supported options are ``'cuda'`` and ``'hip'``. See the
        vendor-specific docs for the supported options for the ``gpu_arch``
        variable.
        '''

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
            raise ValueError('unknown gpu_build option')

    @sanity_function
    def assert_num_gpus(self):
        '''Assert that that all tasks passed.'''

        return sn.assert_eq(
            sn.count(sn.findall(r'^\s*\[[^\]]*\]\s*Test passed', self.stdout)),
            sn.getattr(self.job, 'num_tasks'))

    @run_before('performance')
    def set_perf_patterns(self):
        '''Extract the TF/s achieved.'''

        self.perf_patterns = {
            'perf': sn.min(sn.extractall(
                r'^\s*\[[^\]]*\]\s*GPU\s*\d+: (?P<fp>\S+) TF/s',
                self.stdout, 'fp', float))
        }
