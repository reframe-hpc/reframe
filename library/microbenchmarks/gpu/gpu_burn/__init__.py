# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause


import reframe as rfm
import reframe.utility.sanity as sn


__all__ = ['GpuBurnBase']


class GpuBurnBase(rfm.RegressionTest, pin_prefix=True):
    '''Base class for the GPU Burn test.

    -- Compile --
    The tests sources can be compiled for both CUDA and HIP. This is set with
    the `gpu_build` variable, which must be set by a derived class. This
    source code can also be compiled for a specific device architecture by
    defining the `gpu_arch` variable.

    -- Run --
    The variables required by the run stage are:
     - executable_opts: Defines the options to be passed to the executable.
     - num_tasks: Number of tasks to use for this test.

    -- Sanity --
    Checks that the output matches the number of num_gpus_per_node*num_tasks.

    -- Performance --
    The performance patterns are:
     - perf: Gflops/s achieved on average for the duration of the run.
     - temp: the GPU's temperature in degC after the run.

    '''

    # GPU build variables
    gpu_build = variable(str)
    gpu_arch = variable(str, type(None), value=None)

    # Required variables
    executable_opts = required
    num_tasks = required

    descr = 'GPU burn test'
    build_system = 'Make'
    executable = './gpu_burn.x'
    num_tasks_per_node = 1

    @rfm.run_before('compile')
    def set_gpu_build(self):
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

    @property
    @sn.sanity_function
    def num_tasks_assigned(self):
        return self.job.num_tasks * self.num_gpus_per_node

    @rfm.run_before('sanity')
    def set_sanity_patterns(self):
        self.sanity_patterns = sn.assert_eq(sn.count(sn.findall(
            r'^\s*\[[^\]]*\]\s*GPU\s*\d+\(OK\)', self.stdout)
        ), self.num_tasks_assigned)

    @rfm.run_before('performance')
    def set_perf_patterns(self):
        patt = (r'^\s*\[[^\]]*\]\s*GPU\s+\d+\(\S*\):\s+(?P<perf>\S*)\s+GF\/s'
                r'\s+(?P<temp>\S*)\s+Celsius')
        self.perf_patterns = {
            'perf': sn.min(sn.extractall(patt, self.stdout, 'perf', float)),
            'temp': sn.max(sn.extractall(patt, self.stdout, 'temp', float)),
        }
