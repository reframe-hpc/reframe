# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause


import reframe as rfm
import reframe.utility.sanity as sn


__all__ = ['GpuBurn']


class GpuBurn(rfm.RegressionTest, pin_prefix=True):
    '''Base class for the GPU Burn test.

    -- Compile --
    The test sources can be compiled for both CUDA and HIP. This is set with
    the `gpu_build` variable, which must be set by a derived class to either
    'cuda' or 'hip'. This source code can also be compiled for a specific
    device architecture by setting the `gpu_arch` variable to an AMD or NVIDIA
    supported architecture code.

    -- Run --
    The variables required by the run stage are:
     - num_tasks: Number of tasks to use for this test.
     - num_gpus_per_node: Number of GPUs per node.

    The duration of the run can be changed by passing the value (in seconds) of
    the desired run length. If this value is prepended with `-d`, the matrix
    operations will take place using double precision. By default, the code
    will run for 10s in single precision mode.

    -- Sanity --
    Checks that the output matches the number of num_gpus_per_node*num_tasks.

    -- Performance --
    The performance patterns are:
     - perf: minimum Gflops/s achieved amongst all GPUs.
     - temp: maximum GPU temperature in degC after the run amongst all GPUs.

    '''

    #: Set the build option to either 'cuda' or 'hip'.
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

    descr = 'GPU burn test'
    build_system = 'Make'
    executable = './gpu_burn.x'
    num_tasks_per_node = 1
    reference = {
        '*': {
            'perf': (0, None, None, 'Gflop/s'),
            'temp': (0, None, None, 'degC')
        }
    }

    @rfm.run_before('compile')
    def set_gpu_build(self):
        '''Set the build options before the compile pipeline stage.

        This hook requires the `gpu_build` variable to be set.
        The supported options are 'cuda' and 'hip'. See the vendor-specific
        docs for the supported options for the ``gpu_arch`` variable.
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

    @property
    @sn.sanity_function
    def num_tasks_assigned(self):
        '''Total number of times the gpu burn will run.

        The GPU burn app is multi-threaded and will run in all the gpus present
        in the node.
        '''

        return self.job.num_tasks * self.num_gpus_per_node

    @rfm.run_before('sanity')
    def set_sanity_patterns(self):
        '''Set the sanity patterns to count the number of successful burns.'''

        self.sanity_patterns = sn.assert_eq(sn.count(sn.findall(
            r'^\s*\[[^\]]*\]\s*GPU\s*\d+\(OK\)', self.stdout)
        ), self.num_tasks_assigned)

    @rfm.run_before('performance')
    def set_perf_patterns(self):
        '''Extract the minimum performance and maximum temperature recorded.'''

        patt = (r'^\s*\[[^\]]*\]\s*GPU\s+\d+\(\S*\):\s+(?P<perf>\S*)\s+GF\/s'
                r'\s+(?P<temp>\S*)\s+Celsius')
        self.perf_patterns = {
            'perf': sn.min(sn.extractall(patt, self.stdout, 'perf', float)),
            'temp': sn.max(sn.extractall(patt, self.stdout, 'temp', float)),
        }
