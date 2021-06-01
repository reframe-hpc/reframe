# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


__all__ = ['StridedBandwidth']


class StridedBandwidth(rfm.RegressionTest, pin_prefix=True):
    '''Strided bandwith benchmark.

    The executable takes three required arguments. These are the buffer size
    (in bytes), the stride (in multiples of 8 bytes) and the number of threads
    to run this application with.

    Derived tests must set the parameter ``stride``, and the variables
    ``num_cpus`` and ``num_tasks``.

    The performance stage measures the bandiwdth in GB/s.
    '''

    #: Parameter that controls the stride access pattern.
    #: This parameter must be opverridden by the derived class.
    #:
    #: :default: ``()``
    stride = parameter()

    #: Set the number of cpus per node.
    #:
    #: :default: ``required``
    num_cpus = variable(int)

    # Required variables
    num_tasks = required

    sourcepath = 'strides.cpp'
    build_system = 'SingleSource'
    num_tasks_per_node = 1
    reference = {
        '*': {
            'bandwidth': (None, None, None, 'GB/s')
        }
    }
    maintainers = ['SK']

    @rfm.run_before('run')
    def set_exec_opts(self):
        '''Set the exec options.

        In order, these are the buffer size, stride and number of threads. See
        the main docstring above for more info.
        '''
        self.executable_opts = [
            '100000000', f'{self.stride}', f'{self.num_cpus}'
        ]

    @rfm.run_before('sanity')
    def set_sanity_patterns(self):
        ''' Assert that the bandwidth is reported for all the tasks.'''

        self.sanity_patterns = sn.assert_eq(
            sn.count(sn.findall(r'bandwidth:', self.stdout)),
            self.job.num_tasks
        )

    @rfm.run_before('performance')
    def set_perf_patterns(self):
        '''Extract the min bandwidth as a performance metric.'''

        self.perf_patterns = {
            'bandwidth': sn.min(
                sn.extractall(
                    r'bandwidth: (?P<bw>\S+) GB/s',
                    self.stdout, 'bw', float
                )
            )
        }
