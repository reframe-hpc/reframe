# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn

from hpctestlib.microbenchmarks.cpu.strided_bandwidth import StridedBandwidth


@rfm.simple_test
class strided_bandwidth_check(StridedBandwidth):
    '''Strided bandwidth check.

    This test is parameterized with the ``stride`` parameter, covering the
    following scenarios: 8-byte stride using the full cache line, 64-byte
    stride using 1/8 of the cacheline, and 128-byte using 1/8 of every 2nd
    cacheline.

    This test requires the ``num_cpus`` variable, which is set in a post-setup
    hook. The data for each supported system is stored in ``system_num_cpus``

    Since the performance references change with the ``stride`` parameter, the
    references for each test instace are stored in the ``reference_per_stride``
    variable. The actual references are then set in a pre-performance hook.
    '''

    # Define the stride parameter
    stride = parameter([1, 8, 16])

    valid_systems = ['daint:gpu', 'dom:gpu', 'daint:mc', 'dom:mc',
                     'eiger:mc', 'pilatus:mc']
    valid_prog_environs = ['PrgEnv-gnu']
    system_num_cpus = variable(
        dict, value={
            'daint:mc':  72,
            'daint:gpu': 24,
            'dom:mc':  72,
            'dom:gpu': 24,
            'eiger:mc': 128,
            'pilatus:mc': 128
        }
    )
    reference_per_stride = variable(
        dict, value={
            1: {
                'dom:gpu': {
                    'bandwidth': (50, -0.1, 0.1, 'GB/s')
                },
                'dom:mc': {
                    'bandwidth': (100, -0.1, 0.1, 'GB/s')
                },
                'daint:gpu': {
                    'bandwidth': (50, -0.1, 0.1, 'GB/s')
                },
                'daint:mc': {
                    'bandwidth': (100, -0.1, 0.1, 'GB/s')
                },
                'eiger:mc': {
                    'bandwidth': (270, -0.1, 0.1, 'GB/s')
                },
                'pilatus:mc': {
                    'bandwidth': (270, -0.1, 0.1, 'GB/s')
                }
            },
            8: {
                'dom:gpu': {
                    'bandwidth': (6, -0.1, 0.2, 'GB/s')
                },
                'dom:mc': {
                    'bandwidth': (12.5, -0.1, 0.2, 'GB/s')
                },
                'daint:gpu': {
                    'bandwidth': (6, -0.05, 0.2, 'GB/s')
                },
                'daint:mc': {
                    'bandwidth': (12.5, -0.1, 0.2, 'GB/s')
                },
                'eiger:mc': {
                    'bandwidth': (33, -0.1, 0.2, 'GB/s')
                },
                'pilatus:mc': {
                    'bandwidth': (33, -0.1, 0.2, 'GB/s')
                }
            },
            16: {
                'dom:gpu': {
                    'bandwidth': (4.5, -0.1, 0.2, 'GB/s')
                },
                'dom:mc': {
                    'bandwidth': (9.1, -0.1, 0.2, 'GB/s')
                },
                'daint:gpu': {
                    'bandwidth': (4.5, -0.1, 0.2, 'GB/s')
                },
                'daint:mc': {
                    'bandwidth': (9.1, -0.1, 0.2, 'GB/s')
                },
                'eiger:mc': {
                    'bandwidth': (33, -0.1, 0.2, 'GB/s')
                },
            }
        }
    )
    tags = {'benchmark', 'diagnostic'}

    @rfm.run_after('setup')
    def set_num_cpus(self):
        self.num_cpus = self.system_num_cpus[self.current_partition.fullname]

    @rfm.run_before('performance')
    def set_references(self):
        self.reference = self.reference_per_stride[self.stride]
