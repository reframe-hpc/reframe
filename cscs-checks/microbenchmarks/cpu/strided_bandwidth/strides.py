# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm

from hpctestlib.microbenchmarks.cpu.strided_bandwidth import StridedBandwidth

@rfm.simple_test
class strided_bandwidth_check(StridedBandwidth):
    '''Strided bandwidth check.

    This test is parameterized with the ``stride_bytes`` parameter, covering
    the following scenarios: 8-byte, 64-byte and 128-byte strides.

    This test requires the ``num_cpus`` variable, which is set in a post-setup
    hook.
    '''

    valid_systems = ['daint:gpu', 'dom:gpu', 'daint:mc', 'dom:mc',
                     'eiger:mc', 'pilatus:mc', 'ault:a64fx']
    valid_prog_environs = ['PrgEnv-gnu']

    # Define the stride parameter
    stride_bytes = parameter([8, 64, 128])

    # Set required variables
    num_tasks = 1
    tags = {'benchmark', 'diagnostic'}

    # Bandwidth references
    reference_bw = {
        8: {
            'haswell': (50, -0.1, 0.1, 'GB/s'),
            'broadwell': (100, -0.1, 0.1, 'GB/s'),
            'zen2': (270, -0.1, 0.1, 'GB/s'),
            'a64fx': (50, -0.1, 0.1, 'GB/s')
        },
        64: {
            'haswell': (6, -0.1, 0.2, 'GB/s'),
            'broadwell': (12.5, -0.1, 0.2, 'GB/s'),
            'zen2': (33, -0.1, 0.2, 'GB/s'),
            'a64fx': (45, -0.1, 0.1, 'GB/s')
        },
        128: {
            'haswell': (4.5, -0.1, 0.2, 'GB/s'),
            'broadwell': (9.1, -0.1, 0.2, 'GB/s'),
            'zen2': (33, -0.1, 0.2, 'GB/s'),
            'a64fx': (25, -0.1, 0.1, 'GB/s')
        },
    }

    @run_after('init')
    def set_valid_systems(self):
        cp = self.current_system.name
        if cp == 'ault':
            self.valid_prog_environs = ['PrgEnv-fujitsu']

    @run_after('setup')
    def skip_if_no_topo(self):
        proc = self.current_partition.processor
        pname = self.current_partition.fullname
        if not proc.info:
            self.skip(f'no topology information found for partition {pname!r}')

    @run_before('run')
    def set_exec_opts(self):
        proc = self.current_partition.processor
        self.executable_opts = [
            '100000000', str(self.stride_bytes // 8), f'{proc.num_cpus}'
        ]

    @run_before('performance')
    def set_reference(self):
        proc = self.current_partition.processor
        try:
            ref = self.reference_bw[self.stride_bytes][proc.arch]
        except KeyError:
            return
        else:
            self.reference = {'*': {'bandwidth': ref}}
