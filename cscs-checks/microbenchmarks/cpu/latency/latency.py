# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn

from hpctestlib.microbenchmarks.cpu.latency import CpuLatency


@rfm.simple_test
class cpu_latency_check(CpuLatency):
    buffer_sizes = ['16000', '128000', '8000000', '500000000']
    num_tasks = 0
    valid_systems = [
        'daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
        'ault:intel', 'ault:amdvega', 'tave:compute', 'ault:a64fx'
    ]
    valid_prog_environs = ['PrgEnv-gnu']
    tags = {'benchmark', 'diagnostic'}

    @rfm.run_after('init')
    def set_valid_environs(self):
        if self.current_system.name in {'ault'}:
            self.valid_prog_environs = ['PrgEnv-fujitsu']

    @rfm.run_after('setup')
    def set_modules(self):
        if self.current_system.name in {'daint', 'dom'}:
            self.modules = ['craype-hugepages1G']
        if self.current_system.name in {'tave'}:
            self.modules = ['craype-hugepages512M']

    @rfm.run_before('performance')
    def set_references(self):
        self.reference = {
            'dom:mc': {
                'latencyL1': (1.21, -0.01, 0.26, 'ns'),
                'latencyL2': (3.65, -0.01, 0.26, 'ns'),
                'latencyL3': (18.83, -0.01, 0.05, 'ns'),
                'latencyL4': (76.6, -0.01, 0.05, 'ns')
            },
            'dom:gpu': {
                'latencyL1': (1.14, -0.01, 0.26, 'ns'),
                'latencyL2': (3.44, -0.01, 0.26, 'ns'),
                'latencyL3': (15.65, -0.01, 0.05, 'ns'),
                'latencyL4': (71.7, -0.01, 0.05, 'ns')
            },
            'daint:mc': {
                'latencyL1': (1.21, -0.01, 0.26, 'ns'),
                'latencyL2': (3.65, -0.01, 0.26, 'ns'),
                'latencyL3': (18.83, -0.01, 0.05, 'ns'),
                'latencyL4': (76.6, -0.01, 0.05, 'ns')
            },
            'daint:gpu': {
                'latencyL1': (1.14, -0.01, 0.26, 'ns'),
                'latencyL2': (3.44, -0.01, 0.26, 'ns'),
                'latencyL3': (15.65, -0.01, 0.05, 'ns'),
                'latencyL4': (71.7, -0.01, 0.05, 'ns')
            },
            'ault:intel': {
                'latencyL1': (1.08, -0.01, 0.26, 'ns'),
                'latencyL2': (3.8, -0.01, 0.26, 'ns'),
                'latencyL3': (21.5, -0.01, 0.05, 'ns'),
                'latencyL4': (86.5, -0.01, 0.05, 'ns')
            },
            'ault:amdvega': {
                'latencyL1': (1.32, -0.01, 0.26, 'ns'),
                'latencyL2': (4.02, -0.01, 0.26, 'ns'),
                'latencyL3': (14.4, -0.01, 0.26, 'ns'),
                'latencyL4': (90.0, -0.01, 0.05, 'ns')
            },
            'tave:compute': {
                'latencyL1': (2.86, -0.01, 0.05, 'ns'),
                'latencyL2': (12.15, -0.01, 0.05, 'ns'),
                'latencyL3': (137, -0.01, 0.05, 'ns'),
                'latencyL4': (150, -0.05, 0.05, 'ns')
            },
            'ault:a64fx': {
                'latencyL1': (2.78, None, 0.05, 'ns'),
                'latencyL2': (14.3, None, 0.05, 'ns'),
                'latencyL3': (32.1, None, 0.05, 'ns'),
                'latencyL4': (146,  None, 0.05, 'ns')
            },
        }
