# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class CPULatencyTest(rfm.RegressionTest):
    sourcepath = 'latency.cpp'
    build_system = 'SingleSource'
    valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
                     'ault:intel', 'ault:amdvega', 'eiger:mc', 'pilatus:mc']
    valid_prog_environs = ['PrgEnv-gnu']
    num_tasks = 0
    num_tasks_per_node = 1
    executable_opts = ['16000', '128000', '8000000', '500000000']
    reference = {
        'dom:mc': {
            'latencyL1':  (1.21, -0.01, 0.26, 'ns'),
            'latencyL2':  (3.65, -0.01, 0.26, 'ns'),
            'latencyL3':  (18.83, -0.01, 0.05, 'ns'),
            'latencyMem': (76.6, -0.01, 0.05, 'ns')
        },
        'dom:gpu': {
            'latencyL1':  (1.14, -0.01, 0.26, 'ns'),
            'latencyL2':  (3.44, -0.01, 0.26, 'ns'),
            'latencyL3':  (15.65, -0.01, 0.05, 'ns'),
            'latencyMem': (71.7, -0.01, 0.05, 'ns')
        },
        'daint:mc': {
            'latencyL1':  (1.21, -0.01, 0.26, 'ns'),
            'latencyL2':  (3.65, -0.01, 0.26, 'ns'),
            'latencyL3':  (18.83, -0.01, 0.05, 'ns'),
            'latencyMem': (76.6, -0.01, 0.05, 'ns')
        },
        'daint:gpu': {
            'latencyL1':  (1.14, -0.01, 0.26, 'ns'),
            'latencyL2':  (3.44, -0.01, 0.26, 'ns'),
            'latencyL3':  (15.65, -0.01, 0.05, 'ns'),
            'latencyMem': (71.7, -0.01, 0.05, 'ns')
        },
        'ault:intel': {
            'latencyL1':  (1.08, -0.01, 0.26, 'ns'),
            'latencyL2':  (3.8, -0.01, 0.26, 'ns'),
            'latencyL3':  (21.5, -0.01, 0.05, 'ns'),
            'latencyMem': (86.5, -0.01, 0.05, 'ns')
        },
        'ault:amdvega': {
            'latencyL1':  (1.32, -0.01, 0.26, 'ns'),
            'latencyL2':  (4.02, -0.01, 0.26, 'ns'),
            'latencyL3':  (14.4, -0.01, 0.26, 'ns'),
            'latencyMem': (90.0, -0.01, 0.05, 'ns')
        },
        'eiger:mc': {
            'latencyL1':  (1.19, -0.02, 0.05, 'ns'),
            'latencyL2':  (3.40, -0.03, 0.05, 'ns'),
            'latencyL3':  (11.5, -0.05, 0.05, 'ns'),
            'latencyMem': (105, -0.05, 0.05, 'ns')
        },
        'pilatus:mc': {
            'latencyL1':  (1.19, -0.02, 0.05, 'ns'),
            'latencyL2':  (3.40, -0.03, 0.05, 'ns'),
            'latencyL3':  (11.5, -0.05, 0.05, 'ns'),
            'latencyMem': (105, -0.05, 0.05, 'ns')
        },
    }
    maintainers = ['SK']
    tags = {'benchmark', 'diagnostic'}

    @run_after('init')
    def set_modules(self):
        if self.current_system.name in {'daint', 'dom'}:
            self.modules = ['craype-hugepages1G']

    @run_before('compile')
    def set_flags(self):
        self.build_system.cxxflags = ['-O3']

    @sanity_function
    def assert_success(self):
        return sn.assert_eq(
            sn.count(sn.findall(r'latency', self.stdout)),
            self.num_tasks * len(self.executable_opts)
        )

    def lat_pattern(self, index):
        return sn.extractsingle(
            r'latency \(ns\) for input size %s: (?P<bw>\S+) clocks' %
            self.executable_opts[index], self.stdout, 'bw', float)

    @performance_function('ns')
    def latencyL1(self):
        return self.lat_pattern(0)

    @performance_function('ns')
    def latencyL2(self):
        return self.lat_pattern(1)

    @performance_function('ns')
    def latencyL3(self):
        return self.lat_pattern(2)

    @performance_function('ns')
    def latencyMem(self):
        return self.lat_pattern(3)
