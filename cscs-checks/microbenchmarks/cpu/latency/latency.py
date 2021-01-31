# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.16-dev0')
@rfm.simple_test
class CPULatencyTest(rfm.RegressionTest):
    def __init__(self):
        self.sourcepath = 'latency.cpp'
        self.build_system = 'SingleSource'
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
                              'ault:intel', 'ault:amdvega', 'tave:compute']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.num_tasks = 0
        self.num_tasks_per_node = 1

        self.build_system.cxxflags = ['-O3']

        self.executable_opts = ['16000', '128000', '8000000', '500000000']

        if self.current_system.name in {'daint', 'dom'}:
            self.modules = ['craype-hugepages1G']
        if self.current_system.name in {'tave'}:
            self.modules = ['craype-hugepages512M']

        self.sanity_patterns = sn.assert_eq(
            sn.count(sn.findall(r'latency', self.stdout)),
            self.num_tasks_assigned * len(self.executable_opts))

        def lat_pattern(index):
            return sn.extractsingle(
                r'latency \(ns\) for input size %s: (?P<bw>\S+) clocks' %
                self.executable_opts[index], self.stdout, 'bw', float)

        self.perf_patterns = {
            'latencyL1': lat_pattern(0),
            'latencyL2': lat_pattern(1),
            'latencyL3': lat_pattern(2),
            'latencyMem': lat_pattern(3),
        }

        self.reference = {
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
            'tave:compute': {
                'latencyL1':  (2.86, -0.01, 0.05, 'ns'),
                'latencyL2':  (12.15, -0.01, 0.05, 'ns'),
                'latencyL3':  (137, -0.01, 0.05, 'ns'),
                'latencyMem': (150, -0.05, 0.05, 'ns')
            },
        }

        self.maintainers = ['SK']
        self.tags = {'benchmark', 'diagnostic'}

    @property
    @sn.sanity_function
    def num_tasks_assigned(self):
        return self.job.num_tasks
