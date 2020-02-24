# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


class StridedBase(rfm.RegressionTest):
    def __init__(self):
        self.sourcepath = 'strides.cpp'
        self.build_system = 'SingleSource'
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'daint:mc', 'dom:mc']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.num_tasks = 1
        self.num_tasks_per_node = 1

        self.sanity_patterns = sn.assert_eq(
            sn.count(sn.findall(r'bandwidth', self.stdout)),
            self.num_tasks_assigned)

        self.perf_patterns = {
            'bandwidth': sn.extractsingle(
                r'bandwidth: (?P<bw>\S+) GB/s',
                self.stdout, 'bw', float)
        }

        self.system_num_cpus = {
            'daint:mc':  72,
            'daint:gpu': 24,
            'dom:mc':  72,
            'dom:gpu': 24,
        }

        self.maintainers = ['SK']
        self.tags = {'benchmark', 'diagnostic'}

    @property
    @sn.sanity_function
    def num_tasks_assigned(self):
        return self.job.num_tasks


@rfm.required_version('>=2.16-dev0')
@rfm.simple_test
class StridedBandwidthTest(StridedBase):
    def __init__(self):
        super().__init__()

        self.reference = {
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
            }
        }

    @rfm.run_before('run')
    def set_exec_opts(self):
        self.num_cpus = self.system_num_cpus[self.current_partition.fullname]

        # 8-byte stride, using the full cacheline
        self.executable_opts = ['100000000', '1', '%s' % self.num_cpus]


@rfm.required_version('>=2.16-dev0')
@rfm.simple_test
class StridedBandwidthTest64(StridedBase):
    def __init__(self):
        super().__init__()

        self.reference = {
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
            }
        }

    @rfm.run_before('run')
    def set_exec_opts(self):
        self.num_cpus = self.system_num_cpus[self.current_partition.fullname]

        # 64-byte stride, using 1/8 of the cacheline
        self.executable_opts = ['100000000', '8', '%s' % self.num_cpus]


@rfm.required_version('>=2.16-dev0')
@rfm.simple_test
class StridedBandwidthTest128(StridedBase):
    def __init__(self):
        super().__init__()

        self.reference = {
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
            }
        }

    @rfm.run_before('run')
    def set_exec_opts(self):
        self.num_cpus = self.system_num_cpus[self.current_partition.fullname]

        # 128-byte stride, using 1/8 of every 2nd cacheline
        self.executable_opts = ['100000000', '16', '%s' % self.num_cpus]
