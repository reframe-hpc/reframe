# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class StridedBandwidthTest(rfm.RegressionTest):
    sourcepath = 'strides.cpp'
    build_system = 'SingleSource'
    valid_systems = ['daint:gpu', 'dom:gpu', 'daint:mc', 'dom:mc',
                     'eiger:mc', 'pilatus:mc']
    valid_prog_environs = ['PrgEnv-gnu']
    num_tasks = 1
    num_tasks_per_node = 1
    maintainers = ['SK']
    tags = {'benchmark', 'diagnostic'}
    stride_bytes = parameter([8, 64, 128])
    reference_bw = {
        8: {
            'haswell': (50, -0.1, 0.1, 'GB/s'),
            'broadwell': (100, -0.1, 0.1, 'GB/s'),
            'zen2': (270, -0.1, 0.1, 'GB/s')
        },
        64: {
            'haswell': (6, -0.1, 0.2, 'GB/s'),
            'broadwell': (12.5, -0.1, 0.2, 'GB/s'),
            'zen2': (33, -0.1, 0.2, 'GB/s')
        },
        128: {
            'haswell': (4.5, -0.1, 0.2, 'GB/s'),
            'broadwell': (9.1, -0.1, 0.2, 'GB/s'),
            'zen2': (33, -0.1, 0.2, 'GB/s')
        },
    }

    @run_after('setup')
    def skip_if_no_topo(self):
        proc = self.current_partition.processor
        pname = self.current_partition.fullname
        if not proc.info:
            self.skip(f'no topology information found for partition {pname!r}')

    @sanity_function
    def assert_num_tasks(self):
        return sn.assert_eq(sn.count(sn.findall(r'bandwidth', self.stdout)),
                            self.num_tasks)

    @performance_function('GB/s')
    def bandwidth(self):
        return sn.extractsingle(r'bandwidth: (?P<bw>\S+) GB/s',
                                self.stdout, 'bw', float)

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
