# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause


import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.osext as osext
from reframe.core.exceptions import SanityError

from hpctestlib.microbenchmarks.gpu.gpu_burn import GpuBurn
import cscstests.microbenchmarks.gpu.hooks as hooks


@rfm.simple_test
class gpu_burn_check(GpuBurn):
    valid_systems = [
        'daint:gpu', 'dom:gpu', 'arolla:cn', 'tsa:cn', 'ault:amdv100',
        'ault:intelv100', 'ault:amda100', 'ault:amdvega'
    ]
    valid_prog_environs = ['PrgEnv-gnu']
    exclusive_access = True
    executable_opts = ['-d', '40']
    num_tasks = 0
    reference = {
        'dom:gpu': {
            'min_perf': (4115, -0.10, None, 'Gflop/s'),
        },
        'daint:gpu': {
            'min_perf': (4115, -0.10, None, 'Gflop/s'),
        },
        'arolla:cn': {
            'min_perf': (5861, -0.10, None, 'Gflop/s'),
        },
        'tsa:cn': {
            'min_perf': (5861, -0.10, None, 'Gflop/s'),
        },
        'ault:amda100': {
            'min_perf': (15000, -0.10, None, 'Gflop/s'),
        },
        'ault:amdv100': {
            'min_perf': (5500, -0.10, None, 'Gflop/s'),
        },
        'ault:intelv100': {
            'min_perf': (5500, -0.10, None, 'Gflop/s'),
        },
        'ault:amdvega': {
            'min_perf': (3450, -0.10, None, 'Gflop/s'),
        },
    }

    maintainers = ['AJ', 'TM']
    tags = {'diagnostic', 'benchmark', 'craype'}

    # Inject external hooks
    @run_after('setup')
    def set_gpu_arch(self):
        hooks.set_gpu_arch(self)

    @run_before('run')
    def set_num_gpus_per_node(self):
        hooks.set_num_gpus_per_node(self)

    @run_before('performance')
    def report_slow_nodes(self):
        '''Report the base perf metrics and also all the slow nodes.'''

        # Only report the nodes that don't meet the perf reference
        with osext.change_dir(self.stagedir):
            key = f'{self.current_partition.fullname}:min_perf'
            if key in self.reference:
                regex = r'\[(\S+)\] GPU\s+\d\(OK\): (\d+) GF/s'
                nids = set(sn.extractall(regex, self.stdout, 1))

                # Get the references
                ref, lt, ut, *_ = self.reference[key]

                # Flag the slow nodes
                for nid in nids:
                    try:
                        node_perf = self.min_perf(nid)
                        val = node_perf.evaluate(cache=True)
                        sn.assert_reference(val, ref, lt, ut).evaluate()
                    except SanityError:
                        self.perf_variables[nid] = node_perf

