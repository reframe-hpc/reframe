# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn

from library.microbenchmarks.gpu.gpu_burn import GPU_burn
import cscslib.microbenchmarks.gpu.hooks as hooks

@rfm.simple_test
class GPU_burn_check(GPU_burn):
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
            'perf': (4115, -0.10, None, 'Gflop/s'),
        },
        'daint:gpu': {
            'perf': (4115, -0.10, None, 'Gflop/s'),
        },
        'arolla:cn': {
            'perf': (5861, -0.10, None, 'Gflop/s'),
        },
        'tsa:cn': {
            'perf': (5861, -0.10, None, 'Gflop/s'),
        },
        'ault:amda100': {
            'perf': (15000, -0.10, None, 'Gflop/s'),
        },
        'ault:amdv100': {
            'perf': (5500, -0.10, None, 'Gflop/s'),
        },
        'ault:intelv100': {
            'perf': (5500, -0.10, None, 'Gflop/s'),
        },
        'ault:amdvega': {
            'perf': (3450, -0.10, None, 'Gflop/s'),
        },
        '*': {'temp': (0, None, None, 'degC')}
    }

    maintainers = ['AJ', 'TM']
    tags = {'diagnostic', 'benchmark', 'craype'}

    # Inject external hooks
    @rfm.run_after('setup')
    def set_gpu_arch(self):
        hooks.set_gpu_arch(self)

    @rfm.run_before('run')
    def set_gpus_per_node(self):
        hooks.set_gpus_per_node(self)

    @rfm.run_before('performance')
    def report_nid_with_smallest_flops(self):
        regex = r'\[(\S+)\] GPU\s+\d\(OK\): (\d+) GF/s'
        rptf = os.path.join(self.stagedir, sn.evaluate(self.stdout))
        self.nids = sn.extractall(regex, rptf, 1)
        self.flops = sn.extractall(regex, rptf, 2, float)

        # Find index of smallest flops and update reference dictionary to
        # include our patched units
        index = self.flops.evaluate().index(min(self.flops))
        unit = f'GF/s ({self.nids[index]})'
        for key, ref in self.reference.items():
            if not key.endswith(':temp'):
                self.reference[key] = (*ref[:3], unit)
