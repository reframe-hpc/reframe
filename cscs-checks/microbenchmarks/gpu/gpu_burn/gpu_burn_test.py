# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause


import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.osext as osext
from reframe.core.exceptions import SanityError

from hpctestlib.microbenchmarks.gpu.gpu_burn import gpu_burn_check
import cscstests.microbenchmarks.gpu.hooks as hooks


@rfm.simple_test
class cscs_gpu_burn_check(GpuBurn):
    use_doubles = True
    duration = 40
    valid_systems = [
        'daint:gpu', 'dom:gpu', 'arolla:cn', 'tsa:cn', 'ault:amdv100',
        'ault:intelv100', 'ault:amda100', 'ault:amdvega'
    ]
    valid_prog_environs = ['PrgEnv-gnu']
    exclusive_access = True
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

    maintainers = ['@vkarak']
    tags = {'diagnostic', 'benchmark', 'craype'}

    # Inject external hooks
    @run_after('setup')
    def set_gpu_arch(self):
        hooks.set_gpu_arch(self)

    @run_before('run')
    def set_num_gpus_per_node(self):
        hooks.set_num_gpus_per_node(self)
