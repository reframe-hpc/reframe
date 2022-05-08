# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm   # noqa: F501
from hpctestlib.microbenchmarks.gpu.gpu_burn import gpu_burn_check


@rfm.simple_test
class cscs_gpu_burn_check(gpu_burn_check):
    use_dp = True
    duration = 40
    exclusive_access = True
    reference = {
        'dom:gpu': {
            'gpu_perf_min': (4115, -0.10, None, 'Gflop/s'),
        },
        'daint:gpu': {
            'gpu_perf_min': (4115, -0.10, None, 'Gflop/s'),
        },
        'arolla:cn': {
            'gpu_perf_min': (5861, -0.10, None, 'Gflop/s'),
        },
        'tsa:cn': {
            'gpu_perf_min': (5861, -0.10, None, 'Gflop/s'),
        },
        'ault:amda100': {
            'gpu_perf_min': (15000, -0.10, None, 'Gflop/s'),
        },
        'ault:amdv100': {
            'gpu_perf_min': (5500, -0.10, None, 'Gflop/s'),
        },
        'ault:intelv100': {
            'gpu_perf_min': (5500, -0.10, None, 'Gflop/s'),
        },
        'ault:amdvega': {
            'gpu_perf_min': (3450, -0.10, None, 'Gflop/s'),
        },
    }

    maintainers = ['@vkarak']
    tags = {'diagnostic', 'benchmark', 'craype'}
