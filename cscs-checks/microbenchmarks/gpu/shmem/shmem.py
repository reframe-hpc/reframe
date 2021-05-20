# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn

from hpctestlib.microbenchmarks.gpu.shmem import GpuShmem
import cscstests.microbenchmarks.gpu.hooks as hooks


@rfm.simple_test
class gpu_shmem_check(GpuShmem):
    valid_systems = ['daint:gpu', 'dom:gpu',
                     'ault:amdv100', 'ault:intelv100',
                     'ault:amda100', 'ault:amdvega']
    valid_prog_environs = ['PrgEnv-gnu']
    num_tasks = 0
    num_tasks_per_node = 1
    build_system = 'Make'
    executable = 'shmem.x'
    reference = {
        # theoretical limit for P100:
        # 8 [B/cycle] * 1.328 [GHz] * 16 [bankwidth] * 56 [SM] = 9520 GB/s
        'dom:gpu': {
            'bandwidth': (8850, -0.01, 9520/8850 - 1, 'GB/s')
        },
        'daint:gpu': {
            'bandwidth': (8850, -0.01, 9520/8850 - 1, 'GB/s')
        },
        'ault:amdv100': {
            'bandwidth': (13020, -0.01, None, 'GB/s')
        },
        'ault:intelv100': {
            'bandwidth': (13020, -0.01, None, 'GB/s')
        },
        'ault:amda100': {
            'bandwidth': (18139, -0.01, None, 'GB/s')
        },
        'ault:amdvega': {
            'bandwidth': (9060, -0.01, None, 'GB/s')
        }
    }
    maintainers = ['SK', 'JO']
    tags = {'benchmark', 'diagnostic', 'craype'}

    # Inject external hooks
    @rfm.run_after('setup')
    def set_gpu_arch(self):
        hooks.set_gpu_arch(self)

    @rfm.run_before('run')
    def set_num_gpus_per_node(self):
        hooks.set_num_gpus_per_node(self)
