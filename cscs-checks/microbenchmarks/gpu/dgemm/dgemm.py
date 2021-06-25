# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm

from hpctestlib.microbenchmarks.gpu.dgemm import DgemmGpu
import cscstests.microbenchmarks.gpu.hooks as hooks


@rfm.simple_test
class dgemm_gpu_check(DgemmGpu):
    valid_systems = ['daint:gpu', 'dom:gpu',
                     'ault:amdv100', 'ault:intelv100',
                     'ault:amda100', 'ault:amdvega']
    valid_prog_environs = ['PrgEnv-gnu']
    num_tasks = 0
    reference = {
        'dom:gpu': {
            'perf': (3.35, -0.1, None, 'TF/s')
        },
        'daint:gpu': {
            'perf': (3.35, -0.1, None, 'TF/s')
        },
        'ault:amdv100': {
            'perf': (5.25, -0.1, None, 'TF/s')
        },
        'ault:intelv100': {
            'perf': (5.25, -0.1, None, 'TF/s')
        },
        'ault:amda100': {
            'perf': (10.5, -0.1, None, 'TF/s')
        },
        'ault:amdvega': {
            'perf': (3.45, -0.1, None, 'TF/s')
        }
    }
    maintainers = ['JO', 'SK']
    tags = {'benchmark', 'health'}

    # Inject external hooks
    @run_after('setup')
    def set_gpu_arch(self):
        hooks.set_gpu_arch(self)

    @run_before('run')
    def set_num_gpus_per_node(self):
        hooks.set_num_gpus_per_node(self)
