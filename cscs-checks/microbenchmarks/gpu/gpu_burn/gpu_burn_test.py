# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn

from library.microbenchmarks.gpu.gpu_burn import GpuBurnBase
import cscslib.microbenchmarks.gpu.hooks as hooks

@rfm.simple_test
class GpuBurnTest(GpuBurnBase, hooks.SetCompileOpts, hooks.SetGPUsPerNode):
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
