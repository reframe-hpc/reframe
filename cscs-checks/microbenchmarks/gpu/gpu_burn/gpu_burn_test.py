# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class GpuBurnTest(rfm.RegressionTest):
    def __init__(self):
        self.valid_systems = ['daint:gpu', 'dom:gpu',
                              'arolla:cn', 'tsa:cn',
                              'ault:amdv100', 'ault:intelv100',
                              'ault:amda100', 'ault:amdvega']
        self.descr = 'GPU burn test'
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.exclusive_access = True
        self.executable_opts = ['-d', '40']
        self.build_system = 'Make'
        self.executable = './gpu_burn.x'

        # FIXME workaround due to issue #1639.
        self.readonly_files = ['Xdevice']

        self.num_tasks = 0
        self.num_tasks_per_node = 1
        self.sanity_patterns = self.assert_num_tasks()
        patt = (r'^\s*\[[^\]]*\]\s*GPU\s+\d+\(\S*\): (?P<perf>\S*) GF\/s  '
                r'(?P<temp>\S*) Celsius')
        self.perf_patterns = {
            'perf': sn.min(sn.extractall(patt, self.stdout, 'perf', float)),
        }

        self.reference = {
            'dom:gpu': {
                # 'perf': (4115, -0.10, None, 'Gflop/s'),
                'max_temp': (0, None, None, 'Celsius')
            },
            'daint:gpu': {
                # 'perf': (4115, -0.10, None, 'Gflop/s'),
                'max_temp': (0, None, None, 'Celsius')
            },
            'arolla:cn': {
                # 'perf': (5861, -0.10, None, 'Gflop/s'),
                'max_temp': (0, None, None, 'Celsius')
            },
            'tsa:cn': {
                # 'perf': (5861, -0.10, None, 'Gflop/s'),
                'max_temp': (0, None, None, 'Celsius')
            },
            'ault:amda100': {
                # 'perf': (15000, -0.10, None, 'Gflop/s'),
                'max_temp': (0, None, None, 'Celsius')
            },
            'ault:amdv100': {
                # 'perf': (5500, -0.10, None, 'Gflop/s'),
                'max_temp': (0, None, None, 'Celsius')
            },
            'ault:intelv100': {
                # 'perf': (5500, -0.10, None, 'Gflop/s'),
                'max_temp': (0, None, None, 'Celsius')
            },
            'ault:amdvega': {
                # 'perf': (3450, -0.10, None, 'Gflop/s'),
                'max_temp': (0, None, None, 'Celsius')
            },
        }

        self.maintainers = ['AJ', 'TM']
        self.tags = {'diagnostic', 'benchmark', 'craype'}

    @property
    @sn.sanity_function
    def num_tasks_assigned(self):
        return self.job.num_tasks * self.num_gpus_per_node

    @sn.sanity_function
    def assert_num_tasks(self):
        return sn.assert_eq(sn.count(sn.findall(
            r'^\s*\[[^\]]*\]\s*GPU\s*\d+\s*\(OK\)', self.stdout)
        ), self.num_tasks_assigned)

    @rfm.run_before('compile')
    def set_gpu_arch(self):
        cs = self.current_system.name
        cp = self.current_partition.fullname
        gpu_arch = None

        # Nvidia options
        if cs in {'dom', 'daint'}:
            gpu_arch = '60'
            self.modules = ['craype-accel-nvidia60']
        elif cs in {'arola', 'tsa'}:
            gpu_arch = '70'
            self.modules = ['cuda/10.1.243']
        elif cs in {'ault'}:
            self.modules = ['cuda']
            if cp in {'ault:amdv100', 'ault:intelv100'}:
                gpu_arch = '70'
            elif cp in {'ault:amda100'}:
                gpu_arch = '80'

        if gpu_arch:
            self.build_system.cxxflags = [f'-arch=compute_{gpu_arch}',
                                          f'-code=sm_{gpu_arch}']
            self.build_system.makefile = 'makefile.cuda'
            return

        # AMD options
        if cp in {'ault:amdvega'}:
            self.modules = ['rocm']
            gpu_arch = 'gfx906'

        if gpu_arch:
            self.build_system.cxxflags = [f'--amdgpu-target={gpu_arch}']
            self.build_system.makefile = 'makefile.hip'

    @rfm.run_before('run')
    def set_gpus_per_node(self):
        cs = self.current_system.name
        cp = self.current_partition.fullname
        if cs in {'dom', 'daint'}:
            self.num_gpus_per_node = 1
        elif cs in {'arola', 'tsa'}:
            self.num_gpus_per_node = 8
        elif cp in {'ault:amda100', 'ault:intelv100'}:
            self.num_gpus_per_node = 4
        elif cp in {'ault:amdv100'}:
            self.num_gpus_per_node = 2
        elif cp in {'ault:amdvega'}:
            self.num_gpus_per_node = 3
        else:
            self.num_gpus_per_node = 1

    @rfm.run_before('performance')
    def report_nid_with_smallest_flops(self):
        regex = r'\[(\S+)\] GPU\s+\d\(OK\): (\d+) GF/s'
        rptf = os.path.join(self.stagedir, sn.evaluate(self.stdout))
        self.nids = sn.extractall(regex, rptf, 1)
        self.flops = sn.extractall(regex, rptf, 2, float)
        # find index of smallest flops:
        index = self.flops.evaluate().index(min(self.flops))
        self.unit = f'GF/s ({self.nids[index]})'
        self.reference['dom:gpu:perf'] = (4115, -0.10, None, self.unit)
        self.reference['daint:gpu:perf'] = (4115, -0.10, None, self.unit)
        self.reference['arolla:cn:perf'] = (5861, -0.10, None, self.unit)
        self.reference['tsa:cn:perf'] = (5861, -0.10, None, self.unit)
        self.reference['ault:amda100:perf'] = (15000, -0.10, None, self.unit)
        self.reference['ault:amdv100:perf'] = (5500, -0.10, None, self.unit)
        self.reference['ault:intelv100:perf'] = (5500, -0.10, None, self.unit)
        self.reference['ault:amdvega:perf'] = (3450, -0.10, None, self.unit)
