# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.16-dev0')
@rfm.simple_test
class GPUShmemTest(rfm.RegressionTest):
    def __init__(self):
        self.valid_systems = ['daint:gpu', 'dom:gpu',
                              'ault:amdv100', 'ault:intelv100',
                              'ault:amda100', 'ault:amdvega']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.num_tasks = 0
        self.num_tasks_per_node = 1
        self.build_system = 'Make'
        self.executable = 'shmem.x'

        # Mark the Xdevice symlink as read-only
        self.readonly_files = ['Xdevice']

        self.sanity_patterns = self.assert_num_gpus()
        self.perf_patterns = {
            'bandwidth': sn.min(sn.extractall(
                r'^\s*\[[^\]]*\]\s*GPU\s*\d+: '
                r'Bandwidth\(double\) (?P<bw>\S+) GB/s',
                self.stdout, 'bw', float))
        }
        self.reference = {
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

        self.maintainers = ['SK']
        self.tags = {'benchmark', 'diagnostic', 'craype'}

    @property
    @sn.sanity_function
    def num_tasks_assigned(self):
        return self.job.num_tasks

    @sn.sanity_function
    def assert_num_gpus(self):
        return sn.assert_eq(
            sn.count(sn.findall(r'Bandwidth', self.stdout)),
            self.num_tasks_assigned * 2 * self.num_gpus_per_node)

    @rfm.run_after('setup')
    def select_makefile(self):
        cp = self.current_partition.fullname
        if cp == 'ault:amdvega':
            self.build_system.makefile = 'makefile.hip'
        else:
            self.build_system.makefile = 'makefile.cuda'

    @rfm.run_after('setup')
    def set_gpu_arch(self):
        cp = self.current_partition.fullname

        # Deal with the NVIDIA options first
        nvidia_sm = None
        if cp in {'tsa:cn', 'ault:intelv100', 'ault:amdv100'}:
            nvidia_sm = '70'
        elif cp == 'ault:amda100':
            nvidia_sm = '80'
        elif cp in {'dom:gpu', 'daint:gpu'}:
            nvidia_sm = '60'

        if nvidia_sm:
            self.build_system.cxxflags += [f'-arch=sm_{nvidia_sm}']
            if cp in {'dom:gpu', 'daint:gpu'}:
                self.modules += ['craype-accel-nvidia60']
            else:
                self.modules += ['cuda']

        # Deal with the AMD options
        amd_trgt = None
        if cp == 'ault:amdvega':
            amd_trgt = 'gfx906'

        if amd_trgt:
            self.build_system.cxxflags += [f'--amdgpu-target={amd_trgt}']
            self.modules += ['rocm']

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
