# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class GPUdgemmTest(rfm.RegressionTest):
    def __init__(self):
        self.valid_systems = ['daint:gpu', 'dom:gpu',
                              'ault:amdv100', 'ault:intelv100',
                              'ault:amda100', 'ault:amdvega']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.num_tasks = 0
        self.num_tasks_per_node = 1
        self.build_system = 'Make'
        self.executable = 'dgemm.x'

        # FIXME workaround due to issue #1639.
        self.readonly_files = ['Xdevice']

        self.sanity_patterns = self.assert_num_gpus()
        self.perf_patterns = {
            'perf': sn.min(sn.extractall(
                r'^\s*\[[^\]]*\]\s*GPU\s*\d+: (?P<fp>\S+) TF/s',
                self.stdout, 'fp', float))
        }
        self.reference = {
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

        self.maintainers = ['JO']
        self.tags = {'benchmark'}

    @property
    @sn.sanity_function
    def num_tasks_assigned(self):
        return self.job.num_tasks

    @sn.sanity_function
    def assert_num_gpus(self):
        return sn.assert_eq(
            sn.count(sn.findall(r'^\s*\[[^\]]*\]\s*Test passed', self.stdout)),
            self.num_tasks_assigned)

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
