# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.16-dev0')
@rfm.parameterized_test(['sync'], ['async'])
class KernelLatencyTest(rfm.RegressionTest):
    def __init__(self, kernel_version):
        self.valid_systems = ['daint:gpu', 'dom:gpu',
                              'arolla:cn', 'tsa:cn',
                              'ault:amdv100', 'ault:intelv100',
                              'ault:amda100', 'ault:amdvega']
        cs = self.current_system.name
        if cs in {'dom', 'daint'}:
            self.valid_prog_environs = ['PrgEnv-cray_classic', 'PrgEnv-cray',
                                        'PrgEnv-pgi', 'PrgEnv-gnu']
        elif cs in {'arolla', 'tsa'}:
            self.valid_prog_environs = ['PrgEnv-pgi']
        elif cs in {'ault'}:
            self.valid_prog_environs = ['PrgEnv-gnu']

        self.num_tasks = 0
        self.num_tasks_per_node = 1
        self.build_system = 'Make'
        self.executable = 'kernel_latency.x'
        if kernel_version == 'sync':
            self.build_system.cppflags = ['-D SYNCKERNEL=1']
        else:
            self.build_system.cppflags = ['-D SYNCKERNEL=0']

        self.sanity_patterns = self.assert_count_gpus()

        self.perf_patterns = {
            'latency': sn.max(sn.extractall(
                r'\[\S+\] \[gpu \d+\] Kernel launch latency: '
                r'(?P<latency>\S+) us', self.stdout, 'latency', float))
        }
        self.sys_reference = {
            'sync': {
                'dom:gpu': {
                    'latency': (6.6, None, 0.10, 'us')
                },
                'daint:gpu': {
                    'latency': (6.6, None, 0.10, 'us')
                },
                'ault:intelv100': {
                    'latency': (7.15, None, 0.10, 'us')
                },
                'ault:amdv100': {
                    'latency': (7.15, None, 0.10, 'us')
                },
                'ault:amda100': {
                    'latency': (9.65, None, 0.10, 'us')
                },
                'ault:amdvega': {
                    'latency': (15.1, None, 0.10, 'us')
                },
            },
            'async': {
                'dom:gpu': {
                    'latency': (2.2, None, 0.10, 'us')
                },
                'daint:gpu': {
                    'latency': (2.2, None, 0.10, 'us')
                },
                'ault:intelv100': {
                    'latency': (1.83, None, 0.10, 'us')
                },
                'ault:amdv100': {
                    'latency': (1.83, None, 0.10, 'us')
                },
                'ault:amda100': {
                    'latency': (2.7, None, 0.10, 'us')
                },
                'ault:amdvega': {
                    'latency': (2.64, None, 0.10, 'us')
                },
            },
        }

        self.reference = self.sys_reference[kernel_version]

        self.maintainers = ['TM']
        self.tags = {'benchmark', 'diagnostic', 'craype'}

    @property
    @sn.sanity_function
    def num_tasks_assigned(self):
        return self.job.num_tasks

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
                self.modules = ['craype-accel-nvidia60']
                if cp == 'dom:gpu':
                    self.modules += ['cdt-cuda']

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
    def set_num_gpus_per_node(self):
        cp = self.current_partition.fullname
        cs = self.current_system.name
        if cs in {'dom', 'daint'}:
            self.num_gpus_per_node = 1
        elif cs in {'arola', 'tsa'}:
            self.num_gpus_per_node = 8
        elif cp in {'ault:amda100', 'ault:intelv100'}:
            self.num_gpus_per_node = 4
        elif cp in {'ault:amdav100'}:
            self.num_gpus_per_node = 2
        elif cp in {'ault:amdvega'}:
            self.num_gpus_per_node = 3

    @sn.sanity_function
    def assert_count_gpus(self):
        return sn.all([
            sn.assert_eq(
                sn.count(
                    sn.findall(r'\[\S+\] Found \d+ gpu\(s\)',
                               self.stdout)
                ),
                self.num_tasks_assigned
            ),
            sn.assert_eq(
                sn.count(
                    sn.findall(r'\[\S+\] \[gpu \d+\] Kernel launch '
                               r'latency: \S+ us', self.stdout)
                ),
                self.num_tasks_assigned * self.num_gpus_per_node
            )
        ])
