# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
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
                              'ault:amda100']
        self.num_tasks = 0
        self.num_tasks_per_node = 1
        self.sourcepath = 'kernel_latency.cu'
        self.build_system = 'SingleSource'
        self.build_system.cxxflags = ['-std=c++11', '-O3']
        if self.current_system.name in {'dom', 'daint'}:
            self.modules = ['craype-accel-nvidia60']
            self.valid_prog_environs = ['PrgEnv-cray_classic', 'PrgEnv-cray',
                                        'PrgEnv-pgi', 'PrgEnv-gnu']
        elif self.current_system.name in ['arolla', 'tsa']:
            self.valid_prog_environs = ['PrgEnv-pgi']
            self.modules = ['cuda/10.1.243']
        elif self.current_system.name in ['ault']:
            self.valid_prog_environs = ['PrgEnv-gnu']
            self.modules = ['cuda']
        else:
            self.num_gpus_per_node = 1
            self.valid_systems = ['*']
            self.valid_prog_environs = ['*']

        if kernel_version == 'sync':
            self.build_system.cppflags = ['-D SYNCKERNEL=1']
        else:
            self.build_system.cppflags = ['-D SYNCKERNEL=0']

        self.sanity_patterns = sn.all([
            sn.assert_eq(
                sn.count(sn.findall(r'\[\S+\] Found \d+ gpu\(s\)',
                                    self.stdout)),
                self.num_tasks_assigned),
            sn.assert_eq(
                sn.count(sn.findall(r'\[\S+\] \[gpu \d+\] Kernel launch '
                                    r'latency: \S+ us', self.stdout)),
                self.num_tasks_assigned * self.num_gpus_per_node)
        ])

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

            },
        }

        self.reference = self.sys_reference[kernel_version]

        self.maintainers = ['TM']
        self.tags = {'benchmark', 'diagnostic', 'craype'}

    @property
    @sn.sanity_function
    def num_tasks_assigned(self):
        return self.job.num_tasks

    @rfm.run_before('compile')
    def set_gpu_arch(self):
        cp = self.current_partition.fullname
        cs = self.current_system.name
        gpu_arch = None
        if cs in {'dom', 'daint'}:
            gpu_arch = '60'
        elif (cs in {'arola', 'tsa'} or
              cp in {'ault:amdv100', 'ault:intelv100'}):
            gpu_arch = '70'
        elif cp in {'ault:amda100'}:
            gpu_arch = '80'

        if gpu_arch:
            self.build_system.cxxflags += ['-arch=compute_%s' % gpu_arch,
                                           '-code=sm_%s' % gpu_arch]


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
