# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe.utility.sanity as sn
import reframe as rfm

import os
from math import ceil


class Pchase:
    '''
    Public storage class to avoid writing the parameters below multiple times.
    '''
    single_device = ['daint:gpu', 'dom:gpu']
    multi_device = ['ault:intelv100', 'ault:amdv100',
                    'ault:amda100', 'ault:amdvega',
                    'tsa:cn']
    valid_systems = single_device+multi_device
    valid_prog_environs = ['PrgEnv-gnu']


#
# PChase tests tracking the averaged latencies for all node jumps
#


@rfm.simple_test
class CompileGpuPointerChase(rfm.CompileOnlyRegressionTest):
    def __init__(self):
        self.valid_systems = Pchase.valid_systems
        self.valid_prog_environs = Pchase.valid_prog_environs
        self.exclusive_access = True
        self.build_system = 'Make'
        self.num_tasks = 0
        self.num_tasks_per_node = 1
        self.postbuild_cmds = ['ls .']
        self.sanity_patterns = sn.assert_found(r'pChase.x', self.stdout)
        self.maintainers = ['JO']
        self.tags = {'benchmark'}

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
                if cp == 'dom:gpu':
                    self.modules += ['cdt-cuda']

            else:
                self.modules += ['cuda']

        # Deal with the AMD options
        amd_trgt = None
        if cp == 'ault:amdvega':
            amd_trgt = 'gfx906,gfx908'

        if amd_trgt:
            self.build_system.cxxflags += [f'--amdgpu-target={amd_trgt}']
            self.modules += ['rocm']


class GpuPointerChaseBase(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.valid_prog_environs = Pchase.valid_prog_environs
        self.num_tasks = 0
        self.num_tasks_per_node = 1
        self.exclusive_access = True
        self.sanity_patterns = self.do_sanity_check()
        self.maintainers = ['JO']
        self.tags = {'benchmark'}

    @rfm.run_before('run')
    def set_num_gpus_per_node(self):
        cp = self.current_partition.fullname
        if cp == 'tsa:cn':
            self.num_gpus_per_node = 8
        elif cp in {'ault:intelv100', 'ault:amda100'}:
            self.num_gpus_per_node = 4
        elif cp in {'ault:amdv100'}:
            self.num_gpus_per_node = 2
        elif cp in {'ault:amdvega'}:
            self.num_gpus_per_node = 3
        else:
            self.num_gpus_per_node = 1

    @sn.sanity_function
    def do_sanity_check(self):

        # Check that every node has the right number of GPUs
        # Store this nodes in case they're used later by the perf functions.
        self.my_nodes = set(sn.extractall(
            rf'^\s*\[([^\]]*)\]\s*Found {self.num_gpus_per_node} device\(s\).',
            self.stdout, 1))

        # Check that every node has made it to the end.
        nodes_at_end = len(set(sn.extractall(
            r'^\s*\[([^\]]*)\]\s*Pointer chase complete.',
            self.stdout, 1)))
        return sn.evaluate(sn.assert_eq(
            sn.assert_eq(self.job.num_tasks, len(self.my_nodes)),
            sn.assert_eq(self.job.num_tasks, nodes_at_end)))


class GpuPointerChaseDep(GpuPointerChaseBase):
    def __init__(self):
        super().__init__()
        self.depends_on('CompileGpuPointerChase')

    @rfm.require_deps
    def set_executable(self, CompileGpuPointerChase):
        self.executable = os.path.join(
            CompileGpuPointerChase().stagedir, 'pChase.x')


@rfm.simple_test
class GpuPointerChaseClockLatency(GpuPointerChaseDep):
    '''
    Check the clock latencies.
    '''

    def __init__(self):
        super().__init__()
        self.valid_systems = Pchase.valid_systems
        self.executable_opts = ['--clock']
        self.perf_patterns = {
            'clock_latency': sn.max(sn.extractall(
                r'^\s*\[[^\]]*\]\s*The clock latency on device \d+ '
                r'is (\d+) cycles.', self.stdout, 1, int)
            ),
        }

        self.reference = {
            'daint:gpu': {
                'clock_latency': (56, None, 0.1, 'cycles'),
            },
            'dom:gpu': {
                'clock_latency': (56, None, 0.1, 'cycles'),
            },
            'tsa:cn': {
                'clock_latency': (8, None, 0.1, 'cycles'),
            },
            'ault:amda100': {
                'clock_latency': (7, None, 0.1, 'cycles'),
            },
            'ault:amdv100': {
                'clock_latency': (8, None, 0.1, 'cycles'),
            },
            'ault:amdvega': {
                'clock_latency': (40, None, 0.1, 'cycles'),
            },
        }


@rfm.parameterized_test([1], [2], [4], [4096])
class GpuPointerChaseSingle(GpuPointerChaseDep):
    '''
    Pointer chase on a single device with increasing stride.
    '''

    def __init__(self, stride):
        super().__init__()
        self.valid_systems = Pchase.valid_systems
        self.executable_opts = ['--sparsity', f'{stride}']
        self.perf_patterns = {
            'average_latency': sn.max(sn.extractall(
                r'^\s*\[[^\]]*\]\s* On device \d+, '
                r'the chase took on average (\d+) '
                r'cycles per node jump.', self.stdout, 1, int)
            ),
        }

        if stride == 1:
            self.reference = {
                'tsa:cn': {
                    'average_latency': (80, None, 0.1, 'clock cycles')
                },
                'ault:amda100': {
                    'average_latency': (76, None, 0.1, 'clock cycles')
                },
                'ault:amdv100': {
                    'average_latency': (77, None, 0.1, 'clock cycles')
                },
                'dom:gpu': {
                    'average_latency': (143, None, 0.1, 'clock cycles')
                },
                'daint:gpu': {
                    'average_latency': (143, None, 0.1, 'clock cycles')
                },
                'ault:amdvega': {
                    'average_latency': (225, None, 0.1, 'clock cycles')
                },
            }
        elif stride == 2:
            self.reference = {
                'tsa:cn': {
                    'average_latency': (120, None, 0.1, 'clock cycles')
                },
                'ault:amda100': {
                    'average_latency': (116, None, 0.1, 'clock cycles')
                },
                'ault:amdv100': {
                    'average_latency': (118, None, 0.1, 'clock cycles')
                },
                'dom:gpu': {
                    'average_latency': (181, None, 0.1, 'clock cycles')
                },
                'daint:gpu': {
                    'average_latency': (181, None, 0.1, 'clock cycles')
                },
                'ault:amdvega': {
                    'average_latency': (300, None, 0.1, 'clock cycles')
                },
            }
        elif stride == 4:
            self.reference = {
                'tsa:cn': {
                    'average_latency': (204, None, 0.1, 'clock cycles')
                },
                'ault:amda100': {
                    'average_latency': (198, None, 0.1, 'clock cycles')
                },
                'ault:amdv100': {
                    'average_latency': (204, None, 0.1, 'clock cycles')
                },
                'dom:gpu': {
                    'average_latency': (260, None, 0.1, 'clock cycles')
                },
                'daint:gpu': {
                    'average_latency': (260, None, 0.1, 'clock cycles')
                },
                'ault:amdvega': {
                    'average_latency': (470, None, 0.1, 'clock cycles')
                },
            }
        elif stride == 4096:
            self.reference = {
                'tsa:cn': {
                    'average_latency': (220, None, 0.1, 'clock cycles')
                },
                'ault:amda100': {
                    'average_latency': (206, None, 0.1, 'clock cycles')
                },
                'ault:amdv100': {
                    'average_latency': (220, None, 0.1, 'clock cycles')
                },
                'dom:gpu': {
                    'average_latency': (260, None, 0.1, 'clock cycles')
                },
                'daint:gpu': {
                    'average_latency': (260, None, 0.1, 'clock cycles')
                },
                'ault:amdvega': {
                    'average_latency': (800, None, 0.1, 'clock cycles')
                },
            }


@rfm.simple_test
class GpuPointerChaseAverageP2PLatency(GpuPointerChaseDep):
    '''
    Average inter-node P2P latency.
    '''

    def __init__(self):
        super().__init__()
        self.valid_systems = Pchase.multi_device
        self.executable_opts = ['--multi-gpu']
        self.perf_patterns = {
            'average_latency': self.average_P2P_latency(),
        }

        self.reference = {
            'ault:amda100': {
                'average_latency': (223, None, 0.1, 'clock cycles')
            },
            'ault:amdv100': {
                'average_latency': (611, None, 0.1, 'clock cycles')
            },
            'ault:amdvega': {
                'average_latency': (336, None, 0.1, 'clock cycles')
            },
            'tsa:cn': {
                'average_latency': (394, None, 0.1, 'clock cycles')
            },
        }

    @sn.sanity_function
    def average_P2P_latency(self):
        '''
        Extract the average P2P latency. Note that the pChase code
        returns a table with the cummulative latency for all P2P
        list traversals.
        '''
        return int(sn.evaluate(sn.max(sn.extractall(
                   r'^\s*\[[^\]]*\]\s*GPU\s*\d+\s+(\s*\d+.\s+)+',
                   self.stdout, 1, int)
        ))/(self.num_gpus_per_node-1)
        )
