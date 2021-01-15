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

        # FIXME workaround due to issue #1639.
        self.readonly_files = ['Xdevice']

        self.maintainers = ['JO']

    @rfm.run_after('setup')
    def select_makefile(self):
        cp = self.current_partition.fullname
        if cp == 'ault:amdvega':
            self.prebuild_cmds = ['cp makefile.hip Makefile']
        else:
            self.prebuild_cmds = ['cp makefile.cuda Makefile']

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
            nvidia_sm == '60'

        if nvidia_sm:
            self.build_system.cxxflags += [f'-arch=sm_{nvidia_sm}']
            if cp in {'dom:gpu', 'daint:gpu'}:
                self.modules += ['cudatoolkit']
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
    Check the clock latencies. This can be thought of the
    measuring error.
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
        self.executable_opts = ['--stride', f'{stride}']
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
        self.executable_opts = ['--multiGPU']
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


#
# PChase tests tracking the individual latencies of each node jump
#


@rfm.simple_test
class CompileGpuPointerChaseFine(CompileGpuPointerChase):
    '''
    Compile the pChase code to time each node jump.
    '''

    def __init__(self):
        super().__init__()

    @rfm.run_before('compile')
    def set_cxxflags(self):
        self.build_system.cxxflags += ['-DTIME_EACH_STEP']


class GpuPointerChaseFineDep(GpuPointerChaseBase):
    def __init__(self):
        super().__init__()
        self.depends_on('CompileGpuPointerChaseFine')

    @rfm.require_deps
    def set_executable(self, CompileGpuPointerChaseFine):
        self.executable = os.path.join(
            CompileGpuPointerChaseFine().stagedir, 'pChase.x')

    @sn.sanity_function
    def get_all_latencies(self, pattern):
        return sn.extractall(pattern, self.stdout, 1, int)


class L1_filter:
    def filter_out_L1_hits(self, threshold, all_latencies):
        '''
        Return a list with the latencies that are above 20% threshold.
        '''
        return list(filter(lambda x: x > 1.2*threshold, all_latencies))


@rfm.simple_test
class GpuPointerChaseL1(GpuPointerChaseFineDep, L1_filter):
    '''
    Pointer chase for all the devices present on each node.
    The traversal is done with unit stride, checking the L1 latency,
    L1 miss rate and average latency of an L1 miss.
    '''

    def __init__(self):
        super().__init__()
        self.valid_systems = Pchase.valid_systems
        self.perf_patterns = {
            'L1_latency': self.max_L1_latency(),
            'L1_miss_rate': self.L1_miss_rate(),
            'L1_miss_latency': self.L1_miss_latency(),
        }

        self.reference = {
            'dom:gpu': {
                'L1_latency': (112, None, 0.1, 'clock cycles'),
                'L1_miss_rate': (33.3, None, 0.1, '%'),
                'L1_miss_latency': (268, None, 0.1, 'clock cycles'),

            },
            'daint:gpu': {
                'L1_latency': (112, None, 0.1, 'clock cycles'),
                'L1_miss_rate': (33.3, None, 0.1, '%'),
                'L1_miss_latency': (268, None, 0.1, 'clock cycles'),

            },
            'tsa:cn': {
                'L1_latency': (38, None, 0.1, 'clock cycles'),
                'L1_miss_rate': (25.4, None, 0.1, '%'),
                'L1_miss_latency': (240, None, 0.1, 'clock cycles'),
            },
            'ault:amda100': {
                'L1_latency': (42, None, 0.1, 'clock cycles'),
                'L1_misses': (25.4, None, 0.1, '%'),
                'L1_miss_latency': (215, None, 0.1, 'clock cycles'),
            },
            'ault:amdv100': {
                'L1_latency': (39, None, 0.1, 'clock cycles'),
                'L1_misses': (25.4, None, 0.1, '%'),
                'L1_miss_latency': (208, None, 0.1, 'clock cycles'),
            },
            'ault:amdvega': {
                'L1_latency': (164, None, 0.1, 'clock cycles'),
                'L1_miss_rate': (23.8, None, 0.1, '%'),
                'L1_miss_latency': (840, None, 0.1, 'clock cycles'),
            },
        }

    @staticmethod
    def target_str(node, device):
        return r'^\s*\[%s\]\[device %d\]\s*(\d+)' % (node, device)

    @sn.sanity_function
    def max_L1_latency(self):
        '''
        Max. L1 latency amongst all devices.
        '''
        l1_latency = []
        for n in self.my_nodes:
            for d in range(self.num_gpus_per_node):
                l1_latency.append(
                    sn.min(self.get_all_latencies(self.target_str(n, d)))
                )

        # Return the data from the worst performing device
        return sn.max(l1_latency)

    def get_L1_misses(self, n, d, all_latencies=None):
        '''
        The idea here is to get the lowest value and model the L1 hits as
        implemented in the self.filter_out_L1_hits function. Every
        node jump returned by this function will be counted as an L1 miss.
        '''
        if all_latencies is None:
            all_latencies = self.get_all_latencies(self.target_str(n, d))

        L1 = sn.min(all_latencies)
        return self.filter_out_L1_hits(L1, all_latencies)

    @sn.sanity_function
    def L1_miss_rate(self):
        '''
        Calculate the rate of L1 misses based on the model implemented by the
        get_L1_misses sanity function. Return the worst performing rate from
        all nodes/devices.
        '''
        l1_miss_rate = []
        for n in self.my_nodes:
            for d in range(self.num_gpus_per_node):
                all_lat = sn.evaluate(
                    self.get_all_latencies(self.target_str(n, d))
                )
                l1_miss_rate.append(
                    len(self.get_L1_misses(n, d, all_lat))/len(all_lat)
                )

        return max(l1_miss_rate)*100

    @sn.sanity_function
    def L1_miss_latency(self):
        '''
        Count the average number of cycles taken only by the node jumps
        with an L1 miss. Return the worst performing values for all
        nodes/devices.
        '''
        l1_miss_latency = []
        for n in self.my_nodes:
            for d in range(self.num_gpus_per_node):
                l1_miss_latency.append(
                    ceil(sn.evaluate(sn.avg(self.get_L1_misses(n, d))))
                )

        return max(l1_miss_latency)


@rfm.simple_test
class GpuPointerChaseL1P2P(GpuPointerChaseFineDep, L1_filter):
    '''
    Pointer chase through P2P, checking L1 miss rates and L1 miss
    latency averaged amogst all devices in each node.
    '''

    def __init__(self):
        super().__init__()
        self.valid_systems = Pchase.multi_device
        self.executable_opts = ['--multiGPU']
        self.perf_patterns = {
            'L1_latency': self.max_L1_latency(),
            'L1_miss_rate': self.L1_miss_rate(),
            'L1_miss_latency': self.L1_miss_latency()
        }
        self.reference = {
            'tsa:cn': {
                'L1_latency': (38, None, 0.1, 'clock cycles'),
                'L1_miss_rate': (25.4, None, 0.1, '%'),
                'L1_miss_latency': (1463, None, 0.1, 'clock cycles'),
            },
            'ault:amda100': {
                'L1_latency': (42, None, 0.1, 'clock cycles'),
                'L1_miss_rate': (25.4, None, 0.1, '%'),
                'L1_miss_latency': (792, None, 0.1, 'clock cycles'),
            },
            'ault:amdv100': {
                'L1_latency': (39, None, 0.1, 'clock cycles'),
                'L1_miss_rate': (25.4, None, 0.1, '%'),
                'L1_miss_latency': (2620, None, 0.1, 'clock cycles'),
            },
            'ault:amdvega': {
                'L1_latency': (164, None, 0.1, 'clock cycles'),
                'L1_miss_rate': (19.3, None, 0.1, '%'),
                'L1_miss_latency': (2200, None, 0.1, 'clock cycles'),
            },
        }

    @staticmethod
    def target_str(node, d1, d2):
        return r'^\s*\[%s\]\[device %d\]\[device %d\]\s*(\d+)' % (node, d1, d2)

    @sn.sanity_function
    def max_L1_latency(self):
        '''
        Max. L1 latency amongst all devices.
        '''
        l1_latency = []
        for n in self.my_nodes:
            for d1 in range(self.num_gpus_per_node):
                for d2 in range(self.num_gpus_per_node):
                    l1_latency.append(
                        sn.min(self.get_all_latencies(
                            self.target_str(n, d1, d2))
                        )
                    )

        # Return the data from the worst performing device
        return sn.max(l1_latency)

    @sn.sanity_function
    def L1_miss_rate(self):
        '''
        Calculates the L1 miss rate across P2P list traversals.
        '''
        total_node_jumps = 0
        total_L1_misses = 0
        for n in self.my_nodes:
            for d1 in range(self.num_gpus_per_node):
                for d2 in range(self.num_gpus_per_node):
                    if(d1 != d2):
                        all_lat = sn.evaluate(self.get_all_latencies(
                            self.target_str(n, d1, d2)
                        ))
                        L1 = min(all_lat)
                        total_L1_misses += len(
                            self.filter_out_L1_hits(L1, all_lat)
                        )
                        total_node_jumps += len(all_lat)

        return (total_L1_misses/total_node_jumps)*100

    @sn.sanity_function
    def L1_miss_latency(self):
        '''
        Calculate the latency of all L1 misses across all P2P list traversals
        '''
        L1_misses = []
        for n in self.my_nodes:
            for d1 in range(self.num_gpus_per_node):
                for d2 in range(self.num_gpus_per_node):
                    if (d1 != d2):
                        all_lat = sn.evaluate(self.get_all_latencies(
                            self.target_str(n, d1, d2)
                        ))
                        L1 = min(all_lat)
                        L1_misses += self.filter_out_L1_hits(L1, all_lat)

        return int(sn.evaluate(sn.avg(L1_misses)))
