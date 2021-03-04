# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe.utility.sanity as sn
import reframe.utility.typecheck as typ
import reframe as rfm

import os


class PchaseGlobal(rfm.RegressionMixin):
    '''Handy class to store common test settings.
    '''
    single_device_systems = variable(
        typ.List[str],
        value=['daint:gpu', 'dom:gpu']
    )
    multi_device_systems = variable(
        typ.List[str],
        value=[
            'ault:intelv100', 'ault:amdv100',
            'ault:amda100', 'ault:amdvega', 'tsa:cn'
        ]
    )
    global_prog_environs = variable(list, value=['PrgEnv-gnu'])


@rfm.simple_test
class CompileGpuPointerChase(rfm.CompileOnlyRegressionTest, PchaseGlobal):
    def __init__(self):
        self.valid_systems = (
            self.single_device_systems + self.multi_device_systems
        )
        self.valid_prog_environs = self.global_prog_environs
        self.exclusive_access = True
        self.build_system = 'Make'
        self.num_tasks = 0
        self.num_tasks_per_node = 1
        self.postbuild_cmds = ['ls .']
        self.sanity_patterns = sn.assert_found(r'pChase.x', self.stdout)
        self.maintainers = ['JO', 'SK']
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


class GpuPointerChaseBase(rfm.RunOnlyRegressionTest, PchaseGlobal):
    '''Base RunOnly class.

    This runs the pointer chase algo on the linked list from the code compiled
    in the executable from the test above. The list is fully customisable
    through the command line, so the number of nodes, and the stride size for
    each jump will determine where the memory hits occur. This stride is set to
    32 node lengths (a node is 8 Bytes) to ensure that there is only a single
    node per cache line. The number of node jumps is set relatively large to
    ensure that the background effects are averaged out.

    Derived tests MUST set the number of list nodes.
    '''
    num_list_nodes = variable(int)

    # Use a large stride to ensure there's only a single node per cache line
    stride = variable(int, value=32)

    # Set a large number of node jumps to smooth out spurious effects
    num_node_jumps = variable(int, value=400000)

    def __init__(self):
        self.depends_on('CompileGpuPointerChase')
        self.valid_prog_environs = self.global_prog_environs
        self.num_tasks = 0
        self.num_tasks_per_node = 1
        self.exclusive_access = True
        self.sanity_patterns = self.do_sanity_check()
        self.maintainers = ['JO', 'SK']
        self.tags = {'benchmark'}

    @rfm.require_deps
    def set_executable(self, CompileGpuPointerChase):
        self.executable = os.path.join(
            CompileGpuPointerChase().stagedir, 'pChase.x')

    @rfm.run_before('run')
    def set_exec_opts(self):
        self.executable_opts += [
            f'--stride {self.stride}',
            f'--nodes {self.num_list_nodes}',
            f'--num-jumps {self.num_node_jumps}'
        ]

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


class GpuPointerChaseSingle(GpuPointerChaseBase):
    '''Base class for the single-GPU latency tests.'''

    def __init__(self):
        super().__init__()
        self.valid_systems = (
            self.single_device_systems + self.multi_device_systems
        )
        self.perf_patterns = {
            'average_latency': sn.max(sn.extractall(
                r'^\s*\[[^\]]*\]\s* On device \d+, '
                r'the chase took on average (\d+) '
                r'cycles per node jump.', self.stdout, 1, int)
            ),
        }


@rfm.simple_test
class GpuL1Latency(GpuPointerChaseSingle):
    '''Measure L1 latency.

    The linked list fits in L1. The stride is set pretty large, but that does
    not matter for this case since everything is in L1.
    '''
    num_list_nodes = 16

    def __init__(self):
        super().__init__()
        self.reference = {
            'dom:gpu': {
                'average_latency': (103, None, 0.1, 'clock cycles')
            },
            'daint:gpu': {
                'average_latency': (103, None, 0.1, 'clock cycles')
            },
            'tsa:cn': {
                'average_latency': (28, None, 0.1, 'clock cycles')
            },
            'ault:amda100': {
                'average_latency': (33, None, 0.1, 'clock cycles')
            },
            'ault:amdv100': {
                'average_latency': (28, None, 0.1, 'clock cycles')
            },
            'ault:amdvega': {
                'average_latency': (140, None, 0.1, 'clock cycles')
            },
        }


@rfm.simple_test
class GpuL2Latency(GpuPointerChaseSingle):
    '''Measure the L2 latency.

    The linked list is larger than L1, but it fits in L2. The stride is set
    to be larger than L1's cache line to avoid any hits in L1.
    '''
    num_list_nodes = 5000

    def __init__(self):
        super().__init__()
        self.reference = {
            'dom:gpu': {
                'average_latency': (290, None, 0.1, 'clock cycles')
            },
            'daint:gpu': {
                'average_latency': (258, None, 0.1, 'clock cycles')
            },
            'tsa:cn': {
                'average_latency': (215, None, 0.1, 'clock cycles')
            },
            'ault:amda100': {
                'average_latency': (204, None, 0.1, 'clock cycles')
            },
            'ault:amdv100': {
                'average_latency': (215, None, 0.1, 'clock cycles')
            },
            'ault:amdvega': {
                'average_latency': (290, None, 0.1, 'clock cycles')
            },
        }


@rfm.simple_test
class GpuDRAMLatency(GpuPointerChaseSingle):
    '''Measure the DRAM latency.

    The linked list is large enough to fill the last cache level. Also, the
    stride during the traversal must me large enough that there are no
    cache hits at all.
    '''
    num_list_nodes = 2000000

    def __init__(self):
        super().__init__()
        self.reference = {
            'dom:gpu': {
                'average_latency': (506, None, 0.1, 'clock cycles')
            },
            'daint:gpu': {
                'average_latency': (506, None, 0.1, 'clock cycles')
            },
            'tsa:cn': {
                'average_latency': (425, None, 0.1, 'clock cycles')
            },
            'ault:amda100': {
                'average_latency': (560, None, 0.1, 'clock cycles')
            },
            'ault:amdv100': {
                'average_latency': (425, None, 0.1, 'clock cycles')
            },
            'ault:amdvega': {
                'average_latency': (625, None, 0.1, 'clock cycles')
            },
        }


class GpuP2PLatency(GpuPointerChaseBase):
    '''List traversal is done from a remote GPU.'''
    num_list_nodes = required

    def __init__(self):
        super().__init__()
        self.valid_systems = self.multi_device_systems
        self.executable_opts += ['--multi-gpu']
        self.perf_patterns = {
            'average_latency': self.average_P2P_latency(),
        }

    @sn.sanity_function
    def average_P2P_latency(self):
        '''
        Extract the average P2P latency. Note that the pChase code
        returns a table with the cummulative latency for all P2P
        list traversals, and the last column of this table has the max
        values for each device.
        '''
        return int(sn.evaluate(
            sn.max(sn.extractall(
                r'^\s*\[[^\]]*\]\s*GPU\s*\d+\s+(\s*\d+.\s+)+',
                self.stdout, 1, int)
            )
        ))


@rfm.simple_test
class GpuP2PLatencyP2P(GpuP2PLatency):
    '''Measure the latency to remote device.

    Depending on the list size, the data might be cached in different places.
    A list_size of 2000000 will place the list on the DRAM of the remote device.
    '''
    list_size = parameter([5000, 2000000])
    num_list_nodes = 2000000

    def __init__(self):
        super().__init__()
        self.num_list_nodes = self.list_size
        if self.list_size == 5000:
            self.reference = {
                'tsa:cn': {
                    'average_latency': (2981, None, 0.1, 'clock cycles')
                },
                'ault:amda100': {
                    'average_latency': (760, None, 0.1, 'clock cycles')
                },
                'ault:amdv100': {
                    'average_latency': (760, None, 0.1, 'clock cycles')
                },
                'ault:amdvega': {
                    'average_latency': (315, None, 0.1, 'clock cycles')
                },
            }
        elif self.list_size == 2000000:
            self.reference = {
                'tsa:cn': {
                    'average_latency': (3219, None, 0.1, 'clock cycles')
                },
                'ault:amda100': {
                    'average_latency': (1120, None, 0.1, 'clock cycles')
                },
                'ault:amdv100': {
                    'average_latency': (760, None, 0.1, 'clock cycles')
                },
                'ault:amdvega': {
                    'average_latency': (
                        3550, None, 0.1, 'clock cycles'
                    )
                },
            }
