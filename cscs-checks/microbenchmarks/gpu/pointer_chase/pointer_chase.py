# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe.utility.sanity as sn
import reframe.utility.typecheck as typ
import reframe as rfm

import library.microbenchmarks.gpu.pointer_chase as pchase
import cscslib.microbenchmarks.gpu.hooks as hooks


class PchaseGlobal(rfm.RegressionMixin):
    '''Handy class to store common test settings.'''

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
    global_prog_environs = variable(typ.List[str], value=['PrgEnv-gnu'])


@rfm.simple_test
class CompileGpuPChase(pchase.BuildGpuPChaseBase, PchaseGlobal,
                       hooks.SetCompileOpts):
    ''' Build the executable.'''

    def __init__(self):
        self.valid_systems = (
            self.single_device_systems + self.multi_device_systems
        )
        self.valid_prog_environs = self.global_prog_environs


class RunGpuPChaseSingle(pchase.RunGpuPChaseSingle, PchaseGlobal,
                         hooks.SetGPUsPerNode, hooks.SetCompileOpts):
    def __init__(self):
        self.depends_on('CompileGpuPChase')
        self.valid_systems = (
            self.single_device_systems + self.multi_device_systems
        )
        self.valid_prog_environs = self.global_prog_environs
        self.exclusive_access = True

    @rfm.require_deps
    def set_executable(self, CompileGpuPChase):
        self.executable = os.path.join(
            CompileGpuPChase().stagedir, 'pChase.x')


@rfm.simple_test
class GpuL1Latency(RunGpuPChaseSingle):
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
class GpuL2Latency(RunGpuPChaseSingle):
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
class GpuDRAMLatency(RunGpuPChaseSingle):
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


@rfm.simple_test
class GpuP2PLatencyP2P(pchase.RunGpuPChaseP2P, PchaseGlobal,
                       hooks.SetGPUsPerNode, hooks.SetCompileOpts):
    '''Measure the latency to remote device.

    Depending on the list size, the data might be cached in different places.
    A list_size of 2000000 will place the list on the DRAM of the remote
    device.
    '''

    list_size = parameter([5000, 2000000])

    def __init__(self):
        self.depends_on('CompileGpuPChase')
        self.valid_systems = self.multi_device_systems
        self.valid_prog_environs = self.global_prog_environs
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

    @rfm.require_deps
    def set_executable(self, CompileGpuPChase):
        self.executable = os.path.join(
            CompileGpuPChase().stagedir, 'pChase.x')
