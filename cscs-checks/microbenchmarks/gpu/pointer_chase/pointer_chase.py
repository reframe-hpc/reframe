# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe.utility.typecheck as typ
import reframe as rfm

import cscslib.microbenchmarks.gpu.hooks as hooks
from library.microbenchmarks.gpu.pointer_chase import *


class Base_CSCS(rfm.RegressionMixin):
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

    # Inject external hooks
    set_gpu_arch = rfm.run_after('setup')(hooks.set_gpu_arch)
    set_gpus_per_node = rfm.run_before('run')(hooks.set_gpus_per_node)


@rfm.simple_test
class Build_GPU_pchase_check(Build_GPU_pchase, Base_CSCS):
    ''' Build the executable.'''

    @rfm.run_after('init')
    def set_prgenvs(self):
        self.valid_systems = (
            self.single_device_systems + self.multi_device_systems
        )
        self.valid_prog_environs = self.global_prog_environs


class Base_pchase(Run_GPU_pchase, Base_CSCS):
    @rfm.run_after('init')
    def set_deps_and_prgenvs(self):
        self.depends_on('Build_GPU_pchase_check')
        self.valid_systems = (
            self.single_device_systems + self.multi_device_systems
        )
        self.valid_prog_environs = self.global_prog_environs
        self.exclusive_access = True

    @rfm.require_deps
    def set_executable(self, Build_GPU_pchase_check):
        self.executable = os.path.join(
            Build_GPU_pchase_check().stagedir, 'pChase.x')


@rfm.simple_test
class GPU_L1_latency_check(Base_pchase):
    '''Measure L1 latency.

    The linked list fits in L1. The stride is set pretty large, but that does
    not matter for this case since everything is in L1.
    '''

    num_list_nodes = 16
    reference = {
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
class GPU_L2_latency_check(Base_pchase):
    '''Measure the L2 latency.

    The linked list is larger than L1, but it fits in L2. The stride is set
    to be larger than L1's cache line to avoid any hits in L1.
    '''

    num_list_nodes = 5000
    reference = {
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
class GPU_DRAM_latency_check(Base_pchase):
    '''Measure the DRAM latency.

    The linked list is large enough to fill the last cache level. Also, the
    stride during the traversal must me large enough that there are no
    cache hits at all.
    '''

    num_list_nodes = 2000000
    reference = {
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
class GPU_latency_check_D2D(Run_GPU_pchase_D2D, Base_CSCS):
    '''Measure the latency to remote device.

    Depending on the list size, the data might be cached in different places.
    A list_size of 2000000 will place the list on the DRAM of the remote
    device.
    '''

    list_size = parameter([5000, 2000000])

    @rfm.run_after('init')
    def set_deps_and_prgenvs(self):
        self.depends_on('Build_GPU_pchase_check')
        self.valid_systems = self.multi_device_systems
        self.valid_prog_environs = self.global_prog_environs
        self.num_list_nodes = self.list_size

    @rfm.run_before('performance')
    def set_references(self):
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
    def set_executable(self, Build_GPU_pchase_check):
        self.executable = os.path.join(
            Build_GPU_pchase_check().stagedir, 'pChase.x')
