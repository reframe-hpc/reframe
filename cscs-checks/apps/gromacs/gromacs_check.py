# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
from hpctestlib.apps.gromacs.base_check import Gromacs_BaseCheck

REFERENCE_GPU_PERFORMANCE = {
    'large': {
        'daint:gpu': {
            'maint': (63.0, -0.10, None, 'ns/day'),
            'prod': (63.0, -0.20, None, 'ns/day'),
        },
    },
    'small': {
        'daint:gpu': {
            'prod': (35.0, -0.10, None, 'ns/day'),
        },
        'dom:gpu': {
            'prod': (37.0, -0.05, None, 'ns/day'),
        },
    }
}

REFERENCE_CPU_PERFORMANCE = {
    'large': {
        'daint:mc': {
            'prod': (68.0, -0.20, None, 'ns/day'),
        },
        'eiger:mc': {
            'prod': (146.00, -0.20, None, 'ns/day'),
        },
        'pilatus:mc': {
            'prod': (146.00, -0.20, None, 'ns/day'),
        },
    },
    'small': {
        'daint:mc': {
            'prod': (38.8, -0.10, None, 'ns/day'),
        },
        'dom:mc': {
            'prod': (40.0, -0.05, None, 'ns/day'),
        },
        'eiger:mc': {
            'prod': (103.00, -0.10, None, 'ns/day'),
        },
        'dom:mc': {
            'prod': (103.00, -0.10, None, 'ns/day'),
        },
    }
}


class GromacsBaseCheck(Gromacs_BaseCheck):
    scale = parameter(['small', 'large'])
    modules = ['GROMACS']
    maintainers = ['VH', 'SO']
    strict_check = False
    use_multithreading = False
    extra_resources = {
        'switches': {
            'num_switches': 1
        }
    }
    tags = {'scs', 'external-resources'}

    @run_after('init')
    def env_define(self):
        if self.current_system.name in ['eiger', 'pilatus']:
            self.valid_prog_environs = ['cpeGNU']
        else:
            self.valid_prog_environs = ['builtin']

    @run_after('init')
    def set_tags(self):
        self.tags |= {'maintenance' if self.mode == 'maint'
                      else 'production'}

    @run_after('setup')
    def set_reference(self):
        self.reference = self.reference_dict[self.scale]


@rfm.simple_test
class gromacs_gpu_check(GromacsBaseCheck):
    mode = parameter(['maint', 'prod'])
    valid_systems = ['daint:gpu']
    descr = 'GROMACS GPU check'
    executable_opts = ['mdrun', '-dlb yes', '-ntomp 1', '-npme 0',
                       '-s herflat.tpr']
    variables = {'CRAY_CUDA_MPS': '1'}
    num_gpus_per_node = 1
    reference_dict = REFERENCE_GPU_PERFORMANCE

    @run_after('setup')
    def set_num_tasks(self):
        if self.scale == 'small':
            self.valid_systems += ['dom:gpu']
            self.num_tasks = 72
            self.num_tasks_per_node = 12
        else:
            self.num_tasks = 192
            self.num_tasks_per_node = 12


@rfm.simple_test
class gromacs_cpu_check(GromacsBaseCheck):
    mode = parameter(['prod'])
    valid_systems = ['daint:mc', 'eiger:mc', 'pilatus:mc']
    descr = 'GROMACS CPU check'
    executable_opts = ['mdrun', '-dlb yes', '-ntomp 1', '-npme -1',
                       '-nb cpu', '-s herflat.tpr']
    reference_dict = REFERENCE_CPU_PERFORMANCE

    @run_after('setup')
    def set_num_tasks(self):
        if self.scale == 'small':
            self.valid_systems += ['dom:mc']
            if (self.current_system.name in ['daint', 'dom']):
                self.num_tasks = 216
                self.num_tasks_per_node = 36
            elif (self.current_system.name in ['eiger', 'pilatus']):
                self.num_tasks = 768
                self.num_tasks_per_node = 128
        else:
            if (self.current_system.name in ['daint', 'dom']):
                self.num_tasks = 576
                self.num_tasks_per_node = 36
            elif (self.current_system.name in ['eiger', 'pilatus']):
                self.num_tasks = 2048
                self.num_tasks_per_node = 128
