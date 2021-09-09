# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn
from hpctestlib.apps.namd.base_check import Namd_BaseCheck

REFERENCE_GPU_PERFORMANCE = {
    'dom:gpu': {
        'small': (0.15, None, 0.05, 'days/ns'),
    },
    'daint:gpu': {
        'small': (0.15, None, 0.05, 'days/ns'),
        'large': (0.07, None, 0.05, 'days/ns')
    },
}

REFERENCE_CPU_PERFORMANCE = {
    'dom:mc': {
        'small': (0.51, None, 0.05, 'days/ns'),
    },
    'daint:mc': {
        'small': (0.51, None, 0.05, 'days/ns'),
        'large': (0.28, None, 0.05, 'days/ns')
    },
    'eiger:mc': {
        'small': (0.12, None, 0.05, 'days/ns'),
        'large': (0.05, None, 0.05, 'days/ns')
    },
    'pilatus:mc': {
        'small': (0.12, None, 0.05, 'days/ns'),
        'large': (0.05, None, 0.05, 'days/ns')
    },
}

@rfm.simple_test
class NamdCheckCSCS(Namd_BaseCheck):
    maintainers = ['CB', 'LM']
    tags = {'scs', 'external-resources'}
    modules = ['NAMD']
    extra_resources = {
        'switches': {
            'num_switches': 1
        }
    }
    use_multithreading = True
    num_tasks_per_core = 2
    platform_name = parameter(['gpu', 'cpu'])
    scale = parameter(['small', 'large'])
    mode = parameter(['maint', 'prod'])

    @run_after('init')
    def env_define(self):
        if self.current_system.name in ['eiger', 'pilatus']:
            self.valid_prog_environs = ['cpeGNU']
        else:
            self.valid_prog_environs = ['builtin']

    @run_after('init')
    def set_description(self):
        self.mydescr = f'NAMD check ({self.platform_name}, {self.mode})'

    @run_after('init')
    def set_tags(self):
        self.tags |= {'maintenance' if self.mode == 'maint'
                      else 'production'}

    @run_after('init')
    def set_valid_systems(self):
        if self.platform_name =='gpu':
            self.valid_systems = ['daint:gpu']
        else:
            self.valid_systems = ['daint:mc',
                                  'eiger:mc',
                                  'pilatus:mc']

    @run_after('setup')
    def set_num_tasks(self):
        if self.scale == 'small':
            # On Eiger a no-smp NAMD version is the default
            if self.current_system.name in ['eiger', 'pilatus']:
                self.num_tasks = 768
                self.num_tasks_per_node = 128
            else:
                self.num_tasks = 6
                self.num_tasks_per_node = 1
        else:
            if self.current_system.name in ['eiger', 'pilatus']:
                self.num_tasks = 2048
                self.num_tasks_per_node = 128
            else:
                self.num_tasks = 16
                self.num_tasks_per_node = 1

    @run_after('setup')
    def set_executable_opts(self):
        if self.platform_name =='gpu':
            self.executable_opts = ['+idlepoll', '+ppn 23', 'stmv.namd']
            self.num_cpus_per_task = 24
            self.num_gpus_per_node = 1
        else:
            # On Eiger a no-smp NAMD version is the default
            if self.current_system.name in ['eiger', 'pilatus']:
                self.executable_opts = ['+idlepoll', 'stmv.namd']
                self.num_tasks_per_core = 2
            else:
                self.executable_opts = ['+idlepoll', '+ppn 71', 'stmv.namd']
                self.num_cpus_per_task = 72

    @run_after('setup')
    def set_reference(self):
        if self.platform_name =='gpu':
            self.reference = REFERENCE_GPU_PERFORMANCE
            if self.scale == 'small':
                self.valid_systems += ['dom:gpu']
        else:
            if self.scale == 'small':
                self.valid_systems += ['dom:mc']
            self.reference = REFERENCE_CPU_PERFORMANCE
