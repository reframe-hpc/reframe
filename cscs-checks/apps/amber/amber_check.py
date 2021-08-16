# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
from hpctestlib.apps.amber.nve import Amber_NVE


# FIXME: Use tuples as dictionary keys as soon as
# https://github.com/eth-cscs/reframe/issues/2022 is in
daint_gpu_performance = {
    'Cellulose_production_NVE': (30.0, -0.05, None, 'ns/day'),
    'FactorIX_production_NVE': (134.0, -0.05, None, 'ns/day'),
    'JAC_production_NVE': (388.0, -0.05, None, 'ns/day'),
    'JAC_production_NVE_4fs': (742, -0.05, None, 'ns/day'),
}

REFERENCE_GPU_PERFORMANCE = {
    'daint:gpu': daint_gpu_performance,
    'dom:gpu': daint_gpu_performance
}

daint_mc_performance_small = {
    'Cellulose_production_NVE': (8.0, -0.30, None, 'ns/day'),
    'FactorIX_production_NVE': (34.0, -0.30, None, 'ns/day'),
    'JAC_production_NVE': (90.0, -0.30, None, 'ns/day'),
    'JAC_production_NVE_4fs': (150.0, -0.30, None, 'ns/day'),
}

eiger_mc_performance_small = {
    'Cellulose_production_NVE': (3.2, -0.30, None, 'ns/day'),
    'FactorIX_production_NVE': (7.0, -0.30, None, 'ns/day'),
    'JAC_production_NVE': (30.0, -0.30, None, 'ns/day'),
    'JAC_production_NVE_4fs': (45.0, -0.30, None, 'ns/day'),
}

REFERENCE_CPU_PERFORMANCE_SMALL = {
    'daint:mc': daint_mc_performance_small,
    'dom:mc': daint_mc_performance_small,
    'eiger:mc': eiger_mc_performance_small,
    'pilatus:mc': eiger_mc_performance_small,
}

REFERENCE_CPU_PERFORMANCE_LARGE = {
    'daint:mc': {
        'Cellulose_production_NVE': (10.0, -0.30, None, 'ns/day'),
        'FactorIX_production_NVE': (36.0, -0.30, None, 'ns/day'),
        'JAC_production_NVE': (78.0, -0.30, None, 'ns/day'),
        'JAC_production_NVE_4fs': (135.0, -0.30, None, 'ns/day'),
    },
    'eiger:mc': {
        'Cellulose_production_NVE': (1.3, -0.30, None, 'ns/day'),
        'FactorIX_production_NVE': (3.5, -0.30, None, 'ns/day'),
        'JAC_production_NVE': (17.0, -0.30, None, 'ns/day'),
        'JAC_production_NVE_4fs': (30.5, -0.30, None, 'ns/day'),
    },
}


def inherit_cpu_only(params):
    return tuple(filter(lambda p: p[0] == 'cpu', params))


def inherit_gpu_only(params):
    return tuple(filter(lambda p: p[0] == 'gpu', params))


class AmberCheckCSCS(Amber_NVE):
    modules = ['Amber']
    valid_prog_environs = ['builtin']
    strict_check = False
    extra_resources = {
        'switches': {
            'num_switches': 1
        }
    }
    maintainers = ['VH', 'SO']


@rfm.simple_test
class amber_gpu_check(AmberCheckCSCS):
    valid_systems = ['daint:gpu', 'dom:gpu']
    num_tasks = 1
    num_gpus_per_node = 1
    num_tasks_per_node = 1
    descr = f'Amber GPU check'
    tags.update({'maintenance', 'production', 'health'})
    reference = REFERENCE_GPU_PERFORMANCE
    platform_info = parameter(
        inherit_params=True,
        filter_params=inherit_gpu_only)


@rfm.simple_test
class amber_cpu_check(AmberCheckCSCS):
    tags.update({'maintenance', 'production'})
    scale = parameter(['small', 'large'])
    valid_systems = ['daint:mc', 'eiger:mc']
    platform_info = parameter(
        inherit_params=True,
        filter_params=inherit_cpu_only)

    @run_after('init')
    def set_description(self):
        self.mydescr = f'Amber parallel {self.scale} CPU check'

    @run_after('init')
    def set_additional_systems(self):
        if self.scale == 'small':
            self.valid_systems += ['dom:mc', 'pilatus:mc']

    @run_after('init')
    def set_hierarchical_prgenvs(self):
        if self.current_system.name in ['eiger', 'pilatus']:
            self.valid_prog_environs = ['cpeIntel']

    @run_after('init')
    def set_num_tasks_cray_xc(self):
        if self.current_system.name in ['daint', 'dom']:
            self.num_tasks_per_node = 36
            if self.scale == 'small':
                self.num_nodes = 6
            else:
                self.num_nodes = 16
            self.num_tasks = self.num_nodes * self.num_tasks_per_node

    @run_after('init')
    def set_num_tasks_cray_shasta(self):
        if self.current_system.name in ['eiger', 'pilatus']:
            self.num_tasks_per_node = 128
            if self.scale == 'small':
                self.num_nodes = 4
            else:
                # there are too many processors, the large jobs cannot start
                # need to decrease to just 8 nodes
                self.num_nodes = 8
            self.num_tasks = self.num_nodes * self.num_tasks_per_node

    @run_after('setup')
    def set_perf_reference(self):
        if self.scale == 'small':
            self.reference = REFERENCE_CPU_PERFORMANCE_SMALL
        else:
            self.reference = REFERENCE_CPU_PERFORMANCE_LARGE
