# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib

import reframe as rfm
from hpctestlib.apps.amber.nve import Amber_NVE


@rfm.simple_test
class amber_check(Amber_NVE):
    modules = ['Amber']
    valid_prog_environs = ['builtin']
    extra_resources = {
        'switches': {
            'num_switches': 1
        }
    }
    maintainers = ['VH', 'SO']
    num_nodes = parameter([1, 4, 6, 8, 16])
    allref = {
        1: {
            'p100': {
                'Cellulose_production_NVE': (30.0, -0.05, None, 'ns/day'),
                'FactorIX_production_NVE': (134.0, -0.05, None, 'ns/day'),
                'JAC_production_NVE': (388.0, -0.05, None, 'ns/day'),
                'JAC_production_NVE_4fs': (742, -0.05, None, 'ns/day')
            }
        },
        4: {
            'zen2': {
                'Cellulose_production_NVE': (3.2, -0.30, None, 'ns/day'),
                'FactorIX_production_NVE': (7.0, -0.30, None, 'ns/day'),
                'JAC_production_NVE': (30.0, -0.30, None, 'ns/day'),
                'JAC_production_NVE_4fs': (45.0, -0.30, None, 'ns/day')
            }
        },
        6: {
            'broadwell': {
                'Cellulose_production_NVE': (8.0, -0.30, None, 'ns/day'),
                'FactorIX_production_NVE': (34.0, -0.30, None, 'ns/day'),
                'JAC_production_NVE': (90.0, -0.30, None, 'ns/day'),
                'JAC_production_NVE_4fs': (150.0, -0.30, None, 'ns/day')
            }
        },
        8: {
            'zen2': {
                'Cellulose_production_NVE': (1.3, -0.30, None, 'ns/day'),
                'FactorIX_production_NVE': (3.5, -0.30, None, 'ns/day'),
                'JAC_production_NVE': (17.0, -0.30, None, 'ns/day'),
                'JAC_production_NVE_4fs': (30.5, -0.30, None, 'ns/day')
            }
        },
        16: {
            'broadwell': {
                'Cellulose_production_NVE': (10.0, -0.30, None, 'ns/day'),
                'FactorIX_production_NVE': (36.0, -0.30, None, 'ns/day'),
                'JAC_production_NVE': (78.0, -0.30, None, 'ns/day'),
                'JAC_production_NVE_4fs': (135.0, -0.30, None, 'ns/day')
            }
        }
    }

    @run_after('init')
    def scope_systems(self):
        valid_systems = {
            'gpu': {1: ['daint:gpu', 'dom:gpu']},
            'cpu': {
                4: ['eiger:mc', 'pilatus:mc'],
                6: ['daint:mc', 'dom:mc'],
                8: ['pilatus:mc'],
                16: ['daint:mc']
            }
        }
        try:
            self.valid_systems = valid_systems[self.platform][self.num_nodes]
        except KeyError:
            self.valid_systems = []

    @run_after('init')
    def set_hierarchical_prgenvs(self):
        if self.current_system.name in ['eiger', 'pilatus']:
            self.valid_prog_environs = ['cpeIntel']

    @run_after('init')
    def set_num_gpus_per_node(self):
        if self.platform == 'gpu':
            self.num_gpus_per_node = 1

    @run_after('setup')
    def skip_if_no_topo(self):
        proc = self.current_partition.processor
        pname = self.current_partition.fullname
        if not proc.info:
            self.skip(f'no topology information found for partition {pname!r}')

    @run_after('setup')
    def set_num_tasks(self):
        if self.platform == 'gpu':
            self.num_tasks_per_node = 1
        else:
            proc = self.current_partition.processor
            pname = self.current_partition.fullname
            self.num_tasks_per_node = proc.num_cores

        self.num_tasks = self.num_nodes * self.num_tasks_per_node

    @run_before('performance')
    def set_perf_reference(self):
        proc = self.current_partition.processor
        pname = self.current_partition.fullname
        if pname in ('daint:gpu', 'dom:gpu'):
            arch = 'p100'
        else:
            arch = proc.arch

        with contextlib.suppress(KeyError):
            self.reference = {
                '*': {
                    'perf': self.allref[self.num_nodes][arch][self.benchmark]
                }
            }
