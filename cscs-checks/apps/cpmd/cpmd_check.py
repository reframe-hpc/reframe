# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
from hpctestlib.apps.cpmd import Cpmd

REFERENCE_PERFORMANCE_SMALL = {
    'dom:mc': {
        'prod': (285.5, None, 0.20, 's')
    },
    'daint:mc': {
        'prod': (285.5, None, 0.20, 's')
    },
}

REFERENCE_PERFORMANCE_LARGE = {
    'daint:mc': {
        'prod': (245.0, None, 0.59, 's')
    },
}


@rfm.simple_test
class CPMDCheck(Cpmd):
    scale = parameter(['small', 'large'])
    maintainers = ['AJ', 'LM']
    tags = {'production'}
    valid_systems = ['daint:gpu']
    descr = 'CPMD check (C4H6 metadynamics)'
    num_tasks_per_node = 1
    valid_prog_environs = ['builtin']
    modules = ['CPMD']
    executable = 'cpmd.x'
    input_file = 'ana_c4h6.in'
    readonly_files = ['ana_c4h6.in', 'C_MT_BLYP', 'H_MT_BLYP']
    benchmark = parameter(['prod'])
    use_multithreading = True
    strict_check = False
    extra_resources = {
        'switches': {
            'num_switches': 1
        }
    }
    energy_value = 25.81
    energy_tolerance = 0.26

    @run_after('init')
    def set_reference(self):
        if self.scale == 'small':
            self.reference = REFERENCE_PERFORMANCE_SMALL
        else:
            self.reference = REFERENCE_PERFORMANCE_LARGE

    @run_after('init')
    def set_num_tasks(self):
        if self.scale == 'small':
            self.num_tasks = 9
            self.valid_systems += ['dom:gpu']
        else:
            self.num_tasks = 16
        #  OpenMP version of CPMD segfaults
        #  self.variables = { 'OMP_NUM_THREADS' : '8' }

    @run_after('setup')
    def set_generic_perf_references(self):
        self.reference.update({'*': {
            self.benchmark: (0, None, None, 's')
        }})

    @run_after('setup')
    def set_perf_patterns(self):
        self.perf_patterns = {
            self.benchmark: sn.extractsingle(
                r'^ cpmd(\s+[\d\.]+){3}\s+(?P<perf>\S+)',
                'stdout.txt', 'perf', float)
        }
