# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
from hpctestlib.apps.cpmd.nve import Cpmd_NVE

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

REFERENCE_PERFORMANCE = {
    'small': REFERENCE_PERFORMANCE_SMALL,
    'large': REFERENCE_PERFORMANCE_LARGE,
}

@rfm.simple_test
class cpmd_check(Cpmd_NVE):
    scale = parameter(['small', 'large'])
    mode = parameter(['prod'])
    valid_systems = ['daint:gpu']
    modules = ['CPMD']
    valid_prog_environs = ['builtin']
    num_tasks_per_node = 1
    maintainers = ['AJ', 'LM']
    tags = {'production'}
    use_multithreading = True
    strict_check = False
    descr = 'CPMD check (C4H6 metadynamics)'
    extra_resources = {
        'switches': {
            'num_switches': 1
        }
    }

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
    def set_reference(self):
        self.reference = REFERENCE_PERFORMANCE[self.scale]
