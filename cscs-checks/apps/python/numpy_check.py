# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
from hpctestlib.apps.python.base_check import Numpy_BaseCheck


REFERENCE_PERFOMANCE = {
    'daint:gpu': {
        'dot': (0.4, None, 0.05, 'seconds'),
        'svd': (0.37, None, 0.05, 'seconds'),
        'cholesky': (0.12, None, 0.05, 'seconds'),
        'eigendec': (3.5, None, 0.05, 'seconds'),
        'inv': (0.21, None, 0.05, 'seconds'),
    },
    'daint:mc': {
        'dot': (0.3, None, 0.05, 'seconds'),
        'svd': (0.35, None, 0.05, 'seconds'),
        'cholesky': (0.1, None, 0.05, 'seconds'),
        'eigendec': (4.14, None, 0.05, 'seconds'),
        'inv': (0.16, None, 0.05, 'seconds'),
    },
    'dom:gpu': {
        'dot': (0.4, None, 0.05, 'seconds'),
        'svd': (0.37, None, 0.05, 'seconds'),
        'cholesky': (0.12, None, 0.05, 'seconds'),
        'eigendec': (3.5, None, 0.05, 'seconds'),
        'inv': (0.21, None, 0.05, 'seconds'),
    },
    'dom:mc': {
        'dot': (0.3, None, 0.05, 'seconds'),
        'svd': (0.35, None, 0.05, 'seconds'),
        'cholesky': (0.1, None, 0.05, 'seconds'),
        'eigendec': (4.14, None, 0.05, 'seconds'),
        'inv': (0.16, None, 0.05, 'seconds'),
    },
}


@rfm.simple_test
class Numpy_TestCSCS(Numpy_BaseCheck):
    valid_prog_environs = ['builtin']
    modules = ['numpy']
    variables = {
        'OMP_NUM_THREADS': '$SLURM_CPUS_PER_TASK',
    }
    num_tasks_per_node = 1
    use_multithreading = False
    tags = {'production'}
    maintainers = ['RS', 'TR']
    reference = REFERENCE_PERFOMANCE
    platform = parameter(['gpu', 'cpu'])

    @run_after('init')
    def set_system(self):
        if self.platform == 'gpu':
            self.valid_systems = ['daint:gpu', 'dom:gpu']
            self.num_cpus_per_task = 12
        else:
            self.valid_systems = ['daint:mc', 'dom:mc']
            self.num_cpus_per_task = 36
