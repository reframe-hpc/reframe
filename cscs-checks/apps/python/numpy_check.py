# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm

from hpctestlib.apps.python.numpy.numpy_ops import numpy_ops_check


@rfm.simple_test
class cscs_numpy_test(numpy_ops_check):
    valid_prog_environs = ['builtin']
    valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc']
    modules = ['numpy']
    num_tasks_per_node = 1
    use_multithreading = False
    all_ref = {
        'haswell': {
            'dot': (0.4, None, 0.05, 'seconds'),
            'svd': (0.37, None, 0.05, 'seconds'),
            'cholesky': (0.12, None, 0.05, 'seconds'),
            'eigendec': (3.5, None, 0.05, 'seconds'),
            'inv': (0.21, None, 0.05, 'seconds'),
        },
        'broadwell': {
            'dot': (0.3, None, 0.05, 'seconds'),
            'svd': (0.35, None, 0.05, 'seconds'),
            'cholesky': (0.1, None, 0.05, 'seconds'),
            'eigendec': (4.14, None, 0.05, 'seconds'),
            'inv': (0.16, None, 0.05, 'seconds'),
        }
    }
    tags = {'production'}
    maintainers = ['RS', 'TR']

    @run_after('setup')
    def set_num_cpus_per_task(self):
        self.num_cpus_per_task = self.current_partition.processor.num_cores
        variables = {
            'OMP_NUM_THREADS': self.num_cpus_per_task
        }

    @run_before('performance')
    def set_perf_ref(self):
        arch = self.current_partition.processor.arch
        pname = self.current_partition.fullname
        self.reference = {
            pname: self.all_ref[arch]
        }
