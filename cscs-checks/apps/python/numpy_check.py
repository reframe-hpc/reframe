# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


class NumpyBaseTest(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.descr = 'Test a few typical numpy operations'
        self.valid_prog_environs = ['builtin']
        self.modules = ['numpy']
        self.reference = {
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
        self.perf_patterns = {
            'dot': sn.extractsingle(
                r'^Dotted two 4096x4096 matrices in\s+(?P<dot>\S+)\s+s',
                self.stdout, 'dot', float),
            'svd': sn.extractsingle(
                r'^SVD of a 2048x1024 matrix in\s+(?P<svd>\S+)\s+s',
                self.stdout, 'svd', float),
            'cholesky': sn.extractsingle(
                r'^Cholesky decomposition of a 2048x2048 matrix in'
                r'\s+(?P<cholesky>\S+)\s+s',
                self.stdout, 'cholesky', float),
            'eigendec': sn.extractsingle(
                r'^Eigendecomposition of a 2048x2048 matrix in'
                r'\s+(?P<eigendec>\S+)\s+s',
                self.stdout, 'eigendec', float),
            'inv': sn.extractsingle(
                r'^Inversion of a 2048x2048 matrix in\s+(?P<inv>\S+)\s+s',
                self.stdout, 'inv', float)
        }
        self.sanity_patterns = sn.assert_found(r'Numpy version:\s+\S+',
                                               self.stdout)
        self.variables = {
            'OMP_NUM_THREADS': '$SLURM_CPUS_PER_TASK',
        }
        self.executable = 'python'
        self.executable_opts = ['np_ops.py']
        self.num_tasks_per_node = 1
        self.use_multithreading = False
        self.tags = {'production'}
        self.maintainers = ['RS', 'TR']


@rfm.required_version('>=2.16')
@rfm.simple_test
class NumpyHaswellTest(NumpyBaseTest):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.num_cpus_per_task = 12


@rfm.required_version('>=2.16')
@rfm.simple_test
class NumpyBroadwellTest(NumpyBaseTest):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['daint:mc', 'dom:mc']
        self.num_cpus_per_task = 36
