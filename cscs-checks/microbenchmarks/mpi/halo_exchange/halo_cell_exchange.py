# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.16-dev0')
@rfm.simple_test
class HaloCellExchangeTest(rfm.RegressionTest):
    def __init__(self):
        self.sourcepath = 'halo_cell_exchange.c'
        self.build_system = 'SingleSource'
        self.build_system.cflags = ['-O2']
        self.valid_systems = ['daint:gpu', 'dom:gpu',
                              'arolla:cn', 'tsa:cn']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-pgi',
                                    'PrgEnv-gnu']
        self.num_tasks = 6
        self.num_tasks_per_node = 1
        self.num_gpus_per_node = 0

        self.executable_opts = ['input.txt']

        self.sanity_patterns = sn.assert_eq(
            sn.count(sn.findall(r'halo_cell_exchange', self.stdout)), 9)

        self.perf_patterns = {
            'time_2_10': sn.extractsingle(
                r'halo_cell_exchange 6 2 1 1 10 10 10'
                r' \S+ (?P<time_mpi>\S+)', self.stdout,
                'time_mpi', float),
            'time_2_10000': sn.extractsingle(
                r'halo_cell_exchange 6 2 1 1 10000 10000 10000'
                r' \S+ (?P<time_mpi>\S+)', self.stdout,
                'time_mpi', float),
            'time_2_1000000': sn.extractsingle(
                r'halo_cell_exchange 6 2 1 1 1000000 1000000 1000000'
                r' \S+ (?P<time_mpi>\S+)', self.stdout,
                'time_mpi', float),
            'time_4_10': sn.extractsingle(
                r'halo_cell_exchange 6 2 2 1 10 10 10'
                r' \S+ (?P<time_mpi>\S+)', self.stdout,
                'time_mpi', float),
            'time_4_10000': sn.extractsingle(
                r'halo_cell_exchange 6 2 2 1 10000 10000 10000'
                r' \S+ (?P<time_mpi>\S+)', self.stdout,
                'time_mpi', float),
            'time_4_1000000': sn.extractsingle(
                r'halo_cell_exchange 6 2 2 1 1000000 1000000 1000000'
                r' \S+ (?P<time_mpi>\S+)', self.stdout,
                'time_mpi', float),
            'time_6_10': sn.extractsingle(
                r'halo_cell_exchange 6 3 2 1 10 10 10'
                r' \S+ (?P<time_mpi>\S+)', self.stdout,
                'time_mpi', float),
            'time_6_10000': sn.extractsingle(
                r'halo_cell_exchange 6 3 2 1 10000 10000 10000'
                r' \S+ (?P<time_mpi>\S+)', self.stdout,
                'time_mpi', float),
            'time_6_1000000': sn.extractsingle(
                r'halo_cell_exchange 6 3 2 1 1000000 1000000 1000000'
                r' \S+ (?P<time_mpi>\S+)', self.stdout,
                'time_mpi', float)
        }

        self.reference = {
            'dom:gpu': {
                'time_2_10': (3.925395e-06, None, 0.50, 's'),
                'time_2_10000': (9.721279e-06, None, 0.50, 's'),
                'time_2_1000000': (4.934530e-04, None, 0.50, 's'),
                'time_4_10': (5.878997e-06, None, 0.50, 's'),
                'time_4_10000': (1.495080e-05, None, 0.50, 's'),
                'time_4_1000000': (6.791397e-04, None, 0.50, 's'),
                'time_6_10': (5.428815e-06, None, 0.50, 's'),
                'time_6_10000': (1.540580e-05, None, 0.50, 's'),
                'time_6_1000000': (9.179296e-04, None, 0.50, 's')
            },
            'daint:gpu': {
                'time_2_10': (3.925395e-06, None, 0.50, 's'),
                'time_2_10000': (9.721279e-06, None, 0.50, 's'),
                'time_2_1000000': (4.934530e-04, None, 0.50, 's'),
                'time_4_10': (5.878997e-06, None, 0.50, 's'),
                'time_4_10000': (1.495080e-05, None, 0.50, 's'),
                'time_4_1000000': (6.791397e-04, None, 0.50, 's'),
                'time_6_10': (5.428815e-06, None, 0.50, 's'),
                'time_6_10000': (1.540580e-05, None, 0.50, 's'),
                'time_6_1000000': (9.179296e-04, None, 0.50, 's')
            },
        }

        self.maintainers = ['AJ']
        self.tags = {'benchmark'}

    @rfm.run_before('compile')
    def pgi_workaround(self):
        if self.current_system.name in ['daint', 'dom']:
            if self.current_environ.name == 'PrgEnv-pgi':
                self.variables = {
                    'CUDA_HOME': '$CUDATOOLKIT_HOME',
                }
