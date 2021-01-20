# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.16-dev0')
@rfm.parameterized_test(['nompi'], ['mpi'])
class FFTWTest(rfm.RegressionTest):
    def __init__(self, exec_mode):
        self.sourcepath = 'fftw_benchmark.c'
        self.build_system = 'SingleSource'
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.modules = ['cray-fftw']
        self.num_tasks_per_node = 12
        self.num_gpus_per_node = 0
        self.sanity_patterns = sn.assert_eq(
            sn.count(sn.findall(r'execution time', self.stdout)), 1)
        self.build_system.cflags = ['-O2']
        if self.current_system.name in {'daint', 'dom'}:
            # Cray FFTW library is not officially supported for the PGI
            self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu']

        self.perf_patterns = {
            'fftw_exec_time': sn.extractsingle(
                r'execution time:\s+(?P<exec_time>\S+)', self.stdout,
                'exec_time', float),
        }

        if exec_mode == 'nompi':
            self.num_tasks = 12
            self.executable_opts = ['72 12 1000 0']
            self.reference = {
                'dom:gpu': {
                    'fftw_exec_time': (0.59, None, 0.05, 's'),
                },
                'daint:gpu': {
                    'fftw_exec_time': (0.59, None, 0.05, 's'),
                },
            }
        else:
            self.num_tasks = 72
            self.executable_opts = ['144 72 200 1']
            self.reference = {
                'dom:gpu': {
                    'fftw_exec_time': (0.47, None, 0.50, 's'),
                },
                'daint:gpu': {
                    'fftw_exec_time': (0.47, None, 0.50, 's'),
                },
            }

        self.maintainers = ['AJ']
        self.tags = {'benchmark', 'scs', 'craype'}
