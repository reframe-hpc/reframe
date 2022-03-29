# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class CPMDCheck(rfm.RunOnlyRegressionTest):
    modules = ['CPMD']
    executable = 'cpmd.x'
    executable_opts = ['ana_c4h6.in']
    extra_resources = {
        'switches': {
            'num_switches': 1
        }
    }
    strict_check = False
    use_multithreading = False
    tags = {'maintenance', 'production'}
    maintainers = ['AJ', 'LM']

    num_nodes = parameter([6, 16], loggable=True)
    references = {
        6: {
            'sm_60': {
                'dom:gpu': {'time': (120, None, 0.15, 's')},
                'daint:gpu': {'time': (120, None, 0.15, 's')},
            },
            'broadwell': {
                'dom:mc': {'time': (150.0, None, 0.15, 's')},
                'daint:mc': {'time': (150.0, None, 0.15, 's')},
            },
        },
        16: {
            'sm_60': {
                'daint:gpu': {'time': (120, None, 0.15, 's')}
            },
            'broadwell': {
                'daint:mc': {'time': (150.0, None, 0.15, 's')},
            },
        }
    }

    @performance_function('s')
    def elapsed_time(self):
        return sn.extractsingle(r'^ cpmd(\s+[\d\.]+){3}\s+(?P<time>\S+)',
                                self.stdout, 'time', float)

    @sanity_function
    def assert_energy_diff(self):
        energy = sn.extractsingle(
            r'CLASSICAL ENERGY\s+-(?P<result>\S+)',
            self.stdout, 'result', float)
        energy_reference = 25.81
        energy_diff = sn.abs(energy - energy_reference)
        return sn.assert_lt(energy_diff, 0.26)

    @run_after('init')
    def setup_system_filtering(self):
        self.descr = f'CPMD check ({self.num_nodes} node(s))'

        # setup system filter
        valid_systems = {
            6: ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc'],
            16: ['daint:gpu', 'daint:mc']
        }

        self.skip_if(self.num_nodes not in valid_systems,
                     f'No valid systems found for {self.num_nodes}(s)')
        self.valid_systems = valid_systems[self.num_nodes]

        # setup programming environment filter
        self.valid_prog_environs = ['builtin']

    @run_before('run')
    def setup_run(self):
        # retrieve processor data
        self.skip_if_no_procinfo()
        proc = self.current_partition.processor

        # set architecture for GPU partition (no auto-detection)
        if self.current_partition.fullname in ['daint:gpu', 'dom:gpu']:
            arch = 'sm_60'
            self.variables = {
                'CRAY_CUDA_MPS': '1'
            }
        else:
            arch = proc.arch

        # common setup for every architecture
        self.job.launcher.options = ['--cpu-bind=cores']
        self.job.options = ['--distribution=block:block']
        # FIXME: the current test case does not scale beyond 72 MPI tasks
        # and needs to be updated (see the warning about XC_DRIVER IN &DFT)
        self.num_tasks_per_node = 72 // self.num_nodes
        self.num_tasks = self.num_nodes * self.num_tasks_per_node

        try:
            found = self.references[self.num_nodes][arch]
        except KeyError:
            self.skip(f'Configuration with {self.num_nodes} node(s) '
                      f'is not supported on {arch!r}')

        # setup performance references
        self.reference = self.references[self.num_nodes][arch]
