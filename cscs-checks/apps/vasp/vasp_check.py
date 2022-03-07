# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class VASPCheck(rfm.RunOnlyRegressionTest):
    modules = ['VASP']
    executable = 'vasp_std'
    extra_resources = {
        'switches': {
            'num_switches': 1
        }
    }
    keep_files = ['OUTCAR']
    strict_check = False
    use_multithreading = False
    tags = {'maintenance', 'production'}
    maintainers = ['LM']

    num_nodes = parameter([6, 16], loggable=True)
    references = {
        6: {
            'sm_60': {
                'dom:gpu': {'time': (56.0, None, 0.10, 's')},
                'daint:gpu': {'time': (65.0, None, 0.15, 's')},
            },
            'broadwell': {
                'dom:mc': {'time': (58.0, None, 0.10, 's')},
                'daint:mc': {'time': (65.0, None, 0.15, 's')},
            },
            'zen2': {
                'eiger:mc': {'time': (100.0, None, 0.10, 's')},
                'pilatus:mc': {'time': (100.0, None, 0.10, 's')},
            },
        },
        16: {
            'sm_60': {
                'daint:gpu': {'time': (55.0, None, 0.15, 's')},
            },
            'broadwell': {
                'daint:mc': {'time': (55.0, None, 0.15, 's')},
            },
            'zen2': {
                'eiger:mc': {'time': (100.0, None, 0.10, 's')},
                'pilatus:mc': {'time': (100.0, None, 0.10, 's')}
            }
        }
    }

    @performance_function('s')
    def elapsed_time(self):
        return sn.extractsingle(r'Elapsed time \(sec\):'
                                r'\s+(?P<time>\S+)', 'OUTCAR',
                                'time', float)

    @sanity_function
    def assert_reference(self):
        force = sn.extractsingle(r'1 F=\s+(?P<result>\S+)',
                                 self.stdout, 'result', float)
        return sn.assert_reference(force, -.85026214E+03, -1e-5, 1e-5)

    @run_after('init')
    def setup_system_filtering(self):
        self.descr = f'VASP check ({self.num_nodes} node(s))'

        # setup system filter
        valid_systems = {
            6: ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
                'eiger:mc', 'pilatus:mc'],
            16: ['daint:gpu', 'daint:mc', 'eiger:mc']
        }

        self.skip_if(self.num_nodes not in valid_systems,
                     f'No valid systems found for {self.num_nodes}(s)')
        self.valid_systems = valid_systems[self.num_nodes]

        # setup programming environment filter
        if self.current_system.name in ['eiger', 'pilatus']:
            self.valid_prog_environs = ['cpeIntel']
        else:
            self.valid_prog_environs = ['builtin']


    @run_before('run')
    def setup_run(self):
        # set auto-detected architecture
        self.skip_if_no_procinfo()
        proc = self.current_partition.processor
        arch = proc.arch

        # set architecture for GPU partition (no auto-detection)
        if self.current_partition.fullname in ('daint:gpu', 'dom:gpu'):
            arch = 'sm_60'

        try:
            found = self.references[self.num_nodes][arch]
        except KeyError:
            self.skip(f'Configuration with {self.num_nodes} node(s) '
                      f'is not supported on {arch!r}')

        # common setup for every architecture
        self.job.launcher.options = ['--cpu-bind=cores']
        self.job.options = ['--distribution=block:block']
        self.num_tasks_per_node = proc.num_sockets
        self.num_cpus_per_task = proc.num_cores // self.num_tasks_per_node
        self.num_tasks = self.num_nodes * self.num_tasks_per_node
        self.variables = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'OMP_PLACES': 'cores',
            'OMP_PROC_BIND': 'close'
        }

        # custom settings for selected architectures
        if arch == 'sm_60':
            self.num_gpus_per_node = 1
        elif arch == 'zen2':
            self.variables += {
                'MPICH_OFI_STARTUP_CONNECT': '1'
            }

        # setup performance references
        self.reference = self.references[self.num_nodes][arch]
