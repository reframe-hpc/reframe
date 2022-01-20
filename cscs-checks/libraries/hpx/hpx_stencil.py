# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class Stencil4HPXCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.descr = 'HPX 1d_stencil_4 check'
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc']
        self.valid_prog_environs = ['builtin']

        self.modules = ['HPX']
        self.executable = '1d_stencil_4'

        self.nt_opts = '100'  # number of time steps
        self.np_opts = '100'  # number of partitions
        self.nx_opts = '10000000'  # number of points per partition
        self.executable_opts = ['--nt', self.nt_opts,
                                '--np', self.np_opts,
                                '--nx', self.nx_opts]
        self.sourcesdir = None

        self.use_multithreading = None

        self.perf_patterns = {
            'time': sn.extractsingle(r'\d+,\s*(?P<time>(\d+)?.?\d+),\s*\d+,'
                                     r'\s*\d+,\s*\d+',
                                     self.stdout, 'time', float)
        }
        self.reference = {
            'dom:gpu': {
                'time': (42, None, 0.1, 's')
            },
            'dom:mc': {
                'time': (30, None, 0.1, 's')
            },
            'daint:gpu': {
                'time': (42, None, 0.1, 's')
            },
            'daint:mc': {
                'time': (30, None, 0.1, 's')
            },
        }

        self.maintainers = ['VH', 'JG']

    @run_after('setup')
    def set_tasks(self):
        if self.current_partition.fullname == 'daint:gpu':
            self.num_tasks = 1
            self.num_tasks_per_node = 1
            self.num_cpus_per_task = 12
        elif self.current_partition.fullname == 'daint:mc':
            self.num_tasks = 1
            self.num_tasks_per_node = 1
            self.num_cpus_per_task = 36
        elif self.current_partition.fullname == 'dom:gpu':
            self.num_tasks = 1
            self.num_tasks_per_node = 1
            self.num_cpus_per_task = 12
        elif self.current_partition.fullname == 'dom:mc':
            self.num_tasks = 1
            self.num_tasks_per_node = 1
            self.num_cpus_per_task = 36

    @run_before('run')
    def set_exec_opts(self):
        self.executable_opts += ['--hpx:threads=%s' % self.num_cpus_per_task]

    @run_before('sanity')
    def set_sanity(self):
        result = sn.findall(r'(?P<tid>\d+),\s*(?P<time>(\d+)?.?\d+),'
                            r'\s*(?P<pts>\d+),\s*(?P<parts>\d+),'
                            r'\s*(?P<steps>\d+)',
                            self.stdout)
        assert_num_threads = sn.map(lambda x: sn.assert_eq(
            int(x.group('tid')), self.num_cpus_per_task), result)
        assert_num_points = sn.map(lambda x: sn.assert_eq(
            x.group('pts'), self.nx_opts), result)
        assert_num_parts = sn.map(lambda x: sn.assert_eq(x.group('parts'),
                                                         self.np_opts), result)
        assert_num_steps = sn.map(lambda x: sn.assert_eq(x.group('steps'),
                                                         self.nt_opts), result)

        self.sanity_patterns = sn.all(sn.chain(assert_num_threads,
                                               assert_num_points,
                                               assert_num_parts,
                                               assert_num_steps))


@rfm.simple_test
class Stencil8HPXCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.descr = 'HPX 1d_stencil_8 check'
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc']
        self.valid_prog_environs = ['PrgEnv-gnu']

        self.modules = ['HPX']
        self.executable = '1d_stencil_8'

        self.nt_opts = '100'  # number of time steps
        self.np_opts = '100'  # number of partitions
        self.nx_opts = '10000000'  # number of points per partition
        self.executable_opts = ['--nt', self.nt_opts,
                                '--np', self.np_opts,
                                '--nx', self.nx_opts]
        self.sourcesdir = None

        self.use_multithreading = None

        self.perf_patterns = {
            'time': sn.extractsingle(r'\d+,\s*\d+,\s*(?P<time>(\d+)?.?\d+),'
                                     r'\s*\d+,\s*\d+,\s*\d+',
                                     self.stdout, 'time', float)
        }
        self.reference = {
            'dom:gpu': {
                'time': (26, None, 0.1, 's')
            },
            'dom:mc': {
                'time': (19, None, 0.1, 's')
            },
            'daint:gpu': {
                'time': (26, None, 0.1, 's')
            },
            'daint:mc': {
                'time': (19, None, 0.1, 's')
            },
        }

        self.maintainers = ['VH', 'JG']

    @run_after('setup')
    def set_tasks(self):
        if self.current_partition.fullname == 'daint:gpu':
            self.num_tasks = 2
            self.num_tasks_per_node = 1
            self.num_cpus_per_task = 12
        elif self.current_partition.fullname == 'daint:mc':
            self.num_tasks = 4
            self.num_tasks_per_node = 2
            self.num_cpus_per_task = 18
            self.num_tasks_per_socket = 1
        elif self.current_partition.fullname == 'dom:gpu':
            self.num_tasks = 2
            self.num_tasks_per_node = 1
            self.num_cpus_per_task = 12
        elif self.current_partition.fullname == 'dom:mc':
            self.num_tasks = 4
            self.num_tasks_per_node = 2
            self.num_cpus_per_task = 18
            self.num_tasks_per_socket = 1

    @run_before('run')
    def set_exec_opts(self):
        self.executable_opts += ['--hpx:threads=%s' % self.num_cpus_per_task]

    @run_before('sanity')
    def set_sanity(self):
        result = sn.findall(r'(?P<lid>\d+),\s*(?P<tid>\d+),'
                            r'\s*(?P<time>(\d+)?.?\d+),'
                            r'\s*(?P<pts>\d+),'
                            r'\s*(?P<parts>\d+),'
                            r'\s*(?P<steps>\d+)', self.stdout)
        num_threads = self.num_tasks * self.num_cpus_per_task
        assert_num_tasks = sn.map(
            lambda x: sn.assert_eq(int(x.group('lid')), self.num_tasks),
            result)
        assert_num_threads = sn.map(
            lambda x: sn.assert_eq(int(x.group('tid')), num_threads), result)
        assert_num_points = sn.map(
            lambda x: sn.assert_eq(x.group('pts'), self.nx_opts), result)
        assert_num_parts = sn.map(
            lambda x: sn.assert_eq(x.group('parts'), self.np_opts), result)
        assert_num_steps = sn.map(
            lambda x: sn.assert_eq(x.group('steps'), self.nt_opts), result)

        self.sanity_patterns = sn.all(
            sn.chain(assert_num_tasks, assert_num_threads, assert_num_points,
                     assert_num_parts, assert_num_steps))
