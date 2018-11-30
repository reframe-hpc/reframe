import itertools
import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class Stencil4HPXCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        super().__init__()

        self.descr = 'HPX 1d_stencil_4 check'
        self.valid_systems = ['daint:gpu, daint:mc', 'dom:gpu', 'dom:mc']
        self.valid_prog_environs = ['PrgEnv-gnu']

        self.modules = ['HPX']
        self.executable = '1d_stencil_4'

        self.nt_opts = '100'
        self.np_opts = '100'
        self.nx_opts = '10000000'
        self.executable_opts = ['--nt', self.nt_opts,
                                '--np', self.np_opts,
                                '--nx', self.nx_opts]
        self.sourcesdir = None

        self.use_multithreading = None

        self.perf_patterns = {
            'perf': sn.extractsingle(r'\d+,\s*(?P<perf>\d+.\d+),\s*\d+,'
                                     r'\s*\d+,\s*\d+',
                                     self.stdout, 'perf', float)
        }
        self.reference = {
            'dom:gpu': {
                'perf': (42, None, 0.1, 's')
            },
            'dom:mc': {
                'perf': (30, None, 0.1, 's')
            },
            'daint:gpu': {
                'perf': (42, None, 0.1, 's')
            },
            'daint:mc': {
                'perf': (30, None, 0.1, 's')
            },
        }

        self.tags = {'production'}
        self.maintainers = ['VH', 'JG']

    def setup(self, partition, environ, **job_opts):
        result = sn.findall(r'(\d+),\s*(\d+.\d+),\s*(\d+),\s*(\d+),\s*(\d+)',
                            self.stdout)

        if partition.fullname == 'daint:gpu':
            self.num_tasks = 1
            self.num_tasks_per_node = 1
            self.num_cpus_per_task = 12
        elif partition.fullname == 'daint:mc':
            self.num_tasks = 1
            self.num_tasks_per_node = 1
            self.num_cpus_per_task = 36
        elif partition.fullname == 'dom:gpu':
            self.num_tasks = 1
            self.num_tasks_per_node = 1
            self.num_cpus_per_task = 12
        elif partition.fullname == 'dom:mc':
            self.num_tasks = 1
            self.num_tasks_per_node = 1
            self.num_cpus_per_task = 36
        self.executable_opts += ['--hpx:threads=%s' % self.num_cpus_per_task]

        self.sanity_patterns = sn.all(
            sn.chain(sn.map(
                         lambda x: sn.assert_eq(int(x.group(1)),
                                                self.num_cpus_per_task),
                         result),
                     sn.map(
                         lambda x: sn.assert_eq(x.group(3),
                                                self.nx_opts),
                         result),
                     sn.map(
                         lambda x: sn.assert_eq(x.group(4),
                                                self.np_opts),
                         result),
                     sn.map(
                         lambda x: sn.assert_eq(x.group(5),
                                                self.nt_opts),
                         result),
                     )
        )
        super().setup(partition, environ, **job_opts)

@rfm.simple_test
class Stencil8HPXCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        super().__init__()

        self.descr = 'HPX 1d_stencil_8 check'
        self.valid_systems = ['daint:gpu, daint:mc', 'dom:gpu', 'dom:mc']
        self.valid_prog_environs = ['PrgEnv-gnu']

        self.modules = ['HPX']
        self.executable = '1d_stencil_8'

        self.nt_opts = '100'
        self.np_opts = '100'
        self.nx_opts = '10000000'
        self.executable_opts = ['--nt', self.nt_opts,
                                '--np', self.np_opts,
                                '--nx', self.nx_opts]
        self.sourcesdir = None

        self.use_multithreading = None

        self.perf_patterns = {
            'perf': sn.extractsingle(r'\d+,\s*\d+,\s*(?P<perf>\d+.\d+),'
                                     r'\s*\d+,\s*\d+,\s*\d+',
                                     self.stdout, 'perf', float)
        }
        self.reference = {
            'dom:gpu': {
                'perf': (26, None, 0.1, 's')
            },
            'dom:mc': {
                'perf': (19, None, 0.1, 's')
            },
            'daint:gpu': {
                'perf': (26, None, 0.1, 's')
            },
            'daint:mc': {
                'perf': (19, None, 0.1, 's')
            },
        }

        self.tags = {'production'}
        self.maintainers = ['VH', 'JG']

    def setup(self, partition, environ, **job_opts):
        result = sn.findall(r'(\d+),\s*(\d+),\s*(\d+.\d+),\s*(\d+),'
                            r'\s*(\d+),\s*(\d+)', self.stdout)

        if partition.fullname == 'daint:gpu':
            self.num_tasks = 2
            self.num_tasks_per_node = 1
            self.num_cpus_per_task = 12
        elif partition.fullname == 'daint:mc':
            self.num_tasks = 4
            self.num_tasks_per_node = 2
            self.num_cpus_per_task = 18
            self.num_tasks_per_socket = 1
        elif partition.fullname == 'dom:gpu':
            self.num_tasks = 2
            self.num_tasks_per_node = 1
            self.num_cpus_per_task = 12
        elif partition.fullname == 'dom:mc':
            self.num_tasks = 4
            self.num_tasks_per_node = 2
            self.num_cpus_per_task = 18
            self.num_tasks_per_socket = 1
        self.executable_opts += ['--hpx:threads=%s' % self.num_cpus_per_task]

        num_threads = self.num_tasks * self.num_cpus_per_task
        self.sanity_patterns = sn.all(
            sn.chain(sn.map(
                         lambda x: sn.assert_eq(int(x.group(1)),
                                                self.num_tasks),
                         result),
                     sn.map(
                         lambda x: sn.assert_eq(int(x.group(2)),
                                                num_threads),
                         result),
                     sn.map(
                         lambda x: sn.assert_eq(x.group(4),
                                                self.nx_opts),
                         result),
                     sn.map(
                         lambda x: sn.assert_eq(x.group(5),
                                                self.np_opts),
                         result),
                     sn.map(
                         lambda x: sn.assert_eq(x.group(6),
                                                self.nt_opts),
                         result),
                     )
        )
        super().setup(partition, environ, **job_opts)
