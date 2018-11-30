import itertools
import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class HelloWorldHPXCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        super().__init__()

        self.descr = 'HPX hello, world check'
        self.valid_systems = ['daint:gpu, daint:mc', 'dom:gpu', 'dom:mc']
        self.valid_prog_environs = ['PrgEnv-gnu']

        self.modules = ['HPX']
        self.executable = 'hello_world'
        self.sourcesdir = None

        self.use_multithreading = None

        self.tags = {'production'}
        self.maintainers = ['VH', 'JG']

    def setup(self, partition, environ, **job_opts):
        result = sn.findall(r'hello world from OS-thread \s*(\d+) on '
                            r'locality (\d+)', self.stdout)

        if partition.fullname == 'daint:gpu':
            self.num_tasks = 2
            self.num_tasks_per_node = 1
            self.num_cpus_per_task = 12
        elif partition.fullname == 'daint:mc':
            self.num_tasks = 2
            self.num_tasks_per_node = 1
            self.num_cpus_per_task = 36
        elif partition.fullname == 'dom:gpu':
            self.num_tasks = 2
            self.num_tasks_per_node = 1
            self.num_cpus_per_task = 12
        elif partition.fullname == 'dom:mc':
            self.num_tasks = 2
            self.num_tasks_per_node = 1
            self.num_cpus_per_task = 36
        self.executable_opts = ['--hpx:threads=%s' % self.num_cpus_per_task]

        num_localities = self.num_tasks // self.num_tasks_per_node
        self.sanity_patterns = sn.all(
            sn.chain([sn.assert_eq(sn.count(result), self.num_tasks *
                                   self.num_cpus_per_task)],
                     sn.map(
                         lambda x: sn.assert_lt(int(x.group(1)),
                                                self.num_cpus_per_task),
                         result),
                     sn.map(
                         lambda x: sn.assert_lt(int(x.group(2)),
                                                num_localities),
                         result),
                     )
        )
        super().setup(partition, environ, **job_opts)
