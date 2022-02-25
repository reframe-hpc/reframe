# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class HelloWorldHPXCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.descr = 'HPX hello, world check'
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc']
        self.valid_prog_environs = ['builtin']

        self.modules = ['HPX']
        self.executable = 'hello_world_distributed'
        self.sourcesdir = None

        self.use_multithreading = None

        self.maintainers = ['VH', 'JG']

    @run_after('setup')
    def set_tasks(self):
        if self.current_partition.fullname == 'daint:gpu':
            self.num_tasks = 2
            self.num_tasks_per_node = 1
            self.num_cpus_per_task = 12
        elif self.current_partition.fullname == 'daint:mc':
            self.num_tasks = 2
            self.num_tasks_per_node = 1
            self.num_cpus_per_task = 36
        elif self.current_partition.fullname == 'dom:gpu':
            self.num_tasks = 2
            self.num_tasks_per_node = 1
            self.num_cpus_per_task = 12
        elif self.current_partition.fullname == 'dom:mc':
            self.num_tasks = 2
            self.num_tasks_per_node = 1
            self.num_cpus_per_task = 36

    @run_before('run')
    def set_exec_opts(self):
        self.executable_opts = ['--hpx:threads=%s' % self.num_cpus_per_task]

    @run_before('sanity')
    def set_sanity(self):
        hellos = sn.findall(r'hello world from OS-thread \s*(?P<tid>\d+) on '
                            r'locality (?P<lid>\d+)', self.stdout)
        # https://stellar-group.github.io/hpx/docs/sphinx/branches/master/html/terminology.html#term-locality
        num_localities = self.num_tasks // self.num_tasks_per_node
        assert_num_tasks = sn.assert_eq(sn.count(hellos),
                                        self.num_tasks*self.num_cpus_per_task)
        assert_threads = sn.map(lambda x: sn.assert_lt(int(x.group('tid')),
                                self.num_cpus_per_task), hellos)
        assert_localities = sn.map(lambda x: sn.assert_lt(int(x.group('lid')),
                                   num_localities), hellos)

        self.sanity_patterns = sn.all(sn.chain([assert_num_tasks],
                                               assert_threads,
                                               assert_localities))
