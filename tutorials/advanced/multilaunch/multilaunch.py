# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class MultiLaunchTest(rfm.RunOnlyRegressionTest):
    valid_systems = ['daint:gpu', 'daint:mc']
    valid_prog_environs = ['builtin']
    executable = 'hostname'
    num_tasks = 4
    num_tasks_per_node = 1

    @run_before('run')
    def pre_launch(self):
        cmd = self.job.launcher.run_command(self.job)
        self.prerun_cmds = [
            f'{cmd} -n {n} {self.executable}'
            for n in range(1, self.num_tasks)
        ]

    @sanity_function
    def validate_test(self):
        return sn.assert_eq(
            sn.count(sn.extractall(r'^nid\d+', self.stdout)), 10
        )
