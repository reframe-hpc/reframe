# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class HostnameCheck(rfm.RunOnlyRegressionTest):
    valid_systems = ['daint:gpu', 'daint:mc']
    valid_prog_environs = ['cray']
    executable = 'hostname'
    num_tasks = 0
    num_tasks_per_node = 1

    @sanity_function
    def validate_test(self):
        return sn.assert_eq(
            self.num_tasks,
            sn.count(sn.findall(r'^nid\d+$', self.stdout))
        )
