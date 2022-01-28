# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class AccountingCommandCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.valid_systems = ['daint:login', 'dom:login']
        self.descr = 'Slurm CSCS usertools accounting'
        self.executable = 'accounting'
        self.valid_prog_environs = ['builtin']
        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.tags = {'cscs_usertools', 'production',
                     'maintenance', 'single-node', 'ops'}
        self.sanity_patterns = sn.assert_found(
            r'Per-project usage at CSCS since', self.stdout)
        self.maintainers = ['VH', 'TM']
