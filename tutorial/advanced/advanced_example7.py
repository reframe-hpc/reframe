# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class PrerunDemoTest(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.descr = ('ReFrame tutorial demonstrating the use of '
                      'pre- and post-run commands')
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.prerun_cmds  = ['source scripts/limits.sh']
        self.postrun_cmds = ['echo FINISHED']
        self.executable = './random_numbers.sh'
        numbers = sn.extractall(
            r'Random: (?P<number>\S+)', self.stdout, 'number', float)
        self.sanity_patterns = sn.all([
            sn.assert_eq(sn.count(numbers), 100),
            sn.all(sn.map(lambda x: sn.assert_bounded(x, 50, 80), numbers)),
            sn.assert_found('FINISHED', self.stdout)
        ])
        self.maintainers = ['put-your-name-here']
        self.tags = {'tutorial'}
