# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class PrepostRunTest(rfm.RunOnlyRegressionTest):
    descr = 'Pre- and post-run demo test'
    valid_systems = ['*']
    valid_prog_environs = ['*']
    prerun_cmds = ['source limits.sh']
    postrun_cmds = ['echo FINISHED']
    executable = './random_numbers.sh'

    @rfm.run_before('sanity')
    def set_sanity_patterns(self):
        numbers = sn.extractall(
            r'Random: (?P<number>\S+)', self.stdout, 'number', float
        )
        self.sanity_patterns = sn.all([
            sn.assert_eq(sn.count(numbers), 100),
            sn.all(sn.map(lambda x: sn.assert_bounded(x, 90, 100), numbers)),
            sn.assert_found(r'FINISHED', self.stdout)
        ])
