# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class EchoRandTest(rfm.RunOnlyRegressionTest):
    descr = 'A simple test that echoes a random number'
    valid_systems = ['*']
    valid_prog_environs = ['*']
    lower = variable(int, value=90)
    upper = variable(int, value=100)
    executable = 'echo'
    executable_opts = [
        'Random: ',
        f'$((RANDOM%({upper}+1-{lower})+{lower}))'
    ]

    @rfm.run_before('sanity')
    def set_sanity_patterns(self):
        self.sanity_patterns = sn.assert_bounded(
            sn.extractsingle(
                r'Random: (?P<number>\S+)', self.stdout, 'number', float
            ),
            self.lower, self.upper
        )
