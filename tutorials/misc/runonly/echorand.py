# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class EchoRandTest(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.descr = 'A simple test that echoes a random number'
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        lower = 90
        upper = 100
        self.executable = 'echo'
        self.executable_opts = [
            'Random: ',
            f'$((RANDOM%({upper}+1-{lower})+{lower}))'
        ]
        self.sanity_patterns = sn.assert_bounded(
            sn.extractsingle(
                r'Random: (?P<number>\S+)', self.stdout, 'number', float
            ),
            lower, upper
        )
