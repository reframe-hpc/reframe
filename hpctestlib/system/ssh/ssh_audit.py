# Copyright 2016-2023 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility as util
import reframe.utility.sanity as sn


@rfm.simple_test
class ssh_audit_check(rfm.RunOnlyRegressionTest):
    '''ssh audit config test.

    `ssh-audit is a tool for ssh server & client configuration auditing.

    The check consist on performing the basic ssh server config auditing
    using the master version of https://github.com/jtesta/ssh-audit.
    '''

    executable = './ssh-audit.py'
    executable_opts = ['-n', '-l', 'fail', 'localhost']
    sourcesdir = 'https://github.com/jtesta/ssh-audit'
    tags = {'system', 'ssh'}

    @sanity_function
    def assert_no_fails_are_found(self):
        '''Assert that no fails are reported by the tool.'''

        return sn.assert_not_found(
                   r'\S+\s+--\s+\[fail\]', self.stdout,
                   msg=(f"found ssh config failures")
               )
