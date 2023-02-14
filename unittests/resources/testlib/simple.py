# Copyright 2016-2023 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn


class simple_echo(rfm.RunOnlyRegressionTest, pin_prefix=True):
    descr = 'Simple Echo build fixture'
    executable = 'echo'
    executable_opts = ['Hello']

    @sanity_function
    def assert_success(self):
        return sn.assert_found(r'Hello', self.stdout)


@rfm.simple_test
class simple_echo_check(rfm.RunOnlyRegressionTest):
    descr = 'Simple Echo Test'
    valid_systems = ['*']
    valid_prog_environs = ['builtin']
    executable = 'echo'
    message = variable(str, value='World') 
    hello_output = fixture(simple_echo, scope='environment')

    @run_before('run')
    def add_exec_prefix(self):
        fixture_output = os.path.join(self.hello_output.stagedir,
                                      str(self.hello_output.stdout))
        self.executable_opts = [f'$(cat {fixture_output})', self.message]

    @sanity_function
    def assert_sanity(self):
        return sn.assert_found(rf'Hello {self.message}', self.stdout)
