# Copyright 2016-2023 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


class dummy_fixture(rfm.RunOnlyRegressionTest, pin_prefix=True):
    executable = 'echo'
    sanity_patterns = sn.assert_true(1)


@rfm.simple_test
class simple_echo_check(rfm.RunOnlyRegressionTest):
    descr = 'Simple Echo Test'
    valid_systems = ['*']
    valid_prog_environs = ['builtin']
    executable = 'echo'
    executable_opts = ['Hello']
    message = variable(str, value='World') 
    dummy = fixture(dummy_fixture, scope='environment')

    @run_before('run')
    def set_executable_opts(self):
        self.executable_opts += [self.message]

    @sanity_function
    def assert_sanity(self):
        return sn.assert_found(rf'Hello {self.message}', self.stdout)
