# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class ParameterizedEchoTest(rfm.RunOnlyRegressionTest):
    descr = 'Simple parameterized echo test'
    message = parameter(['foo', 'bar', 'baz'])

    # All available systems are supported
    valid_systems = ['*']
    valid_prog_environs = ['*']
    executable = 'echo'
    executable_opts = ['Hello,']
    tags = {'foo', 'bar', 'baz'}
    maintainers = ['TM']

    @run_before('run')
    def set_exec_options(self):
        self.executable_opts.append(self.message)

    @sanity_function
    def validate(self):
        return sn.assert_found(rf'Hello, {self.message}', self.stdout)
