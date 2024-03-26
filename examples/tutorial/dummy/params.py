# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class echo_test_v0(rfm.RunOnlyRegressionTest):
    valid_systems = ['*']
    valid_prog_environs = ['*']
    executable = 'echo'
    x = parameter([0, 1])
    y = parameter([0, 1])

    @run_after('init')
    def skip_invalid(self):
        self.skip_if(self.x == self.y, 'invalid parameter combination')

    @run_after('init')
    def set_executable_opts(self):
        self.executable_opts = [f'{self.x}', f'{self.y}']

    @sanity_function
    def validate(self):
        x = sn.extractsingle(r'(\d) (\d)', self.stdout, 1, int)
        y = sn.extractsingle(r'(\d) (\d)', self.stdout, 2, int)
        return sn.and_(sn.assert_eq(x, self.x), sn.assert_eq(y, self.y))


@rfm.simple_test
class echo_test_v1(rfm.RunOnlyRegressionTest):
    valid_systems = ['*']
    valid_prog_environs = ['*']
    executable = 'echo'
    xy = parameter([(0, 1), (1, 0)], fmt=lambda val: f'{val[0]}{val[1]}')

    @run_after('init')
    def set_executable_opts(self):
        self.x, self.y = self.xy
        self.executable_opts = [f'{self.x}', f'{self.y}']

    @sanity_function
    def validate(self):
        x = sn.extractsingle(r'(\d) (\d)', self.stdout, 1, int)
        y = sn.extractsingle(r'(\d) (\d)', self.stdout, 2, int)
        return sn.and_(sn.assert_eq(x, self.x), sn.assert_eq(y, self.y))
