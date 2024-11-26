# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


class HelloFixture(rfm.RunOnlyRegressionTest):
    executable = 'echo hello from fixture'
    myvar = variable(str)
    
    @sanity_function
    def assert_output(self):
        return sn.assert_found(r'hello from fixture', self.stdout)
    

@rfm.simple_test
class TestA(rfm.RunOnlyRegressionTest):
    valid_systems = ['*']
    valid_prog_environs = ['*']

    dep1 = fixture(HelloFixture, scope='environment', variables={'myvar': 'a'})
    dep2 = fixture(HelloFixture, scope='environment', variables={'myvar': 'b'})

    @run_after('setup')
    def after_setup(self):
        self.executable = f"echo {self.dep1.myvar} {self.dep2.myvar}"

    @sanity_function
    def validate(self):
        return sn.assert_found(r'a b', self.stdout)
