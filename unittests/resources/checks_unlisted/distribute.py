# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn

import os


@rfm.simple_test
class Simple(rfm.RunOnlyRegressionTest):
    valid_systems = ['*']
    valid_prog_environs = ['*']
    executable = 'true'
    sanity_patterns = sn.assert_true(1)


class MyFixture(rfm.RunOnlyRegressionTest):
    x = parameter([1, 2])
    executable = 'echo hello from fixture'

    @sanity_function
    def assert_True(self):
        return True


@rfm.simple_test
class Complex(rfm.RunOnlyRegressionTest):
    valid_systems = ['*']
    valid_prog_environs = ['*']
    f = fixture(MyFixture, scope='session')
    executable = 'true'

    @sanity_function
    def inspect_fixture(self):
        return sn.assert_found(
            r'hello from fixture',
            os.path.join(self.f.stagedir, self.f.stdout.evaluate())
        )
