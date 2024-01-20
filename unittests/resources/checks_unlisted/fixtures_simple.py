# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn

import os


class HelloFixture(rfm.RunOnlyRegressionTest):
    executable = 'echo hello from fixture'

    @sanity_function
    def assert_output(self):
        return sn.assert_found(r'hello from fixture', self.stdout)


@rfm.simple_test
class HelloTest(HelloFixture):
    valid_systems = ['*']
    valid_prog_environs = ['*']


@rfm.simple_test
class TestA(rfm.RunOnlyRegressionTest):
    valid_systems = ['*']
    valid_prog_environs = ['*']
    executable = '/bin/true'

    # Declare the fixture
    f = fixture(HelloFixture, scope='session')

    @sanity_function
    def inspect_fixture(self):
        return sn.assert_found(
            r'hello from fixture',
            os.path.join(self.f.stagedir, self.f.stdout.evaluate())
        )


@rfm.simple_test
class TestB(TestA):
    '''Use a test as a fixture'''
    ff = fixture(HelloTest, scope='session')
