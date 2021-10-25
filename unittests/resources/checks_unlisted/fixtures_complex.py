# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


class SimpleFixture(rfm.RunOnlyRegressionTest):
    executable = 'echo hello fixture'
    data = variable(int, value=1)

    @sanity_function
    def assert_output(self):
        return sn.assert_found(r'hello fixture', self.stdout)


class ParamFixture(SimpleFixture):
    p = parameter(range(2))


@rfm.simple_test
class TestA(rfm.RunOnlyRegressionTest):
    ''' Test fixture resolution for multiple scopes.

    Fixture ``f4`` is declared with a join action and must resolve to the same
    fixture instance as ``f0`` since they both have the same scope. However,
    since ``f4`` has a join scope, the handle points to a list containing this
    single fixture instance.

    Different scopes lead to difference fixture instances.
    '''

    valid_systems = ['*']
    valid_prog_environs = ['*']
    executable = '/bin/true'

    # Declare the fixtures
    f0 = fixture(SimpleFixture, scope='session', action='fork')
    f1 = fixture(SimpleFixture, scope='partition', action='fork')
    f2 = fixture(SimpleFixture, scope='environment', action='fork')
    f3 = fixture(SimpleFixture, scope='test', action='fork')
    f4 = fixture(SimpleFixture, scope='session', action='join')

    @sanity_function
    def validate_fixture_resolution(self):
        # Access all the fixtures with a fork action.
        if (self.f0.data + self.f1.data + self.f2.data + self.f3.data) != 4:
            return False

        # Assert that only one fixture is resolved with join action.
        if len(self.f4) != 1:
            return False

        # Assert that the fixtures with join and fork actions resolve to the
        # same instance for the same scope.
        if self.f4[0] is not self.f0:
            return False

        # Assert that there are only 4 underlying fixture instances.
        if len({self.f0, self.f1, self.f2, self.f3, *self.f4}) != 4:
            return False

        return True


@rfm.simple_test
class TestB(rfm.RunOnlyRegressionTest):
    ''' Test fixture resolution with multiple variants.'''

    valid_systems = ['*']
    valid_prog_environs = ['*']
    executable = '/bin/true'

    # Declare the fixtures
    f0 = fixture(ParamFixture, variants={'p': lambda x: x==0})
    f1 = fixture(ParamFixture, variants={'p': lambda x: x==1})
    f2 = fixture(ParamFixture)
    f3 = fixture(ParamFixture)
    f4 = fixture(ParamFixture, action='join')

    @sanity_function
    def validate_fixture_resolution(self):
        # Assert that f0 and f1 resolve to the right variants
        if not (self.f0.p == 0 and self.f1.p == 1):
            return False

        # Assert the outer product of the fixtures variants is correct even
        # with both fixtures being exactly the same.
        fixt_info = type(self).get_variant_info(self.variant_num)['fixtures']
        if self.f2.variant_num not in fixt_info['f2']:
            return False
        elif self.f3.variant_num not in fixt_info['f3']:
            return False

        # Assert the join behaviour works correctly
        if len({f.variant_num for f in self.f4}) != ParamFixture.num_variants:
            return False

        return True
