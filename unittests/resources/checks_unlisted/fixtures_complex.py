# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
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
    executable = 'echo'

    # Declare the fixtures
    f0 = fixture(SimpleFixture, scope='session', action='fork')
    f1 = fixture(SimpleFixture, scope='partition', action='fork')
    f2 = fixture(SimpleFixture, scope='environment', action='fork')
    f3 = fixture(SimpleFixture, scope='test', action='fork')
    f4 = fixture(SimpleFixture, scope='session', action='join')

    @sanity_function
    def validate_fixture_resolution(self):
        return sn.all([
            # Access all the fixtures with a fork action.
            sn.assert_eq(
                (self.f0.data + self.f1.data + self.f2.data + self.f3.data), 4
            ),

            # Assert that only one fixture is resolved with join action.
            sn.assert_eq(sn.len(self.f4), 1),

            # Assert that the fixtures with join and fork actions resolve to
            # the same instance for the same scope.
            sn.assert_eq(self.f4[0], self.f0),

            # Assert that there are only 4 underlying fixture instances.
            sn.assert_eq(
                sn.len({self.f0, self.f1, self.f2, self.f3, *self.f4}), 4
            ),

            # Assert is_fixture() function
            sn.assert_true(self.f0.is_fixture()),
            sn.assert_false(self.is_fixture())
        ])


@rfm.simple_test
class TestB(rfm.RunOnlyRegressionTest):
    ''' Test fixture resolution with multiple variants.'''

    valid_systems = ['*']
    valid_prog_environs = ['*']
    executable = 'echo'

    # Declare the fixtures
    f0 = fixture(ParamFixture, variants={'p': lambda x: x == 0})
    f1 = fixture(ParamFixture, variants={'p': lambda x: x == 1})
    f2 = fixture(ParamFixture)
    f3 = fixture(ParamFixture)
    f4 = fixture(ParamFixture, action='join')

    @sanity_function
    def validate_fixture_resolution(self):
        fixt_info = type(self).get_variant_info(self.variant_num)['fixtures']
        return sn.all([
            # Assert that f0 and f1 resolve to the right variants
            sn.all([sn.assert_eq(self.f0.p, 0), sn.assert_eq(self.f1.p, 1)]),

            # Assert the outer product of the fixtures variants is correct even
            # with both fixtures being exactly the same.
            sn.assert_true(self.f2.variant_num in fixt_info['f2']),
            sn.assert_true(self.f3.variant_num in fixt_info['f3']),

            # Assert the join behaviour works correctly
            sn.assert_eq(
                sn.len({f.variant_num for f in self.f4}),
                ParamFixture.num_variants
            )
        ])
