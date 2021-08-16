# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Functionality to use fixtures in ReFrame tests.
#

import copy
import itertools

import reframe.core.namespaces as namespaces
from reframe.core.exceptions import ReframeSyntaxError

class TestFixture:
    '''Regression test fixture class.

    :meta private:
    '''

    def __init__(self, cls, scope='test'):
        # Can't use isinstance here because of circular deps.
        if not hasattr(cls, 'num_variants'):
            raise ValueError(
                f'{cls.__qualname__!r} must be a derived class from '
                f'{"RegressionTest"!r}'
            ) from None

        self.cls = cls
        self.scope = scope


class FixtureSpace(namespaces.Namespace):
    ''' Regression test fixture space.'''

    @property
    def local_namespace_name(self):
        return '_rfm_local_fixture_space'

    @property
    def namespace_name(self):
        return '_rfm_fixture_space'

    def __init__(self, target_cls=None, target_namespace=None):
        super().__init__(target_cls, target_namespace)

        self.__random_access_iter = [x for x in iter(self)]

    def join(self, other, cls):
        '''Join other fixture spaces into the current one.

        :param other: instance of the FixtureSpace class.
        :param cls: the target class.
        '''
        for key, value in other.fixtures.items():
            if key in self.fixtures:
                raise ReframeSyntaxError(
                    f'fixture space conflict: '
                    f'fixture {key!r} is defined in more than '
                    f'one base class of class {cls.__qualname__!r}'
                )

            self.fixtures[key] = copy.deepcopy(value)

    def extend(self, cls):
        local_fixture_space = getattr(cls, self.local_namespace_name)
        while local_fixture_space:
            name, fixture = local_fixture_space.popitem()
            self.fixtures[name] = fixture

        # If any previously declared fixture was defined in the class body
        # by directly assigning it a value, raise an error. Fixtures must be
        # changed using the `x = fixture(...)` syntax.
        for key, values in cls.__dict__.items():
            if key in self.fixtures:
                raise ReframeSyntaxError(
                    f'fixture {key!r} must be modified through the built-in '
                    f'fixture type'
                )

    def inject(self, obj, cls=None, fixture_index=None):
        if fixture_index and fixture_index >= len(self):
            raise RuntimeError(
                f'fixture index out of range for '
                f'{obj.__class__.__qualname__}'
            )

        try:
            sys = obj.valid_systems
        except AttributeError:
            raise ReframeSyntaxError(
                f'valid_systems is undefined in test {obj.name}'
            )

        try:
            pe = obj.valid_prog_environs
        except AttributeError:
            raise ReframeSyntaxError(
                f'valid_prog_environs is undefined in test {obj.name}'
            )

        # Get the fixture indices
        fixture_idx = self[fixture_index]

        # Register the fixtures
        for name, fixture in self.fixtures.items():
            print(fixture.cls.fullname(fixture_idx[name]))

        # The fixtures MUST be registered in the OBJECT.
        # Extend the loop above to set the sys and pe for each of the
        # fixtures in the fixture registry. This registry must contain
        # an attribute with the full fixture depth, so that fixtures
        # with the 'test' scope can use that as the unique ID.


    def __iter__(self):
        '''Walk through all index combinations for all fixtures.'''
        yield from itertools.product(
            *(list(range(f.cls.num_variants)
            for f in self.fixtures.values()))
        )

    def __len__(self):
        l = 1
        for f in self.fixtures.values():
            l *= f.cls.num_variants

        return l

    def __getitem__(self, key):
        if isinstance(key, int):
            ret = dict()
            f_ids = self.__random_access_iter[key]
            for i,f in enumerate(self.fixtures):
                ret[f] = f_ids[i]

            return ret

        return self.fixtures[key]

    @property
    def fixtures(self):
        return self._namespace
