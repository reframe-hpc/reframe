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
import reframe.core.runtime as runtime
import reframe.utility.udeps as udeps
from reframe.core.exceptions import ReframeSyntaxError


class FixtureRegistry:
    '''Fixture registry.

    This is composed of individual sub-registries for each fixture scope:
     - session: only one pe+sys combination per fixture.
     - partition: up to one pe per partition.
     - environment: union of all valid pes per partition.
     - test: steals the pe+sys from the root.
    '''
    def __init__(self, root):
        # Get the full branch path
        branch = getattr(root, '_rfm_fixture_branch', '')
        self.branch = branch + type(root).fullname(root.test_id)

        # Fire up the sub-registries
        self._registry = dict()
        for scope in {'session', 'partition', 'environment', 'test'}:
            self._registry.setdefault(scope, dict())

        # Get valid systems and partitions from the root test
        try:
            part = tuple(root.valid_systems)
        except AttributeError:
            raise ReframeSyntaxError(
                f'valid_systems is undefined in test {root.name}'
            )
        else:
            rt = runtime.runtime()
            if '*' in part or rt.system.name in part:
                part = tuple(p.fullname for p in rt.system.partitions)
        finally:
            self._partitions = part

        try:
            self._pes = tuple(root.valid_prog_environs)
        except AttributeError:
            raise ReframeSyntaxError(
                f'valid_prog_environs is undefined in test {root.name}'
            )

        # Setup the partition sub-registry
        for p in self._partitions:
            self._registry['partition'].setdefault(p, dict())

        # Setup the environment sub-registry
        rt = runtime.runtime()
        all_pes = set()
        for p in rt.system.partitions:
            pname = p.fullname
            self._registry['environment'].setdefault(pname, dict())
            for e in p.environs:
                ename = e.name
                self._registry['environment'][pname].setdefault(ename, dict())
                all_pes.add(ename)

        # Setup the test sub-registry.
        self._registry['test'].setdefault(self.branch, dict())

        #Finish setup for the valid PEs
        if '*' in self._pes:
            self._pes = list(all_pes)

    def add(self, fixture, fid):
        '''Add a fixture to the registry.

        The classes are the keys and the set with the variant IDs are the values.
        The question is, when do we decide on the sys and pe?
        '''

        test = fixture.test
        scope = fixture.scope
        fname = fixture.get_name(fid)
        reg_names = []
        reg = self._registry[scope]
        if scope == 'session':
            name = fname
            reg.setdefault(test, dict())
            reg[test][fid] = name
            reg_names.append(name)
        elif scope == 'partition':
            for p in self._partitions:
                name = '_'.join([fname, p])
                reg[p].setdefault(test, dict())
                reg[p][test][fid] = name
                reg_names.append(name)
        elif scope == 'environment':
            for p in self._partitions:
                for env in self._pes:
                    if env in reg[p]:
                        name = '_'.join([fname, p, env])
                        reg[p][env].setdefault(test, dict())
                        reg[p][env][test][fid] = name
                        reg_names.append(name)
        elif scope == 'test':
            name = '_'.join([fname, self.branch])
            reg[self.branch].setdefault(test, dict())
            reg[self.branch][test][fid] = name
            reg_names.append(name)

        return reg_names

    def instantiate_all(self):
        '''
        This MUST set the full path in each object (self.branch) and also the valid_systems
        and valid_prog_environs.
        This is needed for the tests with the test scope to get a unique name.
        '''
        return []

    def __repr__(self):
        return repr(self._registry)


class TestFixture:
    '''Regression test fixture class.

    :meta private:
    '''

    def __init__(self, cls, scope='test'):
        # Can't use isinstance here because of circular deps.
        rfm_kind = getattr(cls, '_rfm_regression_class_kind', 0)
        if rfm_kind==0:
            raise ReframeSyntaxError(
                f"{cls.__qualname__!r} must be a derived class from "
                f"'RegressionTest'"
            )
        elif rfm_kind & 1:
            if scope in {'session', 'partition'}:
                raise ReframeSyntaxError(
                    f'incompatible scope for fixture {cls.__qualname__}; '
                    f'scope {scope!r} only supports run-only fixtures.'
                )

        if scope not in {'session', 'partition', 'environment', 'test'}:
            raise ReframeSyntaxError(
                f'invalid scope for fixture {cls.__qualname__} ({scope!r})'
            )

        self._cls = cls
        self._scope = scope

    @property
    def test(self):
        return self._cls

    @property
    def scope(self):
        return self._scope

    def get_name(self, variant_id=None):
        return self.test.fullname(variant_id)


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

            self.fixtures[key] = value

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
        if fixture_index is None or fixture_index >= len(self):
            raise RuntimeError(
                f'fixture index out of range for '
                f'{obj.__class__.__qualname__}'
            )

        # Nothing to do if the fixture space is empty
        if not self.fixtures:
            return

        # Create the fixture registry
        # The instantiate_all method from this registry should
        # inject an attribute in the object with the full path
        # of the host class' object.
        obj._rfm_fixture_registry = FixtureRegistry(obj)

        # Get the fixture indices
        fixture_idx = self[fixture_index]

        # Register the fixtures
        for name, fixture in self.fixtures.items():
            fid = fixture_idx[name]
            print(fixture.get_name(fid))

            dep_names = obj._rfm_fixture_registry.add(fixture, fid)

            # Add dependencies
            if fixture.scope == 'session':
                dep_mode = udeps.fully
            elif fixture.scope == 'partition':
                dep_mode = udeps.by_part
            elif fixture.scope == 'environment':
                dep_mode = udeps.by_env
            else:
                dep_mode = udeps.by_case

            # Inject the dependency
#            for name in dep_names:
#                obj.depends_on(name, dep_mode)


        # The fixtures MUST be registered in the OBJECT.
        # Extend the loop above to set the sys and pe for each of the
        # fixtures in the fixture registry. This registry must contain
        # an attribute with the full fixture depth, so that fixtures
        # with the 'test' scope can use that as the unique ID.


    def __iter__(self):
        '''Walk through all index combinations for all fixtures.'''
        yield from itertools.product(
            *(list(range(f.test.num_variants)
            for f in self.fixtures.values()))
        )

    def __len__(self):
        l = 1
        for f in self.fixtures.values():
            l *= f.test.num_variants

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
