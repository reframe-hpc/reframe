# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Functionality to use fixtures in ReFrame tests.
#

import sys
import copy
import itertools
import traceback
from collections.abc import Iterable, Mapping
from hashlib import sha256

import reframe.core.namespaces as namespaces
import reframe.core.runtime as runtime
import reframe.utility.udeps as udeps
import reframe.utility as utils
from reframe.core.exceptions import ReframeSyntaxError, what
from reframe.core.logging import getlogger


class FixtureData:
    '''Store raw data related to a fixture instance.

    The stored data is the fixture class variant number, a list with the valid
    environments, another list with the valid partitions, a dictionary with
    any fixture variables that may be set during the instantiation of the
    fixture class, and the scope used for the fixture.

    This data is required to instantiate the fixture.
    '''

    __slots__ = ('__data',)

    def __init__(self, variant, envs, parts, variables, scope, scope_enc):
        self.__data = (variant, envs, parts, variables, scope, scope_enc)

    @property
    def data(self):
        return self.__data

    @property
    def variant_num(self):
        return self.__data[0]

    @property
    def environments(self):
        return self.__data[1]

    @property
    def partitions(self):
        return self.__data[2]

    @property
    def variables(self):
        return self.__data[3]

    @property
    def scope(self):
        return self.__data[4]

    @property
    def scope_enc(self):
        return self.__data[5]

    def mashup(self):
        s = f'{self.variant_num}/{self.scope_enc}'
        if self.variables:
            s += '/' + '&'.join(f'{k}={self.variables[k]}'
                                for k in sorted(self.variables))

        return sha256(s.encode('utf-8')).hexdigest()[:8]


class FixtureRegistry:
    '''Regression test fixture registry.

    This registry consists of a dictionary where the keys are the fixture
    classes and the values are dictionaries containing raw data that
    identifies each of the registered fixtures to be instantiated from the
    fixture class.

    This registry mangles the fixture name to account for the fixture scope and
    other fixture-specific details that make the fixture unique. This name
    mangling ensures that two different fixtures from the same class do not
    have the same name. In this context, a fixture variant is determined by the
    fixture scope and a 4-element tuple containing the class variant number, a
    list with the valid environments, another list with the valid partitions
    and a key-value pair mapping with any fixture variables that may be set
    during the instantiation of the fixture class.

    Fixtures are added to the registry with the :func:`add` method, and the
    registered fixtures can be instantiated with the :func:`instantiate_all`
    method.

    Since the fixture name is mangled to be unique, a fixture that
    modifies its :attr:`name` attribute may result in an undefined behaviour.

    :meta private:
    '''

    def __init__(self):
        self._registry = dict()

        # Build an index map to access the system partitions
        sys_part = runtime.runtime().system.partitions
        self._env_by_part = {
            p.fullname: {e.name for e in p.environs} for p in sys_part
        }

        # Compact naming switch
        self._hash = runtime.runtime().get_option(
            'general/0/compact_test_names'
        )

        # Store the system name for name-mangling purposes
        self._sys_name = runtime.runtime().system.name

    def add(self, fixture, variant_num, parent_name, partitions, prog_envs):
        '''Register a fixture.

        This method mangles the fixture name, ensuring that different fixture
        variants (see above for the definition of a fixture variant in the
        context of a fixture registry) have a different name. This method
        is aware of the public members from the :class:`TestFixture`.
        Fixtures `steal` the valid environments and valid partitions from
        their parent test, where the number of combinations that trickle
        down into the fixture varies depending on the fixture's scope.
        The rationale is as follows for the different scopes:
         - session: Only one environment+partition combination per fixture.
               This kind of fixture may be shared across all tests. The name
               for fixtures with this scope is mangled with the system name.
               This is necesssary to avoid name conflicts with classes that
               are used both as regular tests and fixtures with session scope.
         - partition: Only one environment per partition. This kind of fixture
               may be shared amongst all the tests running on the same
               partition. The name is mangled to include the partition where
               fixture will run.
         - environment: One fixture per available environment+partition
               combination. This kind of fixture may be shared only with
               other tests that execute on the same environment+partition
               combination. The name is mangled to contain the partition and
               the environment where the fixture will run.
         - test: Use the environments and partitions from the parent test
               without any modifications. Fixtures using this scope are
               private to the parent test and they will not be shared with
               any other test in the session. The fixture name is mangled to
               contain the name of the parent test the fixture belongs to.

        If a fixture modifies the default value of any of its attributes,
        these modified variables are always mangled into the fixture name
        regardless of the scope used. These variables are sorted by their
        name before they're accounted for into the name mangling, so the
        order on which they're specified is irrelevant.

        This method returns a list with the mangled names of the newly
        registered fixtures.

        :param fixture: An instance of :class:`TestFixture`.
        :param variant_num: The variant index for the given ``fixture``.
        :param parent_name: The full name of the parent test. This argument
            is used to mangle the fixture name for those with a ``'test'``
            scope, such that the fixture is private to its parent test.
        :param partitions: The system partitions supported by the parent test.
        :param prog_envs: The valid programming environments from the parent.
        '''

        cls = fixture.cls
        scope = fixture.scope
        fname = fixture.cls.variant_name(variant_num)
        variables = fixture.variables
        reg_names = []
        self._registry.setdefault(cls, dict())

        # Mangle the fixture name with the modified variables (sorted by keys)
        if variables:
            vname = ''.join(
                (f'%{k}={utils.toalphanum(str(v))}' for k, v
                 in sorted(variables.items()))
            )
            if self._hash:
                vname = '_' + sha256(vname.encode('utf-8')).hexdigest()[:8]

            fname += vname

        # Select only the valid partitions
        valid_partitions = self._filter_valid_partitions(partitions)

        # Return if not any valid partition
        if not valid_partitions:
            return []

        # Register the fixture
        if scope == 'session':
            # The name is mangled with the system name
            # Select a valid environment supported by a partition
            for part in valid_partitions:
                valid_envs = self._filter_valid_environs(part, prog_envs)
                if valid_envs:
                    break
            else:
                return []

            # Register the fixture
            fixt_data = FixtureData(variant_num, [valid_envs[0]], [part],
                                    variables, scope, self._sys_name)
            name = f'{cls.__name__}_{fixt_data.mashup()}'
            self._registry[cls][name] = fixt_data
            reg_names.append(name)
        elif scope == 'partition':
            for part in valid_partitions:
                # The mangled name contains the full partition name

                # Select an environment supported by the partition
                valid_envs = self._filter_valid_environs(part, prog_envs)
                if not valid_envs:
                    continue

                # Register the fixture
                fixt_data = FixtureData(variant_num, [valid_envs[0]], [part],
                                        variables, scope, part)
                name = f'{cls.__name__}_{fixt_data.mashup()}'
                self._registry[cls][name] = fixt_data
                reg_names.append(name)
        elif scope == 'environment':
            for part in valid_partitions:
                for env in self._filter_valid_environs(part, prog_envs):
                    # The mangled name contains the full part and env names
                    # Register the fixture
                    fixt_data = FixtureData(variant_num, [env], [part],
                                            variables, scope, f'{part}+{env}')
                    name = f'{cls.__name__}_{fixt_data.mashup()}'
                    self._registry[cls][name] = fixt_data
                    reg_names.append(name)
        elif scope == 'test':
            # The mangled name contains the parent test name.

            # Register the fixture
            fixt_data = FixtureData(variant_num, list(prog_envs),
                                    list(valid_partitions),
                                    variables, scope, parent_name)
            name = f'{cls.__name__}_{fixt_data.mashup()}'
            self._registry[cls][name] = fixt_data
            reg_names.append(name)

        return reg_names

    def update(self, other):
        '''Extend the current registry with the items from another registry.

        In the event of a clash, the elements from ``other`` take precedence.
        Clashes are allowed because they would only happen when two fixtures
        are equivalent (for example, for a fixture with ``'session'`` scope is
        irrelevant which partition or environment is selected to run the
        fixture).
        '''

        self._is_registry(other)
        for cls, variants in other._registry.items():
            self._registry.setdefault(cls, dict())
            for name, args in variants.items():
                self._registry[cls][name] = args

    def difference(self, other):
        '''Build a new registry taking the difference with another registry

        The resulting registry contains the elements from the current registry
        that are not present in ``other``.
        '''

        self._is_registry(other)
        ret = FixtureRegistry()
        for cls, variants in self._registry.items():
            if cls in other:
                other_variants = other._registry[cls]
                for name, args in variants.items():
                    if name not in other_variants:
                        ret._registry.setdefault(cls, dict())
                        ret._registry[cls][name] = args
            else:
                ret._registry[cls] = copy.deepcopy(variants)

        return ret

    def instantiate_all(self):
        '''Instantiate all the fixtures in the registry.'''

        ret = []
        for cls, variants in self._registry.items():
            for name, args in variants.items():
                varnum, penv, part, variables, *_ = args.data

                # Set the fixture name and stolen env and part from the parent,
                # alongside the other variables specified during the fixture's
                # declaration.
                fixtvars = {
                    'valid_prog_environs': penv,
                    'valid_systems': part,
                    **variables
                }

                try:
                    # Instantiate the fixture
                    inst = cls(variant_num=varnum, fixt_name=name,
                               fixt_data=args, fixt_vars=fixtvars)
                except Exception:
                    exc_info = sys.exc_info()
                    getlogger().warning(
                        f"skipping fixture {name!r}: "
                        f"{what(*exc_info)} "
                        f"(rerun with '-v' for more information)"
                    )
                    getlogger().verbose(traceback.format_exc())
                else:
                    ret.append(inst)

        return ret

    def _filter_valid_partitions(self, candidate_parts):
        return [p for p in candidate_parts if p in self._env_by_part]

    def _filter_valid_environs(self, part, candidate_environs):
        ret = []
        environs = self._env_by_part[part]
        for e in candidate_environs:
            if e in environs:
                ret.append(e)

        return ret

    def _is_registry(self, other):
        if not isinstance(other, FixtureRegistry):
            raise TypeError('other is not a FixtureRegistry')

    def __getitem__(self, cls):
        '''Return the sub-dictionary for a given fixture class.

        The keys are the mangled fixture names and the values are the raw data
        to instantiate the fixture class with.
        '''
        return self._registry.get(cls, dict())

    def __contains__(self, cls):
        return cls in self._registry


class TestFixture:
    '''Regression test fixture class.

    A fixture is a regression test that generates a resource that must exist
    before the parent test is executed. A fixture is a class that derives from
    the :class:`reframe.core.pipeline.RegressionTest` class and serves as a
    building block to compose a more complex test structure. Since fixtures are
    full ReFrame tests on their own, a fixture can have multiple fixtures, and
    so on; building a directed acyclic graph.

    However, a given fixture may be shared by multiple regression tests that
    need the same resource. This can be achieved by setting the appropriate
    scope level on which the fixture should be shared. By default, fixtures
    are registered with the ``'test'`` scope, which makes each fixture
    `private` to each of the parent tests. Hence, if all fixtures use this
    scope, the resulting fixture hierarchy can be thought of multiple
    independent branches that emanate from each root regression test. On the
    other hand, setting a more relaxed scope that allows resource sharing
    across different regression tests will effectively interconnect the
    fixture branches that share a resource.

    From a more to less restrictive scope, the valid scopes are ``'test'``,
    ``'environment'``, ``'partition'`` and ``'session'``. Fixtures with
    a scope set to either ``'partition'`` or ``'session'`` must derive from
    the :class:`reframe.core.pipeline.RunOnlyRegressionTest` class, since the
    generated resource must not depend on the programming environment. Fixtures
    with scopes set to either ``'environment'`` or ``'test'`` can derive from
    any derived class from :class:`reframe.core.pipeline.RegressionTest`.

    Fixtures may be parameterized, where a regression test that uses a
    parameterized fixture is by extension a parameterized test. Hence, the
    number of test variants of a test will depend on the test parameters and
    the parameters of each of the fixtures that compose the parent test. Each
    possible parameter-fixture combination has a unique `variant number`, which
    is an index in the range from ``[0, N)``, where `N` is the total number of
    test variants. This is the default behaviour and it is achieved when the
    action argument is set to ``'fork'``. On the other hand, if this argument
    is set to a ``'join'`` action, the parent test will reduce all the fixture
    variants.

    The variants from a given fixture to be used by the parent test can be
    filtered out through the ``variants`` optional argument. This can either be
    a list of the variant numbers to be used, or it can be a dictionary with
    conditions on the parameter space of the fixture.

    Also, a fixture may set or update the default value of a test variable
    by passing the appropriate key-value mapping as the ``variables`` argument.

    :meta private:
    '''

    def __init__(self, cls, *, scope='test', action='fork', variants='all',
                 variables=None):
        # Validate the fixture class: We can't use isinstance here because of
        # circular imports.
        rfm_kind = getattr(cls, '_rfm_regression_class_kind', 0)
        if rfm_kind == 0:
            raise ValueError(
                f"{cls.__qualname__!r} must be a derived class from "
                f"'RegressionTest'"
            )
        elif rfm_kind & 1:
            if scope in {'session', 'partition'}:
                raise ValueError(
                    f'incompatible scope for fixture {cls.__qualname__!r}; '
                    f'scope {scope!r} only supports run-only fixtures.'
                )

        # Check that the fixture class is not an abstract test.
        if cls.is_abstract():
            raise ValueError(
                f'class {cls.__qualname__!r} has undefined parameters'
            )

        # Validate the scope
        if scope not in ('session', 'partition', 'environment', 'test'):
            raise ValueError(
                f'invalid scope for fixture {cls.__qualname__!r}: {scope!r}'
            )

        # Validate the action
        if action not in ('fork', 'join'):
            raise ValueError(
                f'invalid action for fixture {cls.__qualname__!r}: {action!r}'
            )

        self._cls = cls
        self._scope = scope
        self._action = action
        if isinstance(variants, Mapping):
            # If the variants are passed as a mapping, this argument is
            # expected to contain conditions to filter the fixture's
            # parameter space.
            self._variants = tuple(cls.get_variant_nums(**variants))
        elif isinstance(variants, Iterable) and not isinstance(variants, str):
            self._variants = tuple(variants)
        elif variants == 'all':
            self._variants = tuple(range(cls.num_variants))
        else:
            raise ValueError(
                f'invalid variants specified for fixture {cls.__qualname__!r}'
            )

        # Check that we have some variants
        if len(self._variants) == 0:
            raise ValueError('fixture does not have any variants')

        if variables and not isinstance(variables, Mapping):
            raise TypeError(
                "the argument 'variables' must be a mapping."
            )
        elif variables is None:
            variables = {}

        # Store the variables dict
        self._variables = variables

    @property
    def cls(self):
        '''The underlying RegressionTest class.'''
        return self._cls

    @property
    def scope(self):
        '''The fixture scope.'''
        return self._scope

    @property
    def action(self):
        '''Action specified on this fixture.'''
        return self._action

    @property
    def variants(self):
        '''The specified variants for this fixture.'''
        return self._variants

    @property
    def fork_variants(self):
        '''Collection of fixture variant sets that the parent test will fork.

        This function returns a tuple, where each of the elements represents
        a single fork. These elements are iterables containing a set of the
        fixture variant indices that a single fork will reduce. For example,
        for a returned tuple ``((1, 2), (3, 4))``, the parent test using this
        fixture will fork twice, where the first fork will reduce the fixture
        variants #1 and #2, and the second fork will reduce the fixture
        variants #3 and #4.

        With a ``'fork'`` action, the returned tuple will have as many elements
        as specified fixture variants, where each of the elements is a
        one-element tuple containing a unique fixture variant ID (with the
        above example, this is ``((1,), (2,), (3,), (4,))``). With a ``'join'``
        action, the returned tuple will contain a single iterable containing
        all the specified fixture variants (``((1, 2, 3, 4),)``).

        .. note::
          This forking model could be made fully customisable to the user by
          simply allowing to pass a function as the ``action`` argument that
          process the variants in their custom way to generate the tuple of
          iterables. However, this would conceptually overlap with the
          ``variants`` argument, since a user could pass a function that does
          not use the specified fixture variants at all.
        '''
        if self._action == 'join':
            return tuple((self.variants,))
        else:
            return tuple((v,) for v in self.variants)

    @property
    def variables(self):
        '''Variables to be set in the test.'''
        return self._variables


class FixtureSpace(namespaces.Namespace):
    '''Regression test fixture space.

    The fixture space is first built by joining the available fixture spaces
    in the base classes, and later extended by the locally defined fixtures
    that are expected in the local fixture space. Defining fixtures with the
    same name in more than one of the base classes is disallowed. However,
    a fixture defined in a base class can be overridden by a fixture defined
    in the derived class under the same name.

    The fixture injection occurs on an instance of the target class. The
    fixtures are first grouped in a fixture registry, which is then injected
    into the target instance under the ``_rfm_fixture_registry`` attribute.
    '''

    def __init__(self, target_cls=None, illegal_names=None):
        super().__init__(target_cls, illegal_names,
                         ns_name='_rfm_fixture_space',
                         ns_local_name='_rfm_local_fixture_space')

        # Store all fixture variant combinations to allow random access.
        self.__variant_combinations = tuple(
            itertools.product(
                *(list(f.fork_variants for f in self.fixtures.values()))
            )
        )

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
                    f'one base class of {cls.__qualname__!r}'
                )

            self.fixtures[key] = value

    def extend(self, cls):
        '''Extend the inherited fixture space with the local fixture space.'''
        local_fixture_space = getattr(cls, self.local_namespace_name, False)
        while local_fixture_space:
            name, fixture = local_fixture_space.popitem()
            self.fixtures[name] = fixture

        # If any previously declared fixture was defined in the class body
        # by directly assigning it a value, raise an error. Fixtures must be
        # changed using the `x = fixture(...)` syntax.
        for key in self.fixtures:
            if key in cls.__dict__:
                raise ReframeSyntaxError(
                    f'fixture {key!r} can only be redefined through the '
                    f'fixture built-in'
                )

    def inject(self, obj, cls=None, fixtures_index=None):
        '''Build fixture registry and inject it in the parent's test instance.

        A fixture steals the valid_systems and valid_prog_environs from the
        parent tests, and these attributes could be set during the parent
        test's instantiation. Similarly, the fixture registry requires of the
        parent test's full name to build unique IDs for fixtures with the
        ``'test'`` scope (i.e. fixtures private to a parent test).

        :param obj: Parent test's instance.
        :param cls: Parent test's class.
        :param fixtures_index: Index representing a point in the fixture
            space. This point represents a unique variant combination of all
            the fixtures present in the fixture space.

        .. note::
           This function is aware of the implementation of the
           :class:`reframe.core.pipeline.RegressionTest` class.
        '''

        # Nothing to do if the fixture space is empty
        if not self.fixtures or fixtures_index is None:
            return

        # Get the unique fixture variant combiantion (as a k-v map) for
        # the given index.
        try:
            fixture_variants = self[fixtures_index]
        except IndexError:
            raise RuntimeError(
                f'fixture space index out of range for '
                f'{obj.__class__.__qualname__}'
            ) from None

        # Create the fixture registry
        obj._rfm_fixture_registry = FixtureRegistry()

        # Prepare the partitions and prog_envs
        part, prog_envs = self._expand_partitions_envs(obj)

        # Register the fixtures
        for name, fixture in self.fixtures.items():
            dep_names = []
            for variant in fixture_variants[name]:
                # Register all the variants and track the fixture names
                dep_names += obj._rfm_fixture_registry.add(fixture,
                                                           variant,
                                                           obj.name, part,
                                                           prog_envs)

            # Add dependencies
            if fixture.scope == 'session':
                dep_kind = udeps.fully
            elif fixture.scope == 'partition':
                dep_kind = udeps.by_part
            else:
                dep_kind = udeps.by_case

            # Inject the dependency
            for dep_name in dep_names:
                obj.depends_on(dep_name, dep_kind)

    def _expand_partitions_envs(self, obj):
        '''Process the partitions and programming environs of the parent.'''

        try:
            part = tuple(obj.valid_systems)
        except AttributeError:
            raise ReframeSyntaxError(
                f"'valid_systems' is undefined in test {obj.unique_name!r}"
            )
        else:
            rt = runtime.runtime()
            if '*' in part or rt.system.name in part:
                part = tuple(p.fullname for p in rt.system.partitions)

        try:
            prog_envs = tuple(obj.valid_prog_environs)
        except AttributeError:
            raise ReframeSyntaxError(
                f"'valid_prog_environs' is undefined "
                f"in test {obj.unique_name!r}"
            )
        else:
            if '*' in prog_envs:
                all_pes = set()
                for p in runtime.runtime().system.partitions:
                    for e in p.environs:
                        all_pes.add(e.name)

                prog_envs = tuple(all_pes)

        return part, prog_envs

    def __iter__(self):
        '''Walk through all index combinations for all fixtures.'''
        yield from self.__variant_combinations

    def __len__(self):
        if not self.fixtures:
            return 1

        return len(self.__variant_combinations)

    def __getitem__(self, key):
        '''Access an element in the fixture space.

        If the key is an integer, this function will return a mapping with the
        variant numbers of each of the fixtures for the provided point in the
        fixture space. In this case, the fixture must be an index in the range
        of ``[0, len(self))``.
        If the key is just a fixture name, this function will return the
        underlying fixture object with that name.
        '''
        if isinstance(key, int):
            ret = {}
            f_ids = self.__variant_combinations[key]
            for i, f in enumerate(self.fixtures):
                ret[f] = f_ids[i]

            return ret

        return self.fixtures[key]

    @property
    def fixtures(self):
        return self._namespace
