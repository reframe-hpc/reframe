# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
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

import reframe.core.namespaces as namespaces
import reframe.core.runtime as runtime
import reframe.utility.udeps as udeps
import reframe.utility as utils
from reframe.core.exceptions import ReframeSyntaxError, what
from reframe.core.variables import Undefined
from reframe.core.logging import getlogger


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
        self._reg = dict()

        # Build an index map to access the system partitions
        sys_part = runtime.runtime().system.partitions
        self._part_map = {p.fullname: i for i, p in enumerate(sys_part)}

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
               for fixtures with this scope is not mangled by this registry.
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
        regardless of the scope used.

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
        fname = fixture.get_name(variant_num)
        variables = fixture.variables
        reg_names = []
        self._reg.setdefault(cls, dict())

        # Mangle the fixture name with the modified variables
        fname += ''.join(
            (f'%{k}={utils.toalphanum(str(v))}' for k, v
             in variables.items())
        )

        # Select only the valid partitions
        valid_partitions = self._filter_valid_partitions(partitions)

        # Register the fixture
        if scope == 'session':
            # The name is just the class name
            name = fname

            # Select an environment supported by a partition
            valid_envs = self._filter_valid_environs(valid_partitions[0],
                                                     prog_envs)

            # Register the fixture
            self._reg[cls][name] = (
                variant_num, [valid_envs[0]], [valid_partitions[0]], variables
            )
            reg_names.append(name)
        elif scope == 'partition':
            for p in valid_partitions:
                # The mangled name contains the full partition name
                name = '~'.join([fname, p])

                # Select an environment supported by the partition
                valid_envs = self._filter_valid_environs(p, prog_envs)

                # Register the fixture
                self._reg[cls][name] = (
                    variant_num, [valid_envs[0]], [p], variables
                )
                reg_names.append(name)
        elif scope == 'environment':
            for p in valid_partitions:
                for env in self._filter_valid_environs(p, prog_envs):
                    # The mangled name contains the full part and env names
                    name = '~'.join([fname, '+'.join([p, env])])

                    # Register the fixture
                    self._reg[cls][name] = (
                        variant_num, [env], [p], variables
                    )
                    reg_names.append(name)
        elif scope == 'test':
            # The mangled name contains the parent test name.
            name = '~'.join([fname, parent_name])

            # Register the fixture
            self._reg[cls][name] = (
                variant_num, list(prog_envs), list(valid_partitions),
                variables
            )
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
        for cls, variants in other._reg.items():
            self._reg.setdefault(cls, dict())
            for name, args in variants.items():
                self._reg[cls][name] = args

    def difference(self, other):
        '''Build a new registry taking the difference with another registry

        The resulting registry contains the elements from the current registry
        that are not present in ``other``.
        '''

        self._is_registry(other)
        ret = FixtureRegistry()
        for cls, variants in self._reg.items():
            if cls in other:
                other_variants = other._reg[cls]
                for name, args in variants.items():
                    if name not in other_variants:
                        ret._reg.setdefault(cls, dict())
                        ret._reg[cls][name] = args
            else:
                ret._reg[cls] = copy.deepcopy(variants)

        return ret

    def instantiate_all(self):
        '''Instantiate all the fixtures in the registry.'''

        ret = []
        for cls, variants in self._reg.items():
            for name, args in variants.items():
                varnum, penv, part, variables = args

                # Set the fixture name and stolen env and part from the parent
                cls.name = name
                cls.valid_prog_environs = penv
                cls.valid_systems = part

                # Retrieve the variable defautls
                var_def = dict()
                for key, value in variables.items():
                    if key in cls.var_space:
                        try:
                            var_def[key] = getattr(cls, key).default_value
                        except ValueError:
                            var_def[key] = Undefined
                        finally:
                            cls.setvar(key, value)

                try:
                    # Instantiate the fixture
                    inst = cls(variant_num=varnum)
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

                # Reset cls defaults and append instance
                cls.name = Undefined
                cls.valid_prog_environs = Undefined
                cls.valid_systems = Undefined

                # Reinstate the deault values
                for k, v in var_def.items():
                    cls.setvar(k, v)
        return ret

    def _filter_valid_partitions(self, candidate_parts):
        return [p for p in candidate_parts if p in self._part_map]

    def _filter_valid_environs(self, part, candidate_environs):
        sys_part = runtime.runtime().system.partitions
        supported_envs = {
            env.name for env
            in sys_part[self._part_map[part]].environs
        }
        valid_envs = []
        for env in candidate_environs:
            if env in supported_envs:
                valid_envs.append(env)

        return valid_envs

    def _is_registry(self, other):
        if not isinstance(other, FixtureRegistry):
            raise TypeError('other is not a FixtureRegistry')

    def __getitem__(self, cls):
        '''Return the names of all registered fixtures from a given class.'''

        try:
            return self._reg[cls]
        except KeyError:
            return []

    def __contains__(self, cls):
        return cls in self._reg


class TestFixture:
    '''Regression test fixture class.

    A fixture is a regression test that generates a resource that must exist
    before the parent test is executed. A fixture is a class that derives from
    the :class:`reframe.core.pipeline.RegressionTest` class and serves as a
    building block to compose a more complex test structure. Since fixtures are
    full ReFrame tests on their own, a fixture can have multiple fixtures, and
    so on; building a tree-like structure.

    However, a given fixture may be shared by multiple regression tests that
    need the same resource. This can be achieved by setting the appropriate
    scope level on which the fixture should be shared. By default, fixtures
    are registered with the ``'test'`` scope, which makes each fixture
    `private` to each of the parent tests. Hence, if all fixtures use this
    scope, the resulting fixture hierarchy can be thought of multiple
    independent trees that emanate from each root regression test. On the other
    hand, setting a more relaxed scope that allows resource sharing across
    different regression tests will effectively interconnect the fixture trees
    that share a resource.

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
    possible parameter-fixture combination has a unique ``variant_num``, which
    is an index in the range from ``[0, cls.num_variants)``. This is the
    default behaviour and it is achieved when the action argument is set to
    ``'fork'``. On the other hand, if this argument is set to a ``'join'``
    action, the parent test will reduce all the fixture variants.

    The variants from a given fixture to be used by the parent test can be
    filtered out through the variants optional argument. This can either be
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
                f'class {cls.__qualname__!r} is has undefined parameters'
            )

        # Validate the scope
        if scope not in {'session', 'partition', 'environment', 'test'}:
            raise ValueError(
                f'invalid scope for fixture {cls.__qualname__!r} ({scope!r})'
            )

        # Validate the action
        if action not in {'fork', 'join'}:
            raise ValueError(
                f'invalid action for fixture {cls.__qualname__!r} (action!r)'
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
                f'invalid variants specified for fixture {cls.__qualname__}'
            )

        if variables and not isinstance(variables, Mapping):
            raise ValueError(
                "the argument 'variables' must be a mapping."
            )
        elif variables is None:
            variables = {}

        self._variables = variables

    @property
    def cls(self):
        '''The underlying RegressionTest class.'''
        return self._cls

    @property
    def scope(self):
        '''The fixture scope.'''
        return self._scope

    def get_name(self, variant_num=None):
        '''Utility to retrieve the full name of a given fixture variant.'''
        return self.cls.fullname(variant_num)

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
        '''The list of fixture variants that will fork the parent test.

        If the fixture action was set to ``'fork'``, the fork variants match
        the fixture variants. Thus, parameterizing a fixture is effectively
        a parameterisation of the parent test. On the other hand, if the
        fixture was specified a ``'join'`` action, the fixture variants will
        not translate into more variants (forks) of the parent test, and this
        parent test will instead gather all the fixture variants under the same
        instance. To achieve this special behavior, the list of fork variants
        is set to ``[None]``.
        '''
        if self._action == 'join':
            return [None]
        else:
            return self.variants

    @property
    def variables(self):
        '''Variables to be set in the test.'''
        return self._variables


class FixtureSpace(namespaces.Namespace):
    ''' Regression test fixture space.

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

    @property
    def local_namespace_name(self):
        return '_rfm_local_fixture_space'

    @property
    def namespace_name(self):
        return '_rfm_fixture_space'

    def __init__(self, target_cls=None, target_namespace=None):
        super().__init__(target_cls, target_namespace)

        self.__random_access_iter = tuple(
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
                    f'one base class of class {cls.__qualname__!r}'
                )

            self.fixtures[key] = value

    def extend(self, cls):
        '''Extend the inherited fixture space with the local fixture space.'''
        local_fixture_space = getattr(cls, self.local_namespace_name)
        while local_fixture_space:
            name, fixture = local_fixture_space.popitem()
            self.fixtures[name] = fixture

        # If any previously declared fixture was defined in the class body
        # by directly assigning it a value, raise an error. Fixtures must be
        # changed using the `x = fixture(...)` syntax.
        for key in self.fixtures:
            if key in cls.__dict__:
                raise ReframeSyntaxError(
                    f'fixture {key!r} must be modified through the built-in '
                    f'fixture type'
                )

    def inject(self, obj, cls=None, fixture_variant=None):
        '''Build fixture registry and inject it in the parent's test instance.

        A fixture steals the valid_systems and valid_prog_environs from the
        parent tests, and these attributes could be set during the parent
        test's instantiation. Similarly, the fixture registry requires of the
        parent test's full name to build unique IDs for fixtures with the
        ``'test'`` scope (i.e. fixtures private to a parent test).

        :param obj: Parent test's instance.
        :param cls: Parent test's class.
        :param fixture_variant: Index representing a point in the fixture
            space.

        .. note::
           This function is aware of the implementation of the
           :class:`reframe.core.pipeline.RegressionTest` class.
        '''

        # Nothing to do if the fixture space is empty
        if not self.fixtures or fixture_variant is None:
            return

        # Create the fixture registry
        obj._rfm_fixture_registry = FixtureRegistry()

        # Prepare the partitions and prog_envs
        part, prog_envs = self._get_partitions_and_prog_envs(obj)

        # Get the variant numbers for each of the fixtures (as a k-v map) for
        # the given point in the fixture space.
        fixture_variant_num_map = self[fixture_variant]

        # Register the fixtures
        for name, fixture in self.fixtures.items():
            var_num = fixture_variant_num_map[name]

            # Handle the 'fork' and 'join' actions:
            # var_num is None when the fixture has a 'join' action. Otherwise
            # var_num is a nonnegative integer.
            if var_num is None:
                var_num = fixture.variants
            else:
                var_num = [var_num]

            dep_names = []
            for variant in var_num:
                try:
                    # Register all the variants and track the fixture names
                    dep_names += obj._rfm_fixture_registry.add(fixture,
                                                               variant,
                                                               obj.name, part,
                                                               prog_envs)
                except Exception:
                    exc_info = sys.exc_info()
                    getlogger().warning(
                        f"skipping fixture {fixture.cls.__qualname__!r}: "
                        f"{what(*exc_info)} "
                        f"(rerun with '-v' for more information)"
                    )
                    getlogger().verbose(traceback.format_exc())

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

    def _get_partitions_and_prog_envs(self, obj):
        '''Process the partitions and programming environs of the parent.'''
        try:
            part = tuple(obj.valid_systems)
        except AttributeError:
            raise ReframeSyntaxError(
                f'valid_systems is undefined in test {obj.name}'
            )
        else:
            rt = runtime.runtime()
            if '*' in part or rt.system.name in part:
                part = tuple(p.fullname for p in rt.system.partitions)

        try:
            prog_envs = tuple(obj.valid_prog_environs)
        except AttributeError:
            raise ReframeSyntaxError(
                f'valid_prog_environs is undefined in test {obj.name}'
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
        yield from self.__random_access_iter

    def __len__(self):
        if not self.fixtures:
            return 1

        return len(self.__random_access_iter)

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
            ret = dict()
            f_ids = self.__random_access_iter[key]
            for i, f in enumerate(self.fixtures):
                ret[f] = f_ids[i]

            return ret

        return self.fixtures[key]

    @property
    def fixtures(self):
        return self._namespace
