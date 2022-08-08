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

        # Store the system name for name-mangling purposes
        self._sys_name = runtime.runtime().system.name

    def add(self, fixture, variant_num, parent_test):
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
        :param parent_test: The parent test.
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
            fname += '_' + sha256(vname.encode('utf-8')).hexdigest()[:8]

        # Select only the valid partitions
        try:
            valid_sysenv = runtime.valid_sysenv_comb(
                parent_test.valid_systems,
                parent_test.valid_prog_environs
            )
        except AttributeError as e:
            msg = e.args[0] + f' in test {parent_test.display_name!r}'
            raise ReframeSyntaxError(msg) from None

        # Return if there are no valid system/environment combinations
        if not valid_sysenv:
            return []

        # Register the fixture
        if scope == 'session':
            # The name is mangled with the system name
            # Pick the first valid system/environment combination
            pname, ename = None, None
            for part, environs in valid_sysenv.items():
                pname = part.fullname
                for env in environs:
                    ename = env.name
                    break

            if ename is None:
                # No valid environments found
                return []

            # Register the fixture
            fixt_data = FixtureData(variant_num, [ename], [pname],
                                    variables, scope, self._sys_name)
            name = f'{cls.__name__}_{fixt_data.mashup()}'
            self._registry[cls][name] = fixt_data
            reg_names.append(name)
        elif scope == 'partition':
            for part, environs in valid_sysenv.items():
                # The mangled name contains the full partition name
                # Select an environment supported by the partition
                pname = part.fullname
                try:
                    ename = environs[0].name
                except IndexError:
                    continue

                # Register the fixture
                fixt_data = FixtureData(variant_num, [ename], [pname],
                                        variables, scope, pname)
                name = f'{cls.__name__}_{fixt_data.mashup()}'
                self._registry[cls][name] = fixt_data
                reg_names.append(name)
        elif scope == 'environment':
            for part, environs in valid_sysenv.items():
                for env in environs:
                    # The mangled name contains the full part and env names
                    # Register the fixture
                    pname, ename = part.fullname, env.name
                    fixt_data = FixtureData(variant_num, [ename], [pname],
                                            variables, scope,
                                            f'{pname}+{ename}')
                    name = f'{cls.__name__}_{fixt_data.mashup()}'
                    self._registry[cls][name] = fixt_data
                    reg_names.append(name)
        elif scope == 'test':
            # The mangled name contains the parent test name.

            # Register the fixture
            fixt_data = FixtureData(variant_num,
                                    list(parent_test.valid_prog_environs),
                                    list(parent_test.valid_systems),
                                    variables, scope, parent_test.unique_name)
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
        return [e for e in cadidate_environs if e in self._env_by_part[part]]

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
    '''Insert a new fixture in the current test.

    A fixture is a regression test that creates, prepares and/or manages a
    resource for another regression test. Fixtures may contain other fixtures
    and so on, forming a directed acyclic graph. A parent fixture (or a
    regular regression test) requires the resources managed by its child
    fixtures in order to run, and it may only access these fixture resources
    after its ``setup`` pipeline stage. The execution of parent fixtures is
    postponed until all their respective children have completed execution.
    However, the destruction of the resources managed by a fixture occurs in
    reverse order, only after all the parent fixtures have been destroyed.
    This destruction of resources takes place during the ``cleanup`` pipeline
    stage of the regression test. Fixtures must not define the members
    :attr:`~reframe.core.pipeline.RegressionTest.valid_systems` and
    :attr:`~reframe.core.pipeline.RegressionTest.valid_prog_environs`. These
    variables are defined based on the values specified in the parent test,
    ensuring that the fixture runs with a suitable system partition and
    programming environment combination. A fixture's
    :attr:`~reframe.core.pipeline.RegressionTest.name` attribute may be
    internally mangled depending on the arguments passed during the fixture
    declaration. Hence, manually setting or modifying the
    :attr:`~reframe.core.pipeline.RegressionTest.name` attribute in the
    fixture class is disallowed, and breaking this restriction will result in
    undefined behavior.

    .. warning::
       The fixture name mangling is considered an internal framework mechanism
       and it may change in future versions without any notice. Users must not
       express any logic in their tests that relies on a given fixture name
       mangling scheme.


    By default, the resources managed by a fixture are private to the parent
    test. However, it is possible to share these resources across different
    tests by passing the appropriate fixture ``scope`` argument. The different
    scope levels are independent from each other and a fixture only executes
    once per scope, where all the tests that belong to that same scope may use
    the same resources managed by a given fixture instance. The available
    scopes are:

      - **session**: This scope encloses all the tests and fixtures that run
        in the full ReFrame session. This may include tests that use different
        system partition and programming environment combinations. The fixture
        class must derive from
        :class:`~reframe.core.pipeline.RunOnlyRegressionTest` to avoid any
        implicit dependencies on the partition or the programming environment
        used.

      - **partition**: This scope spans across a single system partition. This
        may include different tests that run on the same partition but use
        different programming environments. Fixtures with this scope must be
        independent of the programming environment, which restricts the
        fixture class to derive from
        :class:`~reframe.core.pipeline.RunOnlyRegressionTest`.

      - **environment**: The extent of this scope covers a single combination
        of system partition and programming environment. Since the fixture is
        guaranteed to have the same partition and programming environment as
        the parent test, the fixture class can be any derived class from
        :class:`~reframe.core.pipeline.RegressionTest`. * **test**: This scope
        covers a single instance of the parent test, where the resources
        provided by the fixture are exclusive to each parent test instance.
        The fixture class can be any derived class from
        :class:`~reframe.core.pipeline.RegressionTest`.

    Rather than specifying the scope at the fixture class definition, ReFrame
    fixtures set the scope level from the consumer side (i.e. when used by
    another test or fixture). A test may declare multiple fixtures using the
    same class, where fixtures with different scopes are guaranteed to point
    to different instances of the fixture class. On the other hand, when two
    or more fixtures use the same fixture class and have the same scope, these
    different fixtures will point to the same underlying resource if the
    fixtures refer to the same :ref:`variant<test-variants>` of the fixture
    class. The example below illustrates the different fixture scope usages:

    .. code:: python

       class MyFixture(rfm.RunOnlyRegressionTest):
          my_var = variable(int, value=1)
          ...


       @rfm.simple_test
       class TestA(rfm.RegressionTest):
           valid_systems = ['p1', 'p2']
           valid_prog_environs = ['e1', 'e2']

           # Fixture shared throughout the full session
           f1 = fixture(MyFixture, scope='session')

           # Fixture shared for each supported partition
           f2 = fixture(MyFixture, scope='partition')

           # Fixture shared for each supported part+environ
           f3 = fixture(MyFixture, scope='environment')

           # Fixture private evaluation of MyFixture
           f4 = fixture(MyFixture, scope='test')
           ...


       @rfm.simple_test
       class TestB(rfm.RegressionTest):
           valid_systems = ['p1']
           valid_prog_environs = ['e1']

           # Another private instance of MyFixture
           f1 = fixture(MyFixture, scope='test')

           # Same as f3 in TestA for p1 + e1
           f2 = fixture(MyFixture, scope='environment')

           # Same as f1 in TestA
           f3 = fixture(MyFixture, scope='session')
           ...

           @run_after('setup')
           def access_fixture_resources(self):
               # Dummy pipeline hook to illustrate fixture resource access
               assert self.f1.my_var is not self.f2.my_var
               assert self.f1.my_var is not self.f3.my_var


    :class:`TestA` supports two different valid systems and another two valid
    programming environments. Assuming that both environments are supported by
    each of the system partitions ``'p1'`` and ``'p2'``, this test will
    execute a total of four times. This test uses the very simple
    :class:`MyFixture` fixture multiple times using different scopes, where
    fixture ``f1`` (session scope) will be shared across the four test
    instances, and fixture ``f4`` (test scope) will be executed once per test
    instance. On the other hand, ``f2`` (partition scope) will run once per
    partition supported by test :class:`TestA`, and the multiple per-partition
    executions (i.e. for each programming environment) will share the same
    underlying resource for ``f2``. Lastly, ``f3`` will run a total of four
    times, which is once per partition and environment combination. This
    simple :class:`TestA` shows how multiple instances from the same test can
    share resources, but the real power behind fixtures is illustrated with
    :class:`TestB`, where this resource sharing is extended across different
    tests. For simplicity, :class:`TestB` only supports a single partition
    ``'p1'`` and programming environment ``'e1'``, and similarly to
    :class:`TestA`, ``f1`` (test scope) causes a private evaluation of the
    fixture :class:`MyFixture`. However, the resources managed by fixtures
    ``f2`` (environment scope) and ``f3`` (session scope) are shared with
    :class:`Test1`.

    Fixtures are treated by ReFrame as first-class ReFrame tests, which means
    that these classes can use the same built-in functionalities as in regular
    tests decorated with
    :func:`@rfm.simple_test<reframe.core.decorators.simple_test>`. This
    includes the :func:`~reframe.core.pipeline.RegressionMixin.parameter`
    built-in, where fixtures may have more than one
    :ref:`variant<test-variants>`. When this occurs, a parent test may select
    to either treat a parameterized fixture as a test parameter, or instead,
    to gather all the fixture variants from a single instance of the parent
    test. In essence, fixtures implement `fork-join` model whose behavior may
    be controlled through the ``action`` argument. This argument may be set to
    one of the following options:

      - **fork**: This option parameterizes the parent test as a function of
        the fixture variants. The fixture handle will resolve to a single
        instance of the fixture.

      - **join**: This option gathers all the variants from a fixture into a
        single instance of the parent test. The fixture handle will point to a
        list containing all the fixture variants.

    A test may declare multiple fixtures with different ``action`` options,
    where the default ``action`` option is ``'fork'``. The example below
    illustrates the behavior of these two different options.

    .. code:: python

       class ParamFix(rfm.RegressionTest):
           p = parameter(range(5)) # A simple test parameter
           ...


       @rfm.simple_test
       class TestC(rfm.RegressionTest):
           # Parameterize TestC for each ParamFix variant
           f = fixture(ParamFix, action='fork')
           ...

           @run_after('setup')
           def access_fixture_resources(self):
               print(self.f.p) # Prints the fixture's variant parameter value


       @rfm.simple_test
       class TestD(rfm.RegressionTest):
           # Gather all fixture variants into a single test
           f = fixture(ParamFix, action='join')
           ...

           @run_after('setup')
           def reduce_range(self):
               # Sum all the values of p for each fixture variant
               res = functools.reduce(lambda x, y: x+y,
                                      (fix.p for fix in self.f))
               n = len(self.f)-1
               assert res == (n*n + n)/2

    Here :class:`ParamFix` is a simple fixture class with a single parameter.
    When the test :class:`TestC` uses this fixture with a ``'fork'`` action,
    the test is implicitly parameterized over each variant of
    :class:`ParamFix`. Hence, when the :func:`access_fixture_resources`
    post-setup hook accesses the fixture ``f``, it only access a single
    instance of the :class:`ParamFix` fixture. On the other hand, when this
    same fixture is used with a ``'join'`` action by :class:`TestD`, the test
    is not parameterized and all the :class:`ParamFix` instances are gathered
    into ``f`` as a list. Thus, the post-setup pipeline hook
    :func:`reduce_range` can access all the fixture variants and compute a
    reduction of the different ``p`` values.

    When declaring a fixture, a parent test may select a subset of the fixture
    variants through the ``variants`` argument. This variant selection can be
    done by either passing an iterable containing valid variant indices (see
    :ref:`test-variants` for further information on how the test variants are
    indexed), or instead, passing a mapping with the parameter name (of the
    fixture class) as keys and filtering functions as values. These filtering
    functions are unary functions that return the value of a boolean
    expression on the values of the specified parameter, and they all must
    evaluate to :class:`True` for at least one of the fixture class variants.
    See the example below for an illustration on how to filter-out fixture
    variants.

    .. code:: python

       class ComplexFixture(rfm.RegressionTest):
           # A fixture with 400 different variants.
           p0 = parameter(range(100))
           p1 = parameter(['a', 'b', 'c', 'd'])
           ...

       @rfm.simple_test
       class TestE(rfm.RegressionTest):
           # Select the fixture variants with boolean conditions
           foo = fixture(ComplexFixture,
                         variants={'p0': lambda x: x<10,
                                   'p1': lambda x: x=='d'})

           # Select the fixture variants by index
           bar = fixture(ComplexFixture, variants=range(300,310))
           ...

    A parent test may also specify the value of different variables in the
    fixture class to be set before its instantiation. Each variable must have
    been declared in the fixture class with the
    :func:`~reframe.core.pipeline.RegressionMixin.variable` built-in,
    otherwise it is silently ignored. This variable specification is
    equivalent to deriving a new class from the fixture class, and setting
    these variable values in the class body of a newly derived class.
    Therefore, when fixture declarations use the same fixture class and pass
    different values to the ``variables`` argument, the fixture class is
    interpreted as a different class for each of these fixture declarations.
    See the example below.

    .. code:: python

       class Fixture(rfm.RegressionTest):
           v = variable(int, value=1)
           ...

       @rfm.simple_test
       class TestF(rfm.RegressionTest):
           foo = fixture(Fixture)
           bar = fixture(Fixture, variables={'v':5})
           baz = fixture(Fixture, variables={'v':10})
           ...

           @run_after('setup')
           def print_fixture_variables(self):
               print(self.foo.v) # Prints 1
               print(self.bar.v) # Prints 5
               print(self.baz.v) # Prints 10

    The test :class:`TestF` declares the fixtures ``foo``, ``bar`` and ``baz``
    using the same :class:`Fixture` class. If no variables were set in ``bar``
    and ``baz``, this would result into the same fixture being declared
    multiple times in the same scope (implicitly set to ``'test'``), which
    would lead to a single instance of :class:`Fixture` being referred to by
    ``foo``, ``bar`` and ``baz``. However, in this case ReFrame identifies
    that the declared fixtures pass different values to the ``variables``
    argument in the fixture declaration, and executes these three fixtures
    separately.

    .. note::
       Mappings passed to the ``variables`` argument that define the same
       class variables in different order are interpreted as the same value.
       The two fixture declarations below are equivalent, and both ``foo`` and
       ``bar`` will point to the same instance of the fixture class
       :class:`MyResource`.

       .. code:: python

         foo = fixture(MyResource, variables={'a':1, 'b':2})
         bar = fixture(MyResource, variables={'b':2, 'a':1})


    **Early access to fixture objects**

    The test instance represented by a fixture can be accessed fully from
    within a test only after the setup stage. The reason for that is that
    fixtures eventually translate into test dependencies and access to the
    parent dependencies cannot happen before the this stage.

    However, it is often useful, especially in the case of parameterized
    fixtures, to be able to access the fixture parameters earlier, e.g., in a
    post-init hook in order to properly set the
    :attr:`~reframe.core.pipeline.RegressionTest.valid_systems` and
    :attr:`~reframe.core.pipeline.RegressionTest.valid_prog_environs` of the
    test. These attributes cannot be set later than the test's initialization
    in order to have an effect.

    For this reason, early access to fixture objects is allowed *only* for
    retrieving their parameters.

    .. code-block:: python

       class Fixture(rfm.RegressionTest):
           x = parameter([1, 2, 3])


       class Test(rfm.RunOnlyRegressionTest):
           foo = fixture(Fixture)
           executable = './myexec'
           valid_prog_environs = ['*']

           @run_after('init')
           def early_access(self):
                # Only fixture parameters can be accessed here!
                if self.foo.x == 1:
                    self.valid_systems = ['sys1]
                else:
                    self.valid_systems = ['sys2']

           @run_after('setup')
           def normal_access(self):
               # Any test attribute of the associated fixture test can be
               # accessed here
               self.executable_opts = [
                   '-i', os.path.join(self.foo.stagedir, 'input.txt')
               ]


    During test initialization, ReFrame binds the :attr:`foo` name to a proxy
    object that holds the parameterization of the target fixture. This proxy
    object is recursive, so that if fixture :attr:`foo` contained another
    fixture named :attr:`bar`, it would allow you to access any parameters of
    that fixture with ``self.foo.bar.param``.

    During the test setup stage, the :attr:`foo`'s binding changes and it is
    now bound to the exact test instance that was executed for the target test
    instance.

    :param cls: A class derived from
        :class:`~reframe.core.pipeline.RegressionTest` that manages a given
        resource. The base from this class may be further restricted to other
        derived classes of :class:`~reframe.core.pipeline.RegressionTest`
        depending on the ``scope`` parameter.

    :param scope: Sets the extent to which other regression tests may share
        the resources managed by a fixture. The available scopes are, from
        more to less restrictive, ``'test'``, ``'environment'``,
        ``'partition'`` and ``'session'``. By default a fixture's scope is set
        to ``'test'``, which makes the resource private to the test that uses
        the fixture. This means that when multiple regression tests use the
        same fixture class with a ``'test'`` scope, the fixture will run once
        per regression test. When the scope is set to ``'environment'``, the
        resources managed by the fixture are shared across all the tests that
        use the fixture and run on the same system partition and use the same
        programming environment. When the scope is set to ``'partition'``, the
        resources managed by the fixture are shared instead across all the
        tests that use the fixture and run on the same system partition.
        Lastly, when the scope is set to ``'session'``, the resources managed
        by the fixture are shared across the full ReFrame session. Fixtures
        with either ``'partition'`` or ``'session'`` scopes may be shared
        across different regression tests under different programming
        environments, and for this reason, when using these two scopes, the
        fixture class ``cls`` is required to derive from
        :class:`~reframe.core.pipeline.RunOnlyRegressionTest`.

    :param action: Set the behavior of a parameterized fixture to either
        ``'fork'`` or ``'join'``. With a ``'fork'`` action, a parameterized
        fixture effectively parameterizes the regression test. On the other
        hand, a ``'join'`` action gathers all the fixture variants into the
        same instance of the regression test. By default, the ``action``
        parameter is set to ``'fork'``.

    :param variants: Filter or sub-select a subset of the variants from a
        parameterized fixture. This argument can be either an iterable with
        the indices from the desired variants, or a mapping containing unary
        functions that return the value of a boolean expression on the values
        of a given parameter.

    :param variables: Mapping to set the values of fixture's variables. The
        variables are set after the fixture class has been created (i.e. after
        the class body has executed) and before the fixture class is
        instantiated.


    .. versionadded:: 3.9.0
    .. versionchanged:: 3.11.0
       Allow early access of fixture objects.

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


class FixtureProxy:
    def __init__(self, fixture_info):
        for k, v in fixture_info['params'].items():
            setattr(self, k, v)

        for k, v in fixture_info['fixtures'].items():
            if not isinstance(v, tuple):
                setattr(self, k, FixtureProxy(v))


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

        # Register the fixtures
        for name, fixture in self.fixtures.items():
            dep_names = []
            for variant in fixture_variants[name]:
                # Register all the variants and track the fixture names
                dep_names += obj._rfm_fixture_registry.add(fixture,
                                                           variant, obj)

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
