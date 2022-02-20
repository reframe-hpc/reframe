===================
Regression Test API
===================

This page provides a reference guide of the ReFrame API for writing regression tests covering all the relevant details.
Internal data structures and APIs are covered only to the extent that this might be helpful to the final user of the framework.


 .. _regression-bases:

-----------------
Test Base Classes
-----------------

.. automodule:: reframe.core.pipeline
   :members:
   :show-inheritance:


---------------
Test Decorators
---------------

.. autodecorator:: reframe.core.decorators.parameterized_test(*inst)

.. autodecorator:: reframe.core.decorators.required_version(*versions)

.. autodecorator:: reframe.core.decorators.simple_test


--------------
Built-in types
--------------

.. versionadded:: 3.4.2

ReFrame provides built-in types which facilitate the process of writing extensible regression tests (i.e. a test library).
These *builtins* are only available when used directly in the class body of classes derived from any of the :ref:`regression-bases`.
Through builtins, ReFrame internals are able to *pre-process* and validate the test input before the actual test creation takes place.
This provides the ReFrame internals with further control over the user's input, making the process of writing regression tests less error-prone.
In essence, these builtins exert control over the test creation, and they allow adding and/or modifying certain attributes of the regression test.

.. note::
  The built-in types described below can only be used to declare class variables and must never be part of any container type.
  Ignoring this restriction will result in undefined behavior.

  .. code::

    class MyTest(rfm.RegressionMixin):
        p0 = parameter([1, 2])   # Correct
        p1 = [parameter([1, 2])] # Undefined behavior


.. py:function:: RegressionMixin.parameter(values=None, inherit_params=False, filter_params=None, fmt=None, loggable=False)

  Inserts or modifies a regression test parameter.
  At the class level, these parameters are stored in a separate namespace referred to as the *parameter space*.
  If a parameter with a matching name is already present in the parameter space of a parent class, the existing parameter values will be combined with those provided by this method following the inheritance behavior set by the arguments ``inherit_params`` and ``filter_params``.
  Instead, if no parameter with a matching name exists in any of the parent parameter spaces, a new regression test parameter is created.
  A regression test can be parameterized as follows:

  .. code:: python

    class Foo(rfm.RegressionTest):
        variant = parameter(['A', 'B'])
        # print(variant) # Error: a parameter may only be accessed from the class instance.

        @run_after('init')
        def do_something(self):
            if self.variant == 'A':
                do_this()
            else:
                do_other()

  One of the most powerful features of these built-in functions is that they store their input information at the class level.
  However, a parameter may only be accessed from the class instance and accessing it directly from the class body is disallowed.
  With this approach, extending or specializing an existing parameterized regression test becomes straightforward, since the test attribute additions and modifications made through built-in functions in the parent class are automatically inherited by the child test.
  For instance, continuing with the example above, one could override the :func:`do_something` hook in the :class:`Foo` regression test as follows:

  .. code:: python

    class Bar(Foo):
        @run_after('init')
        def do_something(self):
            if self.variant == 'A':
                override_this()
            else:
                override_other()

  Moreover, a derived class may extend, partially extend and/or modify the parameter values provided in the base class as shown below.

  .. code:: python

    class ExtendVariant(Bar):
        # Extend the full set of inherited variant parameter values to ['A', 'B', 'C']
        variant = parameter(['C'], inherit_params=True)

    class PartiallyExtendVariant(Bar):
        # Extend a subset of the inherited variant parameter values to ['A', 'D']
        variant = parameter(['D'], inherit_params=True,
                            filter_params=lambda x: x[:1])

    class ModifyVariant(Bar):
        # Modify the variant parameter values to ['AA', 'BA']
        variant = parameter(inherit_params=True,
                            filter_params=lambda x: map(lambda y: y+'A', x))

  A parameter with no values is referred to as an *abstract parameter* (i.e. a parameter that is declared but not defined).
  Therefore, classes with at least one abstract parameter are considered abstract classes.

  .. code:: python

    class AbstractA(Bar):
        variant = parameter()

    class AbstractB(Bar):
        variant = parameter(inherit_params=True, filter_params=lambda x: [])


  :param values: An iterable containing the parameter values.
  :param inherit_params: If :obj:`True`, the parameter values defined in any base class will be inherited.
     In this case, the parameter values provided in the current class will extend the set of inherited parameter values.
     If the parameter does not exist in any of the parent parameter spaces, this option has no effect.
  :param filter_params: Function to filter/modify the inherited parameter values that may have been provided in any of the parent parameter spaces.
     This function must accept a single iterable argument and return an iterable.
     It will be called with the inherited parameter values and it must return the filtered set of parameter values.
     This function will only have an effect if used with ``inherit_params=True``.
  :param fmt: A formatting function that will be used to format the values of this parameter in the test's :attr:`~reframe.core.pipeline.RegressionTest.display_name`.
     This function should take as argument the parameter value and return a string representation of the value.
     If the returned value is not a string, it will be converted using the :py:func:`str` function.
  :param loggable: Mark this parameter as loggable.
     If :obj:`True`, this parameter will become a log record attribute under the name ``check_NAME``, where ``NAME`` is the name of the parameter.


  .. versionadded:: 3.10.0
     The ``fmt`` argument is added.

  .. versionadded:: 3.10.2
     The ``loggable`` argument is added.


.. py:function:: RegressionMixin.variable(*types, value=None, field=None, **kwargs)

  Inserts a new regression test variable.
  Declaring a test variable through the :func:`variable` built-in allows for a more robust test implementation than if the variables were just defined as regular test attributes (e.g. ``self.a = 10``).
  Using variables declared through the :func:`variable` built-in guarantees that these regression test variables will not be redeclared by any child class, while also ensuring that any values that may be assigned to such variables comply with its original declaration.
  In essence, declaring test variables with the :func:`variable` built-in removes any potential test errors that might be caused by accidentally overriding a class attribute. See the example below.


  .. code:: python

    class Foo(rfm.RegressionTest):
        my_var = variable(int, value=8)
        not_a_var = my_var - 4

        @run_after('init')
        def access_vars(self):
            print(self.my_var) # prints 8.
            # self.my_var = 'override' # Error: my_var must be an int!
            self.not_a_var = 'override' # However, this would work. Dangerous!
            self.my_var = 10 # tests may also assign values the standard way

  Here, the argument ``value`` in the :func:`variable` built-in sets the default value for the variable.
  This value may be accessed directly from the class body, as long as it was assigned before either in the same class body or in the class body of a parent class.
  This behavior extends the standard Python data model, where a regular class attribute from a parent class is never available in the class body of a child class.
  Hence, using the :func:`variable` built-in enables us to directly use or modify any variables that may have been declared upstream the class inheritance chain, without altering their original value at the parent class level.

  .. code:: python

    class Bar(Foo):
        print(my_var) # prints 8
        # print(not_a_var) # This is standard Python and raises a NameError

        # Since my_var is available, we can also update its value:
        my_var = 4

        # Bar inherits the full declaration of my_var with the original type-checking.
        # my_var = 'override' # Wrong type error again!

        @run_after('init')
        def access_vars(self):
            print(self.my_var) # prints 4
            print(self.not_a_var) # prints 4


    print(Foo.my_var) # prints 8
    print(Bar.my_var) # prints 4


  Here, :class:`Bar` inherits the variables from :class:`Foo` and can see that ``my_var`` has already been declared in the parent class. Therefore, the value of ``my_var`` is updated ensuring that the new value complies to the original variable declaration.
  However, the value of ``my_var`` at :class:`Foo` remains unchanged.

  These examples above assumed that a default value can be provided to the variables in the bases tests, but that might not always be the case.
  For example, when writing a test library, one might want to leave some variables undefined and force the user to set these when using the test.
  As shown in the example below, imposing such requirement is as simple as not passing any ``value`` to the :func:`variable` built-in, which marks the given variable as *required*.

  .. code:: python

    # Test as written in the library
    class EchoBaseTest(rfm.RunOnlyRegressionTest):
      what = variable(str)

      valid_systems = ['*']
      valid_prog_environs = ['*']

      @run_before('run')
      def set_executable(self):
          self.executable = f'echo {self.what}'

      @sanity_function
      def assert_what(self):
          return sn.assert_found(fr'{self.what}')


    # Test as written by the user
    @rfm.simple_test
    class HelloTest(EchoBaseTest):
      what = 'Hello'


    # A parameterized test with type-checking
    @rfm.simple_test
    class FoodTest(EchoBaseTest):
      param = parameter(['Bacon', 'Eggs'])

      @run_after('init')
      def set_vars_with_params(self):
        self.what = self.param


  Similarly to a variable with a value already assigned to it, the value of a required variable may be set either directly in the class body, on the :func:`__init__` method, or in any other hook before it is referenced.
  Otherwise an error will be raised indicating that a required variable has not been set.
  Conversely, a variable with a default value already assigned to it can be made required by assigning it the ``required`` keyword.
  However, this ``required`` keyword is only available in the class body.

  .. code:: python

    class MyRequiredTest(HelloTest):
      what = required


  Running the above test will cause the :func:`set_exec_and_sanity` hook from :class:`EchoBaseTest` to throw an error indicating that the variable ``what`` has not been set.

  :param `*types`: the supported types for the variable.
  :param value: the default value assigned to the variable. If no value is provided, the variable is set as ``required``.
  :param field: the field validator to be used for this variable.
      If no field argument is provided, it defaults to :attr:`reframe.core.fields.TypedField`.
      The provided field validator by this argument must derive from :attr:`reframe.core.fields.Field`.
  :param loggable: Mark this variable as loggable.
     If :obj:`True`, this variable will become a log record attribute under the name ``check_NAME``, where ``NAME`` is the name of the variable.
  :param `**kwargs`: *kwargs* to be forwarded to the constructor of the field validator.

  .. versionadded:: 3.10.2
     The ``loggable`` argument is added.


.. py:function:: RegressionMixin.fixture(cls, *, scope='test', action='fork', variants='all', variables=None)

  Declare a new fixture in the current regression test.
  A fixture is a regression test that creates, prepares and/or manages a resource for another regression test.
  Fixtures may contain other fixtures and so on, forming a directed acyclic graph.
  A parent fixture (or a regular regression test) requires the resources managed by its child fixtures in order to run, and it may only access these fixture resources after its ``setup`` pipeline stage.
  The execution of parent fixtures is postponed until all their respective children have completed execution.
  However, the destruction of the resources managed by a fixture occurs in reverse order, only after all the parent fixtures have been destroyed.
  This destruction of resources takes place during the ``cleanup`` pipeline stage of the regression test.
  Fixtures must not define the members :attr:`~reframe.core.pipeline.RegressionTest.valid_systems` and :attr:`~reframe.core.pipeline.RegressionTest.valid_prog_environs`.
  These variables are defined based on the values specified in the parent test, ensuring that the fixture runs with a suitable system partition and programming environment combination.
  A fixture's :attr:`~reframe.core.pipeline.RegressionTest.name` attribute may be internally mangled depending on the arguments passed during the fixture declaration.
  Hence, manually setting or modifying the :attr:`~reframe.core.pipeline.RegressionTest.name` attribute in the fixture class is disallowed, and breaking this restriction will result in undefined behavior.

  .. warning::
    The fixture name mangling is considered an internal framework mechanism and it may change in future versions without any notice.
    Users must not express any logic in their tests that relies on a given fixture name mangling scheme.


  By default, the resources managed by a fixture are private to the parent test.
  However, it is possible to share these resources across different tests by passing the appropriate fixture ``scope`` argument.
  The different scope levels are independent from each other and a fixture only executes once per scope, where all the tests that belong to that same scope may use the same resources managed by a given fixture instance.
  The available scopes are:

   * **session**: This scope encloses all the tests and fixtures that run in the full ReFrame session.
     This may include tests that use different system partition and programming environment combinations.
     The fixture class must derive from :class:`~reframe.core.pipeline.RunOnlyRegressionTest` to avoid any implicit dependencies on the partition or the programming environment used.
   * **partition**: This scope spans across a single system partition.
     This may include different tests that run on the same partition but use different programming environments.
     Fixtures with this scope must be independent of the programming environment, which restricts the fixture class to derive from :class:`~reframe.core.pipeline.RunOnlyRegressionTest`.
   * **environment**: The extent of this scope covers a single combination of system partition and programming environment.
     Since the fixture is guaranteed to have the same partition and programming environment as the parent test, the fixture class can be any derived class from :class:`~reframe.core.pipeline.RegressionTest`.
   * **test**: This scope covers a single instance of the parent test, where the resources provided by the fixture are exclusive to each parent test instance.
     The fixture class can be any derived class from :class:`~reframe.core.pipeline.RegressionTest`.

  Rather than specifying the scope at the fixture class definition, ReFrame fixtures set the scope level from the consumer side (i.e. when used by another test or fixture).
  A test may declare multiple fixtures using the same class, where fixtures with different scopes are guaranteed to point to different instances of the fixture class.
  On the other hand, when two or more fixtures use the same fixture class and have the same scope, these different fixtures will point to the same underlying resource if the fixtures refer to the same :ref:`variant<test-variants>` of the fixture class.
  The example below illustrates the different fixture scope usages:

  .. code:: python

    class MyFixture(rfm.RunOnlyRegressionTest):
       '''Manage some resource'''
       my_var = variable(int, value=1)
       ...


    @rfm.simple_test
    class TestA(rfm.RegressionTest):
        valid_systems = ['p1', 'p2']
        valid_prog_environs = ['e1', 'e2']
        f1 = fixture(MyFixture, scope='session')     # Shared throughout the full session
        f2 = fixture(MyFixture, scope='partition')   # Shared for each supported partition
        f3 = fixture(MyFixture, scope='environment') # Shared for each supported part+environ
        f4 = fixture(MyFixture, scope='test')        # Private evaluation of MyFixture
        ...


    @rfm.simple_test
    class TestB(rfm.RegressionTest):
        valid_systems = ['p1']
        valid_prog_environs = ['e1']
        f1 = fixture(MyFixture, scope='test')        # Another private instance of MyFixture
        f2 = fixture(MyFixture, scope='environment') # Same as f3 in TestA for p1 + e1
        f3 = fixture(MyFixture, scope='session')     # Same as f1 in TestA
        ...

        @run_after('setup')
        def access_fixture_resources(self):
            '''Dummy pipeline hook to illustrate fixture resource access.'''
            assert self.f1.my_var is not self.f2.my_var
            assert self.f1.my_var is not self.f3.my_var


  :class:`TestA` supports two different valid systems and another two valid programming environments.
  Assuming that both environments are supported by each of the system partitions ``'p1'`` and ``'p2'``, this test will execute a total of four times.
  This test uses the very simple :class:`MyFixture` fixture multiple times using different scopes, where fixture ``f1`` (session scope) will be shared across the four test instances, and fixture ``f4`` (test scope) will be executed once per test instance.
  On the other hand, ``f2`` (partition scope) will run once per partition supported by test :class:`TestA`, and the multiple per-partition executions (i.e. for each programming environment) will share the same underlying resource for ``f2``.
  Lastly, ``f3`` will run a total of four times, which is once per partition and environment combination.
  This simple :class:`TestA` shows how multiple instances from the same test can share resources, but the real power behind fixtures is illustrated with :class:`TestB`, where this resource sharing is extended across different tests.
  For simplicity, :class:`TestB` only supports a single partition ``'p1'`` and programming environment ``'e1'``, and similarly to :class:`TestA`, ``f1`` (test scope) causes a private evaluation of the fixture :class:`MyFixture`.
  However, the resources managed by fixtures ``f2`` (environment scope) and ``f3`` (session scope) are shared with :class:`Test1`.

  Fixtures are treated by ReFrame as first-class ReFrame tests, which means that these classes can use the same built-in functionalities as in regular tests decorated with :func:`@rfm.simple_test<reframe.core.decorators.simple_test>`.
  This includes the :func:`~reframe.core.pipeline.RegressionMixin.parameter` built-in, where fixtures may have more than one :ref:`variant<test-variants>`.
  When this occurs, a parent test may select to either treat a parameterized fixture as a test parameter, or instead, to gather all the fixture variants from a single instance of the parent test.
  In essence, fixtures implement `fork-join` model whose behavior may be controlled through the ``action`` argument.
  This argument may be set to one of the following options:

   * **fork**: This option parameterizes the parent test as a function of the fixture variants.
     The fixture handle will resolve to a single instance of the fixture.
   * **join**: This option gathers all the variants from a fixture into a single instance of the parent test.
     The fixture handle will point to a list containing all the fixture variants.

  A test may declare multiple fixtures with different ``action`` options, where the default ``action`` option is ``'fork'``.
  The example below illustrates the behavior of these two different options.

  .. code:: python

    class ParamFix(rfm.RegressionTest):
        '''Manage some resource'''
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
            '''Sum all the values of p for each fixture variant'''
            res = functools.reduce(lambda x, y: x+y, (fix.p for fix in self.f))
            n = len(self.f)-1
            assert res == (n*n + n)/2

  Here :class:`ParamFix` is a simple fixture class with a single parameter.
  When the test :class:`TestC` uses this fixture with a ``'fork'`` action, the test is implicitly parameterized over each variant of :class:`ParamFix`.
  Hence, when the :func:`access_fixture_resources` post-setup hook accesses the fixture ``f``, it only access a single instance of the :class:`ParamFix` fixture.
  On the other hand, when this same fixture is used with a ``'join'`` action by :class:`TestD`, the test is not parameterized and all the :class:`ParamFix` instances are gathered into ``f`` as a list.
  Thus, the post-setup pipeline hook :func:`reduce_range` can access all the fixture variants and compute a reduction of the different ``p`` values.

  When declaring a fixture, a parent test may select a subset of the fixture variants through the ``variants`` argument.
  This variant selection can be done by either passing an iterable containing valid variant indices (see :ref:`test-variants` for further information on how the test variants are indexed), or instead, passing a mapping with the parameter name (of the fixture class) as keys and filtering functions as values.
  These filtering functions are unary functions that return the value of a boolean expression on the values of the specified parameter, and they all must evaluate to :class:`True` for at least one of the fixture class variants.
  See the example below for an illustration on how to filter-out fixture variants.

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
                      variants={'p0': lambda x: x<10, 'p1': lambda x: x=='d'})

        # Select the fixture variants by index
        bar = fixture(ComplexFixture, variants=range(300,310))
        ...

  A parent test may also specify the value of different variables in the fixture class to be set before its instantiation.
  Each variable must have been declared in the fixture class with the :func:`~reframe.core.pipeline.RegressionMixin.variable` built-in, otherwise it is silently ignored.
  This variable specification is equivalent to deriving a new class from the fixture class, and setting these variable values in the class body of a newly derived class.
  Therefore, when fixture declarations use the same fixture class and pass different values to the ``variables`` argument, the fixture class is interpreted as a different class for each of these fixture declarations.
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

  The test :class:`TestF` declares the fixtures ``foo``, ``bar`` and ``baz`` using the same :class:`Fixture` class.
  If no variables were set in ``bar`` and ``baz``, this would result into the same fixture being declared multiple times in the same scope (implicitly set to ``'test'``), which would lead to a single instance of :class:`Fixture` being referred to by ``foo``, ``bar`` and ``baz``.
  However, in this case ReFrame identifies that the declared fixtures pass different values to the ``variables`` argument in the fixture declaration, and executes these three fixtures separately.

  .. note::
     Mappings passed to the ``variables`` argument that define the same class variables in different order are interpreted as the same value.
     The two fixture declarations below are equivalent, and both ``foo`` and ``bar`` will point to the same instance of the fixture class :class:`MyResource`.

     .. code:: python

       foo = fixture(MyResource, variables={'a':1, 'b':2})
       bar = fixture(MyResource, variables={'b':2, 'a':1})



  :param cls: A class derived from :class:`~reframe.core.pipeline.RegressionTest` that manages a given resource.
    The base from this class may be further restricted to other derived classes of :class:`~reframe.core.pipeline.RegressionTest` depending on the ``scope`` parameter.
  :param scope: Sets the extent to which other regression tests may share the resources managed by a fixture.
    The available scopes are, from more to less restrictive, ``'test'``, ``'environment'``, ``'partition'`` and ``'session'``.
    By default a fixture's scope is set to ``'test'``, which makes the resource private to the test that uses the fixture.
    This means that when multiple regression tests use the same fixture class with a ``'test'`` scope, the fixture will run once per regression test.
    When the scope is set to ``'environment'``, the resources managed by the fixture are shared across all the tests that use the fixture and run on the same system partition and use the same programming environment.
    When the scope is set to ``'partition'``, the resources managed by the fixture are shared instead across all the tests that use the fixture and run on the same system partition.
    Lastly, when the scope is set to ``'session'``, the resources managed by the fixture are shared across the full ReFrame session.
    Fixtures with either ``'partition'`` or ``'session'`` scopes may be shared across different regression tests under different programming environments, and for this reason, when using these two scopes, the fixture class ``cls`` is required to derive from :class:`~reframe.core.pipeline.RunOnlyRegressionTest`.
  :param action: Set the behavior of a parameterized fixture to either ``'fork'`` or ``'join'``.
    With a ``'fork'`` action, a parameterized fixture effectively parameterizes the regression test.
    On the other hand, a ``'join'`` action gathers all the fixture variants into the same instance of the regression test.
    By default, the ``action`` parameter is set to ``'fork'``.
  :param variants: Filter or sub-select a subset of the variants from a parameterized fixture.
    This argument can be either an iterable with the indices from the desired variants, or a mapping containing unary functions that return the value of a boolean expression on the values of a given parameter.
  :param variables: Mapping to set the values of fixture's variables. The variables are set after the fixture class has been created (i.e. after the class body has executed) and before the fixture class is instantiated.


  .. versionadded:: 3.9.0


------------------
Built-in functions
------------------

ReFrame provides the following built-in functions, which are only available in the class body of classes deriving from :class:`~reframe.core.pipeline.RegressionMixin`.

.. py:decorator:: RegressionMixin.sanity_function(func)

  Decorate a member function as the sanity function of the test.

  This decorator will convert the given function into a :func:`~RegressionMixin.deferrable` and mark it to be executed during the test's sanity stage.
  When this decorator is used, manually assigning a value to :attr:`~RegressionTest.sanity_patterns` in the test is not allowed.

  Decorated functions may be overridden by derived classes, and derived classes may also decorate a different method as the test's sanity function.
  Decorating multiple member functions in the same class is not allowed.
  However, a :class:`RegressionTest` may inherit from multiple :class:`RegressionMixin` classes with their own sanity functions.
  In this case, the derived class will follow Python's `MRO <https://docs.python.org/3/library/stdtypes.html#class.__mro__>`_ to find a suitable sanity function.

  .. versionadded:: 3.7.0

.. py:decorator:: RegressionMixin.performance_function(unit, *, perf_key=None)

   Decorate a member function as a performance function of the test.

   This decorator converts the decorated method into a performance deferrable function (see ":ref:`deferrable-performance-functions`" for more details) whose evaluation is deferred to the performance stage of the regression test.
   The decorated function must take a single argument without a default value (i.e. ``self``) and any number of arguments with default values.
   A test may decorate multiple member functions as performance functions, where each of the decorated functions must be provided with the units of the performance quantities to be extracted from the test.
   These performance units must be of type :class:`str`.
   Any performance function may be overridden in a derived class and multiple bases may define their own performance functions.
   In the event of a name conflict, the derived class will follow Python's `MRO <https://docs.python.org/3/library/stdtypes.html#class.__mro__>`_ to choose the appropriate performance function.
   However, defining more than one performance function with the same name in the same class is disallowed.

   The full set of performance functions of a regression test is stored under :attr:`~reframe.core.pipeline.RegressionTest.perf_variables` as key-value pairs, where, by default, the key is the name of the decorated member function, and the value is the deferred performance function itself.
   Optionally, the key under which a performance function is stored in :attr:`~reframe.core.pipeline.RegressionTest.perf_variables` can be customised by passing the desired key as the ``perf_key`` argument to this decorator.

   .. versionadded:: 3.8.0

.. py:decorator:: RegressionMixin.deferrable(func)

  Converts the decorated method into a deferrable function.

  See :ref:`deferrable-functions` for further information on deferrable functions.

  .. versionadded:: 3.7.0

.. autodecorator:: reframe.core.pipeline.RegressionMixin.loggable_as(name)

.. py:decorator:: reframe.core.pipeline.RegressionMixin.loggable

   Equivalent to :func:`@loggable_as(None) <reframe.core.pipeline.RegressionMixin.loggable_as>`.

   .. versionadded:: 3.10.2


.. py:decorator:: RegressionMixin.require_deps(func)

  Decorator to denote that a function will use the test dependencies.

  The arguments of the decorated function must be named after the dependencies that the function intends to use.
  The decorator will bind the arguments to a partial realization of the :func:`~reframe.core.pipeline.RegressionTest.getdep` function, such that conceptually the new function arguments will be the following:

  .. code-block:: python

     new_arg = functools.partial(getdep, orig_arg_name)

  The converted arguments are essentially functions accepting a single argument, which is the target test's programming environment.
  Additionally, this decorator will attach the function to run *after* the test's setup phase, but *before* any other "post-setup" pipeline hook.

  .. warning::
     .. versionchanged:: 3.7.0
        Using this function from the :py:mod:`reframe` or :py:mod:`reframe.core.decorators` modules is now deprecated.
        You should use the built-in function described here.

.. py:function:: RegressionMixin.bind(func, name=None)

  Bind a free function to a regression test.

  By default, the function is bound with the same name as the free function.
  However, the function can be bound using a different name with the ``name`` argument.

  :param func: external function to be bound to a class.
  :param name: bind the function under a different name.

  .. versionadded:: 3.6.2



--------------
Pipeline Hooks
--------------

ReFrame provides built-in functions that allow attaching arbitrary functions to run before and/or after a given stage of the execution pipeline.
Once attached to a given stage, these functions are referred to as *pipeline hooks*.
A hook may be attached to multiple pipeline stages and multiple hooks may also be attached to the same pipeline stage.
Pipeline hooks attached to multiple stages will be executed on each pipeline stage the hook was attached to.
Pipeline stages with multiple hooks attached will execute these hooks in the order in which they were attached to the given pipeline stage.
A derived class will inherit all the pipeline hooks defined in its bases, except for those whose hook function is overridden by the derived class.
A function that overrides a pipeline hook from any of the base classes will not be a pipeline hook unless the overriding function is explicitly reattached to any pipeline stage.
In the event of a name clash arising from multiple inheritance, the inherited pipeline hook will be chosen following Python's `MRO <https://docs.python.org/3/library/stdtypes.html#class.__mro__>`__.

A function may be attached to any of the following stages (listed in order of execution): ``init``, ``setup``, ``compile``, ``run``, ``sanity``, ``performance`` and ``cleanup``.
The ``init`` stage refers to the test's instantiation and it runs before entering the execution pipeline.
Therefore, a test function cannot be attached to run before the ``init`` stage.
Hooks attached to any other stage will run exactly before or after this stage executes.
So although a "post-init" and a "pre-setup" hook will both run *after* a test has been initialized and *before* the test goes through the first pipeline stage, they will execute in different times:
the post-init hook will execute *right after* the test is initialized.
The framework will then continue with other activities and it will execute the pre-setup hook *just before* it schedules the test for executing its setup stage.

Pipeline hooks are executed in reverse MRO order, i.e., the hooks of the least specialized class will be executed first.
In the following example, :func:`BaseTest.x` will execute before :func:`DerivedTest.y`:

.. code:: python

   class BaseTest(rfm.RegressionTest):
       @run_after('setup')
       def x(self):
           '''Hook x'''

   class DerivedTest(BaseTeset):
       @run_after('setup')
       def y(self):
           '''Hook y'''


.. note::
   Pipeline hooks do not execute in the test's stage directory.
   However, the test's :attr:`~reframe.core.pipeline.RegressionTest.stagedir` can be accessed by explicitly changing the working directory from within the hook function itself (see the :class:`~reframe.utility.osext.change_dir` utility for further details):

   .. code:: python

     import reframe.utility.osext as osext

     class MyTest(rfm.RegressionTest):
         ...
         @run_after('run')
         def my_post_run_hook(self):
             # Access the stage directory
             with osext.change_dir(self.stagedir):
                 ...

.. warning::
   .. versionchanged:: 3.7.0
      Declaring pipeline hooks using the same name functions from the :py:mod:`reframe` or :py:mod:`reframe.core.decorators` modules is now deprecated.
      You should use the built-in functions described in this section instead.

.. warning::
   .. versionchanged:: 3.9.2

      Execution of pipeline hooks until this version was implementation-defined.
      In practice, hooks of a derived class were executed before those of its parents.

      This version defines the execution order of hooks, which now follows a strict reverse MRO order, so that parent hooks will execute before those of derived classes.
      Tests that relied on the execution order of hooks might break with this change.


.. py:decorator:: RegressionMixin.run_before(stage)

  Decorator for attaching a function to a given pipeline stage.

  The function will run just before the specified pipeline stage and it cannot accept any arguments except ``self``.
  This decorator can be stacked, in which case the function will be attached to multiple pipeline stages.
  See above for the valid ``stage`` argument values.


.. py:decorator:: RegressionMixin.run_after(stage)

  Decorator for attaching a function to a given pipeline stage.

  This is analogous to :func:`~RegressionMixin.run_before`, except that the hook will execute right after the stage it was attached to.
  This decorator also supports ``'init'`` as a valid ``stage`` argument, where in this case, the hook will execute right after the test is initialized (i.e. after the :func:`__init__` method is called) and before entering the test's pipeline.
  In essence, a post-init hook is equivalent to defining additional :func:`__init__` functions in the test.
  The following code

  .. code-block:: python

   class MyTest(rfm.RegressionTest):
     @run_after('init')
     def foo(self):
         self.x = 1

  is equivalent to

  .. code-block:: python

   class MyTest(rfm.RegressionTest):
     def __init__(self):
         self.x = 1

  .. versionchanged:: 3.5.2
     Add support for post-init hooks.


.. _test-variants:

-------------
Test variants
-------------

Through the :func:`~reframe.core.pipeline.RegressionMixin.parameter` and :func:`~reframe.core.pipeline.RegressionMixin.fixture` builtins, a regression test may store multiple versions or `variants` of a regression test at the class level.
During class creation, the test's parameter and fixture spaces are constructed and combined, assigning a unique index to each of the available test variants.
In most cases, the user does not need to be aware of all the internals related to this variant indexing, since ReFrame will run by default all the available variants for each of the registered tests.
On the other hand, in more complex use cases such as setting dependencies across different test variants, or when performing some complex variant sub-selection on a fixture declaration, the user may need to access some of this low-level information related to the variant indexing.
Therefore, classes that derive from the base :class:`~reframe.core.pipeline.RegressionMixin` provide `classmethods` and properties to query these data.

.. warning::
  When selecting test variants through their variant index, no index ordering should ever be assumed, being the user's responsibility to ensure on each ReFrame run that the selected index corresponds to the desired parameter and/or fixture variants.

.. py:attribute:: RegressionMixin.num_variants

   Total number of variants of the test.

.. automethod:: reframe.core.pipeline.RegressionMixin.get_variant_nums

.. automethod:: reframe.core.pipeline.RegressionMixin.variant_name


------------------------
Environments and Systems
------------------------

.. automodule:: reframe.core.environments
   :members: Environment, ProgEnvironment, _EnvironmentSnapshot, snapshot
   :show-inheritance:

.. automodule:: reframe.core.systems
   :members:
   :show-inheritance:

-------------------------------------
Job Schedulers and Parallel Launchers
-------------------------------------

.. automodule:: reframe.core.schedulers
   :members:
   :show-inheritance:

.. automodule:: reframe.core.launchers
   :members:
   :show-inheritance:


.. autofunction:: reframe.core.backends.getlauncher(name)

   Retrieve the :class:`reframe.core.launchers.JobLauncher` concrete
   implementation for a parallel launcher backend.

   :arg name: The registered name of the launcher backend.


.. autofunction:: reframe.core.backends.getscheduler(name)

   Retrieve the :class:`reframe.core.schedulers.JobScheduler` concrete
   implementation for a scheduler backend.

   :arg name: The registered name of the scheduler backend.

----------------
Runtime Services
----------------

.. automodule:: reframe.core.runtime
   :members:
   :show-inheritance:


---------------
Modules Systems
---------------

.. autoclass:: reframe.core.modules.ModulesSystem
   :members:
   :show-inheritance:


-------------
Build Systems
-------------

.. versionadded:: 2.14

ReFrame delegates the compilation of the regression test to a `build system`.
Build systems in ReFrame are entities that are responsible for generating the necessary shell commands for compiling a code.
Each build system defines a set of attributes that users may set in order to customize their compilation.
An example usage is the following:

.. code:: python

  self.build_system = 'SingleSource'
  self.build_system.cflags = ['-fopenmp']

Users simply set the build system to use in their regression tests and then they configure it.
If no special configuration is needed for the compilation, users may completely ignore the build systems.
ReFrame will automatically pick one based on the regression test attributes and will try to compile the code.

All build systems in ReFrame derive from the abstract base class :class:`reframe.core.buildsystems.BuildSystem`.
This class defines a set of common attributes, such us compilers, compilation flags etc. that all subclasses inherit.
It is up to the concrete build system implementations on how to use or not these attributes.

.. automodule:: reframe.core.buildsystems
   :members:
   :exclude-members: BuildSystemField
   :show-inheritance:


.. _container-platforms:

-------------------
Container Platforms
-------------------

.. versionadded:: 2.20

.. automodule:: reframe.core.containers
   :members:
   :exclude-members: ContainerPlatformField
   :show-inheritance:

----------------------------
The :py:mod:`reframe` module
----------------------------

The :py:mod:`reframe` module offers direct access to the basic test classes, constants and decorators.


.. py:class:: reframe.CompileOnlyRegressionTest

   See :class:`reframe.core.pipeline.CompileOnlyRegressionTest`.


.. py:class:: reframe.RegressionTest

   See :class:`reframe.core.pipeline.RegressionTest`.


.. py:class:: reframe.RunOnlyRegressionTest

   See :class:`reframe.core.pipeline.RunOnlyRegressionTest`.

.. py:attribute:: reframe.DEPEND_BY_ENV

   See :attr:`reframe.core.pipeline.DEPEND_BY_ENV`.


.. py:attribute:: reframe.DEPEND_EXACT

   See :attr:`reframe.core.pipeline.DEPEND_EXACT`.


.. py:attribute:: reframe.DEPEND_FULLY

   See :attr:`reframe.core.pipeline.DEPEND_FULLY`.


.. py:decorator:: reframe.parameterized_test

   See :func:`@reframe.core.decorators.parameterized_test <reframe.core.decorators.parameterized_test>`.


.. py:decorator:: reframe.require_deps

   .. deprecated:: 3.7.0
      Please use the :func:`~reframe.core.pipeline.RegressionMixin.require_deps` built-in function


.. py:decorator:: reframe.required_version

   See :func:`@reframe.core.decorators.required_version <reframe.core.decorators.required_version>`.


.. py:decorator:: reframe.run_after

   .. deprecated:: 3.7.0
      Please use the :func:`~reframe.core.pipeline.RegressionMixin.run_after` built-in function


.. py:decorator:: reframe.run_before

   .. deprecated:: 3.7.0
      Please use the :func:`~reframe.core.pipeline.RegressionMixin.run_before` built-in function


.. py:decorator:: reframe.simple_test

   See :func:`@reframe.core.decorators.simple_test <reframe.core.decorators.simple_test>`.



.. _scheduler_options:

----------------------------------------------------
Mapping of Test Attributes to Job Scheduler Backends
----------------------------------------------------

.. table::
   :align: left

   ============================ ============================= ========================================================================================== ==================
   Test attribute               Slurm option                  Torque option                                                                              PBS option
   ============================ ============================= ========================================================================================== ==================
   :attr:`num_tasks`            :obj:`--ntasks`:sup:`1`       :obj:`-l nodes={num_tasks//num_tasks_per_node}:ppn={num_tasks_per_node*num_cpus_per_task}` :obj:`-l select={num_tasks//num_tasks_per_node}:mpiprocs={num_tasks_per_node}:ncpus={num_tasks_per_node*num_cpus_per_task}`
   :attr:`num_tasks_per_node`   :obj:`--ntasks-per-node`      see :attr:`num_tasks`                                                                      see :attr:`num_tasks`
   :attr:`num_tasks_per_core`   :obj:`--ntasks-per-core`      n/a                                                                                        n/a
   :attr:`num_tasks_per_socket` :obj:`--ntasks-per-socket`    n/a                                                                                        n/a
   :attr:`num_cpus_per_task`    :obj:`--cpus-per-task`        see :attr:`num_tasks`                                                                      see :attr:`num_tasks`
   :attr:`time_limit`           :obj:`--time=hh:mm:ss`        :obj:`-l walltime=hh:mm:ss`                                                                :obj:`-l walltime=hh:mm::ss`
   :attr:`exclusive_access`     :obj:`--exclusive`            n/a                                                                                        n/a
   :attr:`use_smt`              :obj:`--hint=[no]multithread` n/a                                                                                        n/a
   ============================ ============================= ========================================================================================== ==================


If any of the attributes is set to :class:`None` it will not be emitted at all in the job script.
In cases that the attribute is required, it will be set to ``1``.

:sup:`1` The :obj:`--nodes` option may also be emitted if the :js:attr:`use_nodes_option <config_reference.html#schedulers-.use_nodes_option>` scheduler configuration parameter is set.
