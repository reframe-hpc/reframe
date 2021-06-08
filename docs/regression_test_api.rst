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


--------
Builtins
--------

.. versionadded:: 3.4.2

ReFrame provides built-in types and functions which facilitate the process of writing extensible regression tests (i.e. a test library).
These *builtins* are only available when used directly in the class body of classes derived from any of the :ref:`regression-bases`.
Through builtins, ReFrame internals are able to *pre-process* and validate the test input before the actual test creation takes place.
This provides the ReFrame internals with further control over the user's input, making the process of writing regression tests less error-prone.
In essence, these builtins exert control over the test creation, and they allow adding and/or modifying certain attributes of the regression test.


Built-in types
--------------

.. py:function:: RegressionMixin.parameter(values=None, inherit_params=False, filter_params=None)

  Inserts or modifies a regression test parameter.
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


  :param values: A list containing the parameter values.
     If no values are passed when creating a new parameter, the parameter is considered as *declared* but not *defined* (i.e. an abstract parameter).
     Instead, for an existing parameter, this depends on the parameter's inheritance behaviour and on whether any values where provided in any of the parent parameter spaces.
  :param inherit_params: If :obj:`False`, no parameter values that may have been defined in any of the parent parameter spaces will be inherited.
  :param filter_params: Function to filter/modify the inherited parameter values that may have been provided in any of the parent parameter spaces.
     This function must accept a single argument, which will be passed as an iterable containing the inherited parameter values.
     This only has an effect if used with ``inherit_params=True``.


.. py:function:: RegressionMixin.variable(*types, value=None)

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
      def set_exec_and_sanity(self):
          self.executable = f'echo {self.what}'
          self.sanity_patterns = sn.assert_found(fr'{self.what}')


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

  .. code:: python

    class MyRequiredTest(HelloTest):
      what = required


  Running the above test will cause the :func:`set_exec_and_sanity` hook from :class:`EchoBaseTest` to throw an error indicating that the variable ``what`` has not been set.

  :param types: the supported types for the variable.
  :param value: the default value assigned to the variable. If no value is provided, the variable is set as ``required``.
  :param field: the field validator to be used for this variable.
      If no field argument is provided, it defaults to
      :class:`reframe.core.fields.TypedField`.
      Note that the field validator provided by this argument must derive from
      :class:`reframe.core.fields.Field`.


Pipeline Hooks
--------------

Pipeline hooks is a type of built-in functions that provide an easy way to perform operations while the test traverses the execution pipeline.
You can attach arbitrary functions to run before or after any pipeline stage, which are called *pipeline hooks*.
Multiple hooks can be attached before or after the same pipeline stage, in which case the order of execution will match the order in which the functions are defined in the class body of the test.
A single hook can also be applied to multiple stages and it will be executed multiple times.
All pipeline hooks of a test class are inherited by its subclasses.
Subclasses may override a pipeline hook of their parents by redefining the hook function and re-attaching it at the same pipeline stage.
There are seven pipeline stages where you can attach test methods: ``init``, ``setup``, ``compile``, ``run``, ``sanity``, ``performance`` and ``cleanup``.
The ``init`` stage is not a real pipeline stage, but it refers to the test initialization.

Hooks attached to any stage will run exactly before or after this stage executes.
So although a "post-init" and a "pre-setup" hook will both run *after* a test has been initialized and *before* the test goes through the first pipeline stage, they will execute in different times:
the post-init hook will execute *right after* the test is initialized.
The framework will then continue with other activities and it will execute the pre-setup hook *just before* it schedules the test for executing its setup stage.

.. note::
   Pipeline hooks were introduced in 2.20 but since 3.6.2 can be declared using the regression test built-in function described in this section.

.. warning::
   .. versionchanged:: 3.7.0
      Declaring pipeline hooks using the same name functions from the :py:mod:`reframe` or :py:mod:`reframe.core.decorators` modules is now deprecated.
      You should use the built-in functions described in this section instead.

.. py:decorator:: RegressionMixin.run_before(stage)

  Decorator for attaching a test method to a pipeline stage.

  The method will run just before the specified pipeline stage and it should not accept any arguments except ``self``.
  This decorator can be stacked, in which case the function will be attached to multiple pipeline stages.
  The ``stage`` argument can be any of ``'setup'``, ``'compile'``, ``'run'``, ``'sanity'``, ``'performance'`` or ``'cleanup'``.


.. py:decorator:: RegressionMixin.run_after(stage)

  Decorator for attaching a test method to a pipeline stage.

  This is analogous to :func:`~RegressionTest.run_before`, except that ``'init'`` can also be used as the ``stage`` argument.
  In this case, the hook will execute right after the test is initialized (i.e. after the :func:`__init__` method is called), before entering the test's pipeline.
  In essence, a post-init hook is equivalent to defining additional :func:`__init__` functions in the test.
  All the other properties of pipeline hooks apply equally here.
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
     Add the ability to define post-init hooks in tests.


Built-in functions
------------------

.. py:decorator:: RegressionMixin.sanity_function(func)

  Shorthand for assigning a member function as the sanity function of the test.

  This decorator will convert the decorated method into a :func:`~RegressionMixin.deferrable` and mark it to be executed during the test's sanity stage.
  This syntax removes the need of setting the variable :attr:`~RegressionTest.sanity_patterns` in the test.
  In fact, when this decorator is used, manually setting :attr:`~RegressionTest.sanity_patterns` in the test is not allowed.

  Decorated functions may be overridden by derived classes, and derived classes may also decorate a different method as the test's sanity function.
  Decorating multiple member functions in the same class is not allowed.
  However, a :class:`RegressionTest` may inherit from multiple :class:`RegressionMixin` classes with their own sanity functions.
  In this case, the derived class will follow Python's `MRO <https://docs.python.org/3/library/stdtypes.html#class.__mro__>`_ to find a suitable sanity function.

  .. warning:: Not to be mistaken with :func:`reframe.utility.sanity.sanity_function`.
  .. versionadded:: 3.7.0

.. py:decorator:: RegressionMixin.deferrable(func)

  Converts the decorated method into a deferrable expression (see :ref:`deferrable-functions`).

  This decorator is equivalent to :func:`reframe.utility.sanity.sanity_function`.

  .. versionadded:: 3.7.0

.. py:function:: RegressionMixin.bind(func, name=None)

  Bind a free function to a regression test.

  By default, the function is bound with the same name as the free function.
  However, the function can be bound using a different name with the ``name`` argument.

  :param func: external function to be bound to a class.
  :param name: bind the function under a different name.

  .. versionadded:: 3.6.2

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
