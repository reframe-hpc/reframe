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
  :param `**kwargs`: *kwargs* to be forwarded to the constructor of the field validator.


Pipeline Hooks
--------------

ReFrame provides built-in functions that allow attaching arbitrary functions to run before and/or after a given stage of the execution pipeline.
Once attached to a given stage, these functions are referred to as *pipeline hooks*.
A hook may be attached to multiple pipeline stages and multiple hooks may also be attached to the same pipeline stage.
Pipeline hooks attached to multiple stages will be executed on each pipeline stage the hook was attached to.
Pipeline stages with multiple hooks attached will execute these hooks in the order in which they were attached to the given pipeline stage.
A derived class will inherit all the pipeline hooks defined in its bases, except for those whose hook function is overridden by the derived class.
A function that overrides a pipeline hook from any of the base classes will not be a pipeline hook unless the overriding function is explicitly reattached to any pipeline stage.
In the event of a name clash arising from multiple inheritance, the inherited pipeline hook will be chosen following Python's `MRO <https://docs.python.org/3/library/stdtypes.html#class.__mro__>`_.

A function may be attached to any of the following stages (listed in order of execution): ``init``, ``setup``, ``compile``, ``run``, ``sanity``, ``performance`` and ``cleanup``.
The ``init`` stage refers to the test's instantiation and it runs before entering the execution pipeline.
Therefore, a test function cannot be attached to run before the ``init`` stage.
Hooks attached to any other stage will run exactly before or after this stage executes.
So although a "post-init" and a "pre-setup" hook will both run *after* a test has been initialized and *before* the test goes through the first pipeline stage, they will execute in different times:
the post-init hook will execute *right after* the test is initialized.
The framework will then continue with other activities and it will execute the pre-setup hook *just before* it schedules the test for executing its setup stage.

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


Built-in functions
------------------

.. py:decorator:: RegressionMixin.sanity_function(func)

  Decorate a member function as the sanity function of the test.

  This decorator will convert the given function into a :func:`~RegressionMixin.deferrable` and mark it to be executed during the test's sanity stage.
  When this decorator is used, manually assigning a value to :attr:`~RegressionTest.sanity_patterns` in the test is not allowed.

  Decorated functions may be overridden by derived classes, and derived classes may also decorate a different method as the test's sanity function.
  Decorating multiple member functions in the same class is not allowed.
  However, a :class:`RegressionTest` may inherit from multiple :class:`RegressionMixin` classes with their own sanity functions.
  In this case, the derived class will follow Python's `MRO <https://docs.python.org/3/library/stdtypes.html#class.__mro__>`_ to find a suitable sanity function.

  .. versionadded:: 3.7.0

.. py:decorator:: RegressionMixin.deferrable(func)

  Converts the decorated method into a deferrable function.

  See :ref:`deferrable-functions` for further information on deferrable functions.

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
