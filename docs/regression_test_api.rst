====================
Regression Tests API
====================

This page provides a reference guide of the ReFrame API for writing regression tests covering all the relevant details.
Internal data structures and APIs are covered only to the extent that this might be helpful to the final user of the framework.



Regression Test Base Classes
----------------------------

.. automodule:: reframe.core.pipeline
   :members:
   :show-inheritance:



Regression Test Class Decorators
--------------------------------

.. autodecorator:: reframe.core.decorators.parameterized_test(*inst)

.. autodecorator:: reframe.core.decorators.required_version(*versions)

.. autodecorator:: reframe.core.decorators.simple_test


Pipeline Hooks
--------------

.. autodecorator:: reframe.core.decorators.run_after(stage)

.. autodecorator:: reframe.core.decorators.run_before(stage)

.. autodecorator:: reframe.core.decorators.require_deps



Builtins
--------

ReFrame provides built-in functions that facilitate the creation of extensible tests (i.e. a test library).
These *builtins* are intended to be used directly in the class body of the test, allowing the ReFrame internals to *pre-process* their input before the actual test creation takes place.
This provides the ReFrame internals with further control over the user's input, making the process of writing regression tests less error-prone thanks to a better error checking.
In essence, these builtins exert control over the test creation, and they allow adding and/or modifying certain attributes of the regression test.

.. versionadded:: 3.5

.. py:function:: reframe.core.pipeline.RegressionTest.parameter(values=None, inherit_params=False, filter_params=None)

  Inserts or modifies a regression test parameter.
  If a parameter with a matching name is already present in the parameter space of a parent class, the existing parameter values will be combined with those provided by this method following the inheritance behavior set by the arguments ``inherit_params`` and ``filter_params``.
  Instead, if no parameter with a matching name exists in any of the parent parameter spaces, a new regression test parameter is created.
  A regression test can be parametrized as follows:

  .. code:: python

    class Foo(rfm.RegressionTest):
        variant = parameter(['A', 'B'])

        def __init__(self):
            if self.variant == 'A':
                do_this()
            else:
                do_other()

  One of the most powerful features about these built-in functions is that they store their input information at the class level. 
  This means if one were to extend or specialize an existing regression test, the test attribute additions and modifications made through built-in functions in the parent class will be automatically inherited by the child test.
  For instance, continuing with the example above, one could override the :func:`__init__` method in the :class:`MyTest` regression test as follows:

  .. code:: python

    class Bar(Foo):

        def __init__(self):
            if self.variant == 'A':
                override_this()
            else:
                override_other()

  Note that this built-in parameter function provides an alternative method to parameterize a test to :func:`reframe.core.decorators.parameterized_test`, and the use of both approaches in the same test is currently disallowed.
  The two main advantages of the built-in :func:`parameter` over the decorated approach reside in the parameter inheritance across classes and the handling of large parameter sets.
  As shown in the example above, the parameters declared with the built-in :func:`parameter` are automatically carried over into derived tests through class inheritance, whereas tests using the decorated approach would have to redefine the parameters on every test. 
  Similarly, parameters declared through the built-in :func:`parameter` are regarded as fully independent from each other and ReFrame will automatically generate as many tests as available parameter combinations. This is a major advantage over the decorated approach, where one would have to manually expand the parameter combinations.
  This is illustrated in the example below, consisting of a case with two parameters, each having two possible values.

  .. code:: python

    # Parameterized test with two parameters (p0 = ['a', 'b'] and p1 = ['x', 'y'])
    @rfm.parameterized_test(['a','x'], ['a','y'], ['b','x'], ['b', 'y'])
    class Foo(rfm.RegressionTest):
      def __init__(self, p0, p1):
        do_something(p0, p1)

    # This is easier to write with the parameter built-in.
    @rfm.simple_test
    class Bar(rfm.RegressionTest):
      p0 = parameter(['a', 'b'])
      p1 = parameter(['x', 'y'])

      def __init__(self):
        do_something(self.p0, self.p1)


  :param values: A list containing the parameter values.
     If no values are passed when creating a new parameter, the parameter is considered as *declared* but not *defined* (i.e. an abstract parameter).
     Instead, for an existing parameter, this depends on the parameter's inheritance behaviour and on whether any values where provided in any of the parent parameter spaces.
  :param inherit_params: If :obj:`False`, no parameter values that may have been defined in any of the parent parameter spaces will be inherited.
  :param filter_params: Function to filter/modify the inherited parameter values that may have been provided in any of the parent parameter spaces.
     This function must accept a single argument, which will be passed as an iterable containing the inherited parameter values.
     This only has an effect if used with ``inherit_params=True``.


.. py:function:: reframe.core.pipeline.RegressionTest.variable(*types, value=None, field=None)

  Inserts a new regression test variable.
  Declaring a test variable through the :func:`variable` built-in allows for a more robust test implementation than if the variables were just defined as regular test attributes by direct assignment in the body of the test (e.g. ``self.a = 10``).
  Using variables declared through the :func:`variable` built-in guarantees that these regression test variables will not be redeclared by any child class, while also ensuring that any values that may be assigned to such variables comply with its original declaration. 
  In essence, by using test variables, the user removes any potential test errors that might be caused by accidentally overriding a class attribute. See the example below.


  .. code:: python

    class Foo(rfm.RegressionTest):
        my_var = variable(int, value=8)
        not_a_var = 4

        def __init__(self):
            print(self.my_var) # prints 8.
            # self.my_var = 'override' # Error: my_var must be an int!
            self.not_a_var = 'override' # However, this would work. Dangerous!

  The argument ``value`` in the :func:`variable` built-in sets the default value for the variable.
  As mentioned above, a variable may not be declared more than once, but its value can be updated by simply assigning it a new value directly in the class body.

  .. code:: python

    class Bar(Foo):
        my_var = 4
        # my_var = 'override' # Error again!

        def __init__(self):
            print(self.my_var) # prints 4.

  Here, the class :class:`Bar` inherits the variables from :class:`Foo` and can see that ``my_var`` has already been declared in the parent class. Therefore, the value of ``my_var`` is updated ensuring that the new value complies to the original variable declaration.

  These examples above assumed that a default value can be provided to the variables in the bases tests, but that might not always be the case.
  For example, when writing a test library, one might want to leave some variables undefined and force the user to set these when using the test.
  As shown in the example below, imposing such requirement is as simple as not passing any ``value`` to the :func:`variable` built-in, which marks the given variable as ``required``.

  .. code:: python

    # Test as written in the library
    class EchoBaseTest(rfm.RunOnlyRegressionTest):
      what = variable(str)

      def __init__(self):
          self.valid_systems = ['*']
          self.valid_prog_environs = ['PrgEnv-gnu']
          self.executable = f'echo {self.what}'
          self.sanity_patterns = sn.assert_found(fr'{self.what}')

    # Test as written by the user:
    @rfm.simple_test
    class HelloTest(EchoBaseTest):
      what = 'Hello'

    # A parametrized test with type-checking
    @rfm.simple_test
    class FoodTest(EchoBaseTest):
      param = parameter(['Bacon', 'Eggs'])

      def __init__(self):
        self.what = self.param
        super().__init__()


  Similarly to a variable with a value already assigned to it, the value of a required variable may be set either directly in the class body, on the :func:`__init__` method, or in any other hook before it is referenced.
  Otherwise an error will be raised indicating that a required variable has not been set.
  Conversely, a variable with a default value already assigned to it can be made required by assigning it the ``required`` keyword.

  .. code:: python
    class MyRequiredTest(HelloTest):
      what = required 

  Running the above test will cause the :func:`__init__` method from :class:`EchoBaseTest` to throw an error indicating that the variable ``what`` has not been set.

  :param types: the supported types for the variable.
  :param value: the default value assigned to the variable. If no value is provided, the variable is set as ``required``.
  :param field: the field validator to be used for this variable.
      If no field argument is provided, it defaults to
      :class:`reframe.core.fields.TypedField`.
      Note that the field validator provided by this argument must derive from
      :class:`reframe.core.fields.Field`.


Environments and Systems
------------------------

.. automodule:: reframe.core.environments
   :members: Environment, ProgEnvironment, _EnvironmentSnapshot, snapshot
   :show-inheritance:

.. automodule:: reframe.core.systems
   :members:
   :show-inheritance:


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

Runtime Services
----------------

.. automodule:: reframe.core.runtime
   :members:
   :show-inheritance:


Modules Systems
---------------

.. autoclass:: reframe.core.modules.ModulesSystem
   :members:
   :show-inheritance:


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

Container Platforms
-------------------

.. versionadded:: 2.20

.. automodule:: reframe.core.containers
   :members:
   :exclude-members: ContainerPlatformField
   :show-inheritance:


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

   See :func:`@reframe.core.decorators.require_deps <reframe.core.decorators.require_deps>`.


.. py:decorator:: reframe.required_version

   See :func:`@reframe.core.decorators.required_version <reframe.core.decorators.required_version>`.


.. py:decorator:: reframe.run_after

   See :func:`@reframe.core.decorators.run_after <reframe.core.decorators.run_after>`.


.. py:decorator:: reframe.run_before

   See :func:`@reframe.core.decorators.run_before <reframe.core.decorators.run_before>`.


.. py:decorator:: reframe.simple_test

   See :func:`@reframe.core.decorators.simple_test <reframe.core.decorators.simple_test>`.



.. _scheduler_options:

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
