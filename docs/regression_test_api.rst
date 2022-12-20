==================
Test API Reference
==================

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

.. autodecorator:: reframe.core.decorators.simple_test


.. _builtins:

--------
Builtins
--------

.. versionadded:: 3.4.2

ReFrame test base classes and, in particular, the :class:`reframe.core.pipeline.RegressionMixin` class, define a set of functions and decorators that can be used to define essential test elements, such as variables, parameters, fixtures, pipeline hooks etc.
These are called *builtins* because they are directly available for use inside the test class body that is being defined without the need to import any module.
However, almost all of these builtins are also available from the :obj:`reframe.core.builtins` module.
The use of this module is required only when creating new tests programmatically using the :func:`~reframe.core.meta.make_test` function.

.. py:method:: reframe.core.pipeline.RegressionMixin.bind(func, name=None)

   Bind a free function to a regression test.

   By default, the function is bound with the same name as the free function.
   However, the function can be bound using a different name with the ``name`` argument.

   :param func: external function to be bound to a class.
   :param name: bind the function under a different name.

   .. note::
      This is the only builtin that is not available through the :obj:`reframe.core.builtins` module.
      The reason is that the :func:`bind` method needs to access the class namespace directly in order to bind the free function to the class.

   .. versionadded:: 3.6.2

.. autodecorator:: reframe.core.builtins.deferrable

.. autofunction:: reframe.core.builtins.fixture

.. autodecorator:: reframe.core.builtins.loggable_as(name)

.. autodecorator:: reframe.core.builtins.loggable

.. autofunction:: reframe.core.builtins.parameter

.. autodecorator:: reframe.core.builtins.performance_function

.. autodecorator:: reframe.core.builtins.require_deps

.. autodecorator:: reframe.core.builtins.run_after(stage)

.. autodecorator:: reframe.core.builtins.run_before(stage)

.. autodecorator:: reframe.core.builtins.sanity_function

.. autofunction:: reframe.core.builtins.variable


.. versionchanged:: 3.7.0
   Expose :func:`@deferrable <reframe.core.builtins.deferrable>` as a builtin.

.. versionchanged:: 3.11.0
   Builtins are now available also through the :obj:`reframe.core.builtins` module.


.. _pipeline-hooks:

--------------
Pipeline Hooks
--------------

ReFrame provides a mechanism to allow attaching arbitrary functions to run before or after a given stage of the execution pipeline.
This is achieved through the :func:`@run_before <reframe.core.builtins.run_before>` and :func:`@run_after <reframe.core.builtins.run_after>` test builtins.
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


.. seealso::
   - :func:`@run_before <reframe.core.builtins.run_before>`, :func:`@run_after <reframe.core.builtins.run_after>` decorators

.. note::
   Pipeline hooks do not execute in the test's stage directory, but in the directory that ReFrame executes in.
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
      You should use the builtin functions described in the :ref:`builtins` section..

   .. versionchanged:: 4.0.0
      Pipeline hooks can only be defined through the built-in functions described in this section.

.. warning::
   .. versionchanged:: 3.9.2

      Execution of pipeline hooks until this version was implementation-defined.
      In practice, hooks of a derived class were executed before those of its parents.

      This version defines the execution order of hooks, which now follows a strict reverse MRO order, so that parent hooks will execute before those of derived classes.
      Tests that relied on the execution order of hooks might break with this change.



.. _test-variants:

-------------
Test variants
-------------

Through the :func:`~reframe.core.builtins.parameter` and :func:`~reframe.core.builtins.fixture` builtins, a regression test may store multiple versions or `variants` of a regression test at the class level.
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

-------------------------
Dynamic Creation of Tests
-------------------------

.. versionadded:: 3.10.0


.. autofunction:: reframe.core.meta.make_test


------------------------
Environments and Systems
------------------------

.. automodule:: reframe.core.environments
   :members: Environment, ProgEnvironment, _EnvironmentSnapshot, snapshot
   :show-inheritance:

.. automodule:: reframe.core.systems
   :members:
   :show-inheritance:

---------------------------
Jobs and Parallel Launchers
---------------------------

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
