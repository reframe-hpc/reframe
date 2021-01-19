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


.. _directives:

Directives
----------

Directives are functions that can be called directly in the body of a ReFrame regression test class. 
These functions exert control over the test creation, and they allow adding and/or modifying certain attributes of the regression test.
For example, a test can be parameterized using the :func:`parameter` directive as follows:

.. code:: python

    class MyTest(rfm.RegressionTest):
        parameter('variant', 'A', 'B')
 
        def __init__(self):
            if self.variant == 'A':
                do_this()
            else:
                do_other()

One of the most powerful features about using directives is that they store their input information at the class level. 
This means if one were to extend or specialize an existing regression test, the test attribute additions and modifications made throught directives in the parent class will be automatically inherited by the child test. 
For instance, continuing with the example above, one could override the ``__init__`` method in the ``MyTest`` regression test as follows:

.. code:: python

    class MyModifiedTest(MyTest):

        def __init__(self):
            if self.variant == 'A':
                override_this()
            else:
                override_other()


.. py:function:: reframe.core.pipeline.RegressionTest.parameter(name, *values, inherit_params=False, filter_params=None)

   Inserts or modifies a regression test parameter.
   If a parameter with a matching name is already present in the parameter space of a parent class, the existing parameter values will be combined with those provided by this method following the inheritance behaviour set by the arguments ``inherit_params`` and ``filt_params``.
   Instead, if no parameter with a matching name exist in any of the parent parameter spaces, a new regression test parameter is created.

   :param name: parameter name.
   :param values: parameter values.
       If no values are passed when creating a new parameter, the parameter is considered as declared but not defined (i.e. an abstract parameter).
       Instead, for an existing parameter, this depends on the parameter's inheritance behaviour and on whether any values where provided in any of the parent parameter spaces.
   :param inherit_params: If false, no parameter values that may have been defined in any of the parent parameter spaces will be inherited.
   :param filter_params: Function to filter/modify the inherited parameter values that may have been provided in any of the parent parameter spaces.
       This function must only expect a tuple containing the inherited parameter values as its only argument.
       This only has an effect if used with ``inherit_params=True``.




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
