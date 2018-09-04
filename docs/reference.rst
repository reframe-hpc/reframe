===============
Reference Guide
===============

This page provides a reference guide of the ReFrame API for writing regression tests covering all the relevant details.
Internal data structures and APIs are covered only to the extent that might be helpful to the final user of the framework.



Regression test classes and related utilities
---------------------------------------------

.. class:: reframe.RegressionTest(name=None, prefix=None)

   This is an alias of :class:`reframe.core.pipeline.RegressionTest`.

   .. versionadded:: 2.13


.. class:: reframe.RunOnlyRegressionTest(*args, **kwargs)

   This is an alias of :class:`reframe.core.pipeline.RunOnlyRegressionTest`.

   .. versionadded:: 2.13


.. class:: reframe.CompileOnlyRegressionTest(*args, **kwargs)

   This is an alias of :class:`reframe.core.pipeline.CompileOnlyRegressionTest`.

   .. versionadded:: 2.13


.. py:decorator:: reframe.simple_test

   This is an alias of :func:`reframe.core.decorators.simple_test`.

   .. versionadded:: 2.13


.. py:decorator:: reframe.parameterized_test(inst=[])

   This is an alias of :func:`reframe.core.decorators.parameterized_test`.

   .. versionadded:: 2.13


.. automodule:: reframe.core.decorators
   :members:
   :show-inheritance:

.. automodule:: reframe.core.pipeline
   :members:
   :show-inheritance:


Environments and Systems
------------------------

.. automodule:: reframe.core.environments
   :members:
   :show-inheritance:

.. automodule:: reframe.core.systems
   :members:
   :show-inheritance:


Job schedulers and parallel launchers
-------------------------------------

.. autoclass:: reframe.core.schedulers.Job
   :members:
   :show-inheritance:

.. automodule:: reframe.core.launchers
   :members:
   :show-inheritance:


.. automodule:: reframe.core.launchers.registry
   :members:
   :show-inheritance:


Runtime services
----------------

.. automodule:: reframe.core.runtime
   :members:
   :exclude-members: temp_runtime, switch_runtime
   :show-inheritance:


Modules System API
------------------

.. autoclass:: reframe.core.modules.ModulesSystem


Build systems
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
