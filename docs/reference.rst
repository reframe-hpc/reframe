===============
Reference Guide
===============

This page provides a reference guide of the ReFrame API for writing regression tests covering all the relevant details.
Internal data structures and APIs are covered only to the extent that might be helpful to the final user of the framework.



Regression test classes and related utilities
---------------------------------------------

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


.. automodule:: reframe.core.backends
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


Container platforms
-------------------

.. versionadded:: 2.20

ReFrame can run a regression test inside a container.
To achieve that you have to set the :attr:`reframe.core.pipeline.RegressionTest.container_platform` attribute and then set up the container platform (e.g., image to load, commands to execute).
The :class:`reframe.core.ContainerPlatform` abstract base class define the basic interface and a minimal set of attributes that all concrete container platforms must implement.
Concrete container platforms may also define additional fields that are specific to them.

.. automodule:: reframe.core.containers
   :members:
   :exclude-members: ContainerPlatformField
   :show-inheritance:
