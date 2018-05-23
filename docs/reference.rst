===============
Reference Guide
===============

This page provides a reference guide of the ReFrame API for writing regression tests covering all the relevant details.
Internal data structures and APIs are covered only to the extent that might be helpful to the final user of the framework.



Regression test classes and related utilities
---------------------------------------------

.. py:decorator:: reframe.core.decorators.simple_test

   Class decorator for registering parameterless tests with ReFrame.

   The decorated class must derive from :class:`reframe.core.pipeline.RegressionTest`.
   This decorator is also available directly under the :mod:`reframe` module.

   .. versionadded:: 2.13


.. py:decorator:: reframe.core.decorators.parameterized_test(inst=[])

   Class decorator for registering multiple instantiations of a test class.

   The decorated class must derive from :class:`reframe.core.pipeline.RegressionTest`.
   This decorator is also available directly under the :mod:`reframe` module.

   :arg inst: An iterable of the argument lists of the difference instantiations.
              Instantiation arguments may also be passed as keyword dictionaries.

   .. versionadded:: 2.13

   .. note::
      This decorator does not instantiate any test.
      It only registers them.
      The actual instantiation happens during the loading phase of the test.


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


.. py:decorator:: reframe.core.launchers.registry.register_launcher(name, local=False)

    .. versionadded:: 2.8

    Class decorator for registering new job launchers.

    .. epigraph::
       *This decorator is only relevant to developers of new job launchers.*

    :arg name: The registration name of this launcher
    :arg local: :class:`True` if launcher may only submit local jobs,
        :class:`False` otherwise.
    :raises ValueError: if a job launcher is already registered with
        the same name.


Runtime services
----------------

.. automodule:: reframe.core.runtime
   :members:
   :show-inheritance:


Modules System API
------------------

.. automodule:: reframe.core.modules
   :members:
   :show-inheritance:
