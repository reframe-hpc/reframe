======================
Migrating to ReFrame 3
======================

ReFrame 3 brings substantial changes in its configuration.
The configuration component was completely revised and rewritten from scratch in order to allow much more flexibility in how the framework's configuration options are handled, as well as to ensure the maintainability of the framework in the future.

At the same time, ReFrame 3 deprecates some common pre-2.20 test syntax in favor of the more modern and intuitive pipeline hooks.

This guide details the necessary steps in order to easily migrate to ReFrame 3.


Updating Your Site Configuration
--------------------------------

As described in `Configuring ReFrame for Your Site <configure.html>`__, ReFrame's configuration file has changed substantially.
However, you don't need to manually update your configuration; ReFrame will do that automatically for you.
As soon as it detects an old-style configuration file, it will convert it to the new syntax save it in a temporary file:


.. code-block:: none

   $ ./bin/reframe -C unittests/resources/settings_old_syntax.py -l
   ./bin/reframe: the syntax of the configuration file 'unittests/resources/settings_old_syntax.py' is deprecated
   ./bin/reframe: configuration file has been converted to the new syntax here: '/var/folders/h7/k7cgrdl13r996m4dmsvjq7v80000gp/T/tmph5n8u3kf.py'

Alternatively, you can convert any old configuration file using the conversion tool ``tools/convert_config.py``:

.. code-block:: none

   $ ./tools/convert-config unittests/resources/settings_old_syntax.py
   Conversion successful! The converted file can be found at '/var/folders/h7/k7cgrdl13r996m4dmsvjq7v80000gp/T/tmpz4f6yer4.py'.


Another important change is that default locations for looking up a configuration file has changed (see `Configuring ReFrame for Your Site <configure.html>`__ for more details).
That practically means that if you were relying on ReFrame loading your ``reframe/settings.py`` by default, this is no longer true.
You have to move it to any of the default settings locations or set the corresponding command line option or environment variable.


Automatic conversion limitations
================================

ReFrame does a pretty good job in converting correctly your old configuration files, but there are some limitations:

- Your code formatting will be lost.
  ReFrame will use its own, which is PEP8 compliant nonetheless.
- Any comments will be lost.
- Any code that was used to dynamically generate configuration parameters will be lost.
  ReFrame will generate the new configuration based on what was the actual old configuration after any dynamic generation.



.. note::

   The very old logging configuration syntax (prior to ReFrame 2.13) is no more recognized and the configuration conversion tool does not take it into account.


Updating Your Tests
-------------------


ReFrame 2.20 introduced a new powerful mechanism for attaching arbitrary functions hooks at the different pipeline stages.
This mechanism provides an easy way to configure and extend the functionality of a test, eliminating essentially the need to override pipeline stages for this purpose.
ReFrame 3.0 deprecates the old practice of overriding pipeline stage methods in favor of using pipeline hooks.
In the old syntax, it was quite common to override the ``setup()`` method, in order to configure your test based on the current programming environment or the current system partition.
The following is a typical example of that:


.. code:: python

   def setup(self, partition, environ, **job_opts):
       if environ.name == 'gnu':
           self.build_system.cflags = ['-fopenmp']
       elif environ.name == 'intel':
           self.build_system.cflags = ['-qopenmp']

       super().setup(partition, environ, **job_opts)


Alternatively, this example could have been written as follows:

.. code:: python

   def setup(self, partition, environ, **job_opts):
       super().setup(partition, environ, **job_opts)
       if self.current_environ.name == 'gnu':
           self.build_system.cflags = ['-fopenmp']
       elif self.current_environ.name == 'intel':
           self.build_system.cflags = ['-qopenmp']


This syntax now issues a deprecation warning.
Rewriting this using pipeline hooks is quite straightforward and leads to nicer and more intuitive code:

.. code:: python

   @rfm.run_before('compile')
   def setflags(self):
       if self.current_environ.name == 'gnu':
           self.build_system.cflags = ['-fopenmp']
       elif self.current_environ.name == 'intel':
           self.build_system.cflags = ['-qopenmp']


You could equally attach this function to run after the "setup" phase with ``@rfm.run_after('setup')``, as in the original example, but attaching it to the "compile" phase makes more sense.
However, you can't attach this function *before* the "setup" phase, because the ``current_environ`` will not be available and it will be still ``None``.


Force override a pipeline method
================================

Although pipeline hooks should be able to cover almost all the cases for writing tests in ReFrame, there might be corner cases that you need to override one of the pipeline methods, e.g., because you want to implement a stage differently.
In this case, all you have to do is mark your test class as "special", and ReFrame will not issue any deprecation warning if you override pipeline stage methods:

.. code:: python

   class MyExtendedTest(rfm.RegressionTest, special=True):
       def setup(self, partition, environ, **job_opts):
           # do your custom stuff
           super().setup(partition, environ, **job_opts)


If you try to override the ``setup()`` method in any of the subclasses of ``MyExtendedTest``, you will still get a deprecation warning, which a desired behavior since the subclasses should be normal tests.


Suppressing deprecation warnings
================================

Although not recommended, you can suppress any deprecation warning issued by ReFrame by passing the ``--no-deprecation-warnings`` flag.


Getting schedulers and launchers by name
========================================


The way to get a scheduler or launcher instance by name has changed.
Prior to ReFrame 3, this was written as follows:

.. code:: python

	 from reframe.core.launchers.registry import getlauncher


	 class MyTest(rfm.RegressionTest):
	     ...

	     @rfm.run_before('run')
	     def setlauncher(self):
	         self.job.launcher = getlauncher('local')()



Now you have to simply replace the import statement with the following:


.. code:: python

	 from reframe.core.backends import getlauncher


Similarly for schedulers, the ``reframe.core.schedulers.registry`` module must be replaced with ``reframe.core.backends``.


Other Changes
-------------

ReFrame 3.0-dev0 introduced a `change <https://github.com/eth-cscs/reframe/pull/1125>`__ in the way that a search path for checks was constructed in the command-line using the ``-c`` option.
ReFrame 3.0 reverts the behavior of the ``-c`` to its original one (i.e., ReFrame 2.x behavior), in which multiple paths can be specified by passing multiple times the ``-c`` option.
Overriding completely the check search path can be achieved in ReFrame 3.0 through the :envvar:`RFM_CHECK_SEARCH_PATH` environment variable or the corresponding configuration option.
