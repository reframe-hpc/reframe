======================
Migrating to ReFrame 3
======================


Updating your tests
-------------------


ReFrame 2.20 introduced a new powerful mechanism for attaching arbitrary functions hooks at the different pipeline stages.
This mechanism provides an easy way to configure and extend the functionality of a test, eliminating essentially the need to override pipeline stages for this purpose.
ReFrame 3.0 deprecates the old practice for overriding pipeline stage methods in favor of using pipeline hooks.
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

You can suppress any deprecation warning issued by ReFrame by passing the ``--no-deprecation-warnings`` flag.

