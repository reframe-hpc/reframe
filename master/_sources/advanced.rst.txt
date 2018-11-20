=====================================
Customizing Further a Regression Test
=====================================

In this section, we are going to show some more elaborate use cases of ReFrame.
Through the use of more advanced examples, we will demonstrate further customization options which modify the default options of the ReFrame pipeline.
The corresponding scripts as well as the source code of the examples discussed here can be found in the directory ``tutorial/advanced``.

Working with Makefiles
----------------------

We have already shown how you can compile a single source file associated with your regression test.
In this example, we show how ReFrame can leverage Makefiles to build executables.

Compiling a regression test through a Makefile is straightforward with ReFrame.
If the :attr:`sourcepath <reframe.core.pipeline.RegressionTest.sourcepath>` attribute refers to a directory, then ReFrame will automatically invoke ``make`` in that directory.
More specifically, ReFrame first copies the :attr:`sourcesdir` to the stage directory at the beginning of the compilation phase and then constructs the path ``os.path.join('{STAGEDIR}', self.sourcepath)`` to determine the actual compilation path.
If this is a directory, it will invoke ``make`` in it.

.. note::
   The :attr:`sourcepath <reframe.core.pipeline.RegressionTest.sourcepath>` attribute must be a relative path refering to a subdirectory of :attr:`sourcesdir <reframe.core.pipeline.RegressionTest.sourcesdir>`, i.e., relative paths starting with ``..`` will be rejected.

By default, :attr:`sourcepath <reframe.core.pipeline.RegressionTest.sourcepath>` is the empty string and :attr:`sourcesdir <reframe.core.pipeline.RegressionTest.sourcesdir>` is set to ``'src/'``.
As a result, by not specifying a :attr:`sourcepath <reframe.core.pipeline.RegressionTest.sourcepath>` at all, ReFrame will eventually compile the files found in the ``src/`` directory.
This is exactly what our first example here does.

For completeness, here are the contents of ``Makefile`` provided:

.. literalinclude:: ../tutorial/advanced/src/Makefile
  :language: makefile

The corresponding ``advanced_example1.c`` source file consists of a simple printing of a message, whose content depends on the preprocessor variable ``MESSAGE``:

.. literalinclude:: ../tutorial/advanced/src/advanced_example1.c
  :language: c

The purpose of the regression test in this case is to set the preprocessor variable ``MESSAGE`` via ``CPPFLAGS`` and then check the standard output for the message ``SUCCESS``, which indicates that the preprocessor flag has been passed and processed correctly by the Makefile.

The contents of this regression test are the following (``tutorial/advanced/advanced_example1.py``):

.. literalinclude:: ../tutorial/advanced/advanced_example1.py

The important bit here is how we set up the build system for this test:

.. literalinclude:: ../tutorial/advanced/advanced_example1.py
  :lines: 14-15
  :dedent: 4


First, we set the build system to :attr:`Make <reframe.core.buildsystems.Make>` and then set the preprocessor flags for the compilation.
ReFrame will invoke ``make`` as follows:

.. code::

  make -j CC='cc' CXX='CC' FC='ftn' NVCC='nvcc' CPPFLAGS='-DMESSAGE'

The compiler variables (``CC``, ``CXX`` etc.) are set based on the corresponding values specified in the `coniguration of the current environment <configure.html#environments-configuration>`__.
You may instruct the build system to ignore the default values from the environment by setting the following:

.. code-block:: python

  self.build_system.flags_from_environ = False

In this case, ``make`` will be invoked as follows:

.. code::

  make -j CPPFLAGS='-DMESSAGE'

Notice that the ``-j`` option is always generated.
If you want to limit build concurrency, you can do it as follows:

.. code-block:: python

  self.build_system.max_concurrency = 4


Finally, you may also customize the name of the ``Makefile``.
You can achieve that by setting the corresponding variable of the :class:`Make <reframe.core.buildsystems.Make>` build system:

.. code-block:: python

  self.build_system.makefile = 'Makefile_custom'


More details on ReFrame's build systems, you may find `here <reference.html#build-systems>`__.


Retrieving the source code from a Git repository
================================================

It might be the case that a regression test needs to clone its source code from a remote repository.
This can be achieved in two ways with ReFrame.
One way is to set the :attr:`sourcesdir` attribute to :class:`None` and explicitly clone or checkout a repository using the :attr:`prebuild_cmd <reframe.core.pipeline.RegressionTest.prebuild_cmd>`:

.. code-block:: python

   self.sourcesdir = None
   self.prebuild_cmd = ['git clone https://github.com/me/myrepo .']


By setting :attr:`sourcesdir` to :class:`None`, you are telling ReFrame that you are going to provide the source files in the stage directory.
The working directory of the :attr:`prebuild_cmd` and :attr:`postbuild_cmd` commands will be the stage directory of the test.


An alternative way to retrieve specifically a Git repository is to assign its URL directly to the :attr:`sourcesdir` attribute:

.. code-block:: python

   self.sourcesdir = 'https://github.com/me/myrepo'

ReFrame will attempt to clone this repository inside the stage directory by executing ``git clone <repo> .`` and will then procede with the compilation as described above.


.. note::
   ReFrame recognizes only URLs in the :attr:`sourcesdir` attribute and requires passwordless access to the repository.
   This means that the SCP-style repository specification will not be accepted.
   You will have to specify it as URL using the ``ssh://`` protocol (see `Git documentation page <https://git-scm.com/docs/git-clone#_git_urls_a_id_urls_a>`__).


Add a configuration step before compiling the code
==================================================

It is often the case that a configuration step is needed before compiling a code with ``make``.
To address this kind of projects, ReFrame aims to offer specific abstractions for "configure-make"-style build systems.
It supports `CMake-based <https://cmake.org/>`__ projects through the :class:`CMake <reframe.core.buildsystems.CMake>` build system, as well as `Autotools-based <https://www.gnu.org/software/automake/>`__ projects through the :class:`Autotools <reframe.core.buildsystems.Autotools>` build system.

For other build systems, you can achieve the same effect using the :class:`Make <reframe.core.buildsystems.Make>` build system and the :attr:`prebuild_cmd <reframe.core.pipeline.RegressionTest.prebuild_cmd>` for performing the configuration step.
The following code snippet will configure a code with ``./custom_configure`` before invoking ``make``:

.. code-block:: python

  self.prebuild_cmd = ['./custom_configure -with-mylib']
  self.build_system = 'Make'
  self.build_system.cppflags = ['-DHAVE_FOO']
  self.build_system.flags_from_environ = False

The generated build script then will have the following lines:

.. code-block:: bash

  ./custom_configure -with-mylib
  make -j CPPFLAGS='-DHAVE_FOO'


Implementing a Run-Only Regression Test
---------------------------------------

There are cases when it is desirable to perform regression testing for an already built executable.
The following test uses the ``echo`` Bash shell command to print a random integer between specific lower and upper bounds.
Here is the full regression test (``tutorial/advanced/advanced_example2.py``):

.. literalinclude:: ../tutorial/advanced/advanced_example2.py

There is nothing special for this test compared to those presented `earlier <tutorial.html>`__ except that it derives from the :class:`RunOnlyRegressionTest <reframe.core.pipeline.RunOnlyRegressionTest>` and that it does not contain any resources (``self.sourcesdir = None``).
Note that run-only regression tests may also have resources, as for instance a precompiled executable or some input data. The copying of these resources to the stage directory is performed at the beginning of the run phase.
For standard regression tests, this happens at the beginning of the compilation phase, instead.
Furthermore, in this particular test the :attr:`executable <reframe.core.pipeline.RegressionTest.executable>` consists only of standard Bash shell commands.
For this reason, we can set :attr:`sourcesdir <reframe.core.pipeline.RegressionTest.sourcesdir>` to ``None`` informing ReFrame that the test does not have any resources.

Implementing a Compile-Only Regression Test
-------------------------------------------

ReFrame provides the option to write compile-only tests which consist only of a compilation phase without a specified executable.
This kind of tests must derive from the :class:`CompileOnlyRegressionTest <reframe.core.pipeline.CompileOnlyRegressionTest>` class provided by the framework.
The following example (``tutorial/advanced/advanced_example3.py``) reuses the code of our first example in this section and checks that no warnings are issued by the compiler:

.. literalinclude:: ../tutorial/advanced/advanced_example3.py

The important thing to note here is that the standard output and standard error of the tests, accessible through the :attr:`stdout <reframe.core.pipeline.RegressionTest.stdout>` and :attr:`stderr <reframe.core.pipeline.RegressionTest.stderr>` attributes, are now the corresponding those of the compilation command.
So sanity checking can be done in exactly the same way as with a normal test.

Leveraging Environment Variables
--------------------------------

We have already demonstrated in the `tutorial <tutorial.html>`__ that ReFrame allows you to load the required modules for regression tests and also set any needed environment variables. When setting environment variables for your test through the :attr:`variables <reframe.core.pipeline.RegressionTest.variables>` attribute, you can assign them values of other, already defined, environment variables using the standard notation ``$OTHER_VARIABLE`` or ``${OTHER_VARIABLE}``.
The following regression test (``tutorial/advanced/advanced_example4.py``) sets the ``CUDA_HOME`` environment variable to the value of the ``CUDATOOLKIT_HOME`` and then compiles and runs a simple program:

.. literalinclude:: ../tutorial/advanced/advanced_example4.py

Before discussing this test in more detail, let's first have a look in the source code and the Makefile of this example:

.. literalinclude:: ../tutorial/advanced/src/advanced_example4.c
  :language: c

This program is pretty basic, but enough to demonstrate the use of environment variables from ReFrame.
It simply compares the value of the ``CUDA_HOME`` macro with the value of the environment variable ``CUDA_HOME`` at runtime, printing ``SUCCESS`` if they are not empty and match.
The Makefile for this example compiles this source by simply setting ``CUDA_HOME`` to the value of the ``CUDA_HOME`` environment variable:

.. literalinclude:: ../tutorial/advanced/src/Makefile_example4
  :language: makefile

Coming back now to the ReFrame regression test, the ``CUDATOOLKIT_HOME`` environment variable is defined by the ``cudatoolkit`` module.
If you try to run the test, you will see that it will succeed, meaning that the ``CUDA_HOME`` variable was set correctly both during the compilation and the runtime.

When ReFrame `sets up <pipeline.html#the-setup-phase>`__ a test, it first loads its required modules and then sets the required environment variables expanding their values.
This has the result that ``CUDA_HOME`` takes the correct value in our example at the compilation time.

At runtime, ReFrame will generate the following instructions in the shell script associated with this test:

.. code-block:: bash

  module load cudatoolkit
  export CUDA_HOME=$CUDATOOLKIT_HOME

This ensures that the environment of the test is also set correctly at runtime.

Finally, as already mentioned `previously <#working-with-makefiles>`__, since the name of the makefile is not one of the standard ones, it must be set explicitly in the build system:

.. literalinclude:: ../tutorial/advanced/advanced_example4.py
  :lines: 17
  :dedent: 8

Setting a Time Limit for Regression Tests
-----------------------------------------

ReFrame gives you the option to limit the execution time of regression tests.
The following example (``tutorial/advanced/advanced_example5.py``) demonstrates how you can achieve this by limiting the execution time of a test that tries to sleep 100 seconds:

.. literalinclude:: ../tutorial/advanced/advanced_example5.py

The important bit here is the following line that sets the time limit for the test to one minute:

.. literalinclude:: ../tutorial/advanced/advanced_example5.py
  :lines: 13
  :dedent: 8

The :attr:`time_limit <reframe.core.pipeline.RegressionTest.time_limit>` attribute is a three-tuple in the form ``(HOURS, MINUTES, SECONDS)``.
Time limits are implemented for all the scheduler backends.

The sanity condition for this test verifies that associated job has been canceled due to the time limit (note that this message is SLURM-specific).

.. literalinclude:: ../tutorial/advanced/advanced_example5.py
  :lines: 16-17
  :dedent: 8

Applying a sanity function iteratively
--------------------------------------

It is often the case that a common sanity pattern has to be applied many times.
In this example we will demonstrate how the above situation can be easily tackled using the :mod:`sanity <reframe.utility.sanity>` functions offered by ReFrame.
Specifically, we would like to execute the following shell script and check that its output is correct:

.. literalinclude:: ../tutorial/advanced/src/random_numbers.sh
  :language: bash

The above script simply prints 100 random integers between the limits given by the variables ``LOWER`` and ``UPPER``.
In the corresponding regression test we want to check that all the random numbers printed lie between 90 and 100 ensuring that the script executed correctly.
Hence, a common sanity check has to be applied to all the printed random numbers.
In ReFrame this can achieved by the use of :func:`map <reframe.utility.sanity.map>` sanity function accepting a function and an iterable as arguments.
Through :func:`map <reframe.utility.sanity.map>` the given function will be applied to all the members of the iterable object.
Note that since :func:`map <reframe.utility.sanity.map>` is a sanity function, its execution will be deferred.
The contents of the ReFrame regression test contained in ``advanced_example6.py`` are the following:

.. literalinclude:: ../tutorial/advanced/advanced_example6.py

First the random numbers are extracted through the :func:`extractall <reframe.utility.sanity.extractall>` function as follows:

.. literalinclude:: ../tutorial/advanced/advanced_example6.py
  :lines: 14-15
  :dedent: 8

The ``numbers`` variable is a deferred iterable, which upon evaluation will return all the extracted numbers.
In order to check that the extracted numbers lie within the specified limits, we make use of the :func:`map <reframe.utility.sanity.map>` sanity function, which will apply the :func:`assert_bounded <reframe.utility.sanity.assert_bounded>` to all the elements of ``numbers``.
Additionally, our requirement is that all the numbers satisfy the above constraint and we therefore use :func:`all <reframe.utility.sanity.all>`.

There is still a small complication that needs to be addressed.
The :func:`all <reframe.utility.sanity.all>` function returns :class:`True` for empty iterables, which is not what we want.
So we must ensure that all the numbers are extracted as well.
To achieve this, we make use of :func:`count <reframe.utility.sanity.count>` to get the number of elements contained in ``numbers`` combined with :func:`assert_eq <reframe.utility.sanity.assert_eq>` to check that the number is indeed 100.
Finally, both of the above conditions have to be satisfied for the program execution to be considered successful, hence the use of the :func:`and_ <reframe.utility.sanity.and_>` function.
Note that the ``and`` operator is not deferrable and will trigger the evaluation of any deferrable argument passed to it.

The full syntax for the :attr:`sanity_patterns` is the following:

.. literalinclude:: ../tutorial/advanced/advanced_example6.py
  :lines: 16-18
  :dedent: 8

Customizing the Generated Job Script
------------------------------------

It is often the case that you must run some commands before and/or after the parallel launch of your executable.
This can be easily achieved by using the :attr:`pre_run <reframe.core.pipeline.RegressionTest.pre_run>` and :attr:`post_run <reframe.core.pipeline.RegressionTest.post_run>` attributes of :class:`RegressionTest`.

The following example is a slightly modified version of the previous one.
The lower and upper limits for the random numbers are now set inside a helper shell script in ``scripts/limits.sh`` and we want also to print the word ``FINISHED`` after our executable has finished.
In order to achieve this, we need to source the helper script just before launching the executable and ``echo`` the desired message just after it finishes.
Here is the test file:

.. literalinclude:: ../tutorial/advanced/advanced_example7.py

Notice the use of the :attr:`pre_run` and :attr:`post_run` attributes.
These are list of shell commands that are emitted verbatim in the job script.
The generated job script for this example is the following:

.. code-block:: bash

   #!/bin/bash -l
   #SBATCH --job-name="prerun_demo_check_daint_gpu_PrgEnv-gnu"
   #SBATCH --time=0:10:0
   #SBATCH --ntasks=1
   #SBATCH --output=/path/to/stage/gpu/prerun_demo_check/PrgEnv-gnu/prerun_demo_check.out
   #SBATCH --error=/path/to/stage/gpu/prerun_demo_check/PrgEnv-gnu/prerun_demo_check.err
   #SBATCH --constraint=gpu
   module load daint-gpu
   module unload PrgEnv-cray
   module load PrgEnv-gnu
   source scripts/limits.sh
   srun ./random_numbers.sh
   echo FINISHED

ReFrame generates the job shell script using the following pattern:

.. code-block:: bash

   #!/bin/bash -l
   {job_scheduler_preamble}
   {test_environment}
   {pre_run}
   {parallel_launcher} {executable} {executable_opts}
   {post_run}

The ``job_scheduler_preamble`` contains the directives that control the job allocation.
The ``test_environment`` are the necessary commands for setting up the environment of the test.
This is the place where the modules and environment variables specified in :attr:`modules <reframe.core.pipeline.RegressionTest.modules>` and :attr:`variables <reframe.core.pipeline.RegressionTest.variables>` attributes are emitted.
Then the commands specified in :attr:`pre_run` follow, while those specified in the :attr:`post_run` come after the launch of the parallel job.
The parallel launch itself consists of three parts:

#. The parallel launcher program (e.g., ``srun``, ``mpirun`` etc.) with its options,
#. the regression test executable as specified in the :attr:`executable <reframe.core.pipeline.RegressionTest.executable>` attribute and
#. the options to be passed to the executable as specified in the :attr:`executable_opts <reframe.core.pipeline.RegressionTest.executable_opts>` attribute.

A key thing to note about the generated job script is that ReFrame submits it from the stage directory of the test, so that all relative paths are resolved against it.


Working with parameterized tests
--------------------------------

.. versionadded:: 2.13

We have seen already in the `basic tutorial <tutorial.html#combining-it-all-together>`__ how we could better organize the tests so as to avoid code duplication by using test class hierarchies.
An alternative technique, which could also be used in parallel with the class hierarchies, is to use `parameterized tests`.
The following is a test that takes a ``variant`` parameter, which controls which variant of the code will be used.
Depending on that value, the test is set up differently:

.. literalinclude:: ../tutorial/advanced/advanced_example8.py

If you have already gone through the `tutorial <tutorial.html>`__, this test can be easily understood.
The new bit here is the ``@parameterized_test`` decorator of the ``MatrixVectorTest`` class.
This decorator takes an arbitrary number of arguments, which are either of a sequence type (i.e., list, tuple etc.) or of a mapping type (i.e., dictionary).
Each of the decorator's arguments corresponds to the constructor arguments of the decorated test that will be used to instantiate it.
In the example shown, the test will be instantiated twice, once passing ``variant`` as ``MPI`` and a second time with ``variant`` passed as ``OpenMP``.
The framework will try to generate unique names for the generated tests by stringifying the arguments passed to the test's constructor:

.. code-block:: none

   Command line: ./bin/reframe -C tutorial/config/settings.py -c tutorial/advanced/advanced_example8.py -l
   Reframe version: 2.15-dev1
   Launched by user: XXX
   Launched on host: daint101
   Reframe paths
   =============
       Check prefix      :
       Check search path : 'tutorial/advanced/advanced_example8.py'
       Stage dir prefix  : current/working/dir/reframe/stage/
       Output dir prefix : current/working/dir/reframe/output/
       Logging dir       : current/working/dir/reframe/logs
   List of matched checks
   ======================
     * MatrixVectorTest_MPI (Matrix-vector multiplication test (MPI))
     * MatrixVectorTest_OpenMP (Matrix-vector multiplication test (OpenMP))
   Found 2 check(s).


There are a couple of different ways that we could have used the ``@parameterized_test`` decorator.
One is to use dictionaries for specifying the instantiations of our test class.
The dictionaries will be converted to keyword arguments and passed to the constructor of the test class:

.. code-block:: python

   @rfm.parameterized_test({'variant': 'MPI'}, {'variant': 'OpenMP'})


Another way, which is quite useful if you want to generate lots of different tests at the same time, is to use either `list comprehensions <https://docs.python.org/3.6/tutorial/datastructures.html#list-comprehensions>`__ or `generator expressions <https://www.python.org/dev/peps/pep-0289/>`__ for specifying the different test instantiations:

.. code-block:: python

   @rfm.parameterized_test(*([variant] for variant in ['MPI', 'OpenMP']))


.. note::
   In versions of the framework prior to 2.13, this could be achieved by explicitly instantiating your tests inside the ``_get_checks()`` method.


.. tip::

   Combining parameterized tests and test class hierarchies can offer you a very flexible way for generating multiple related tests at once keeping at the same time the maintenance cost low.
   We use this technique extensively in our tests.


Flexible Regression Tests
-------------------------

.. versionadded:: 2.15

ReFrame can automatically set the number of tasks of a particular test, if its :attr:`num_tasks <reframe.core.pipeline.RegressionTest.num_tasks>` attribute is set to ``0``.
In ReFrame's terminology, such tests are called `flexible`.
By default, ReFrame will spawn such a test on all the idle nodes of the current system partition, but this behavior can be adjusted from the command-line.
Flexible tests are very useful for diagnostics tests, e.g., tests for checking the health of a whole set nodes.
In this example, we demonstrate this feature through a simple test that runs ``hostname``.
The test will verify that all the nodes print the expected host name:

.. literalinclude:: ../tutorial/advanced/advanced_example9.py

The first thing to notice in this test is that :attr:`num_tasks <reframe.core.pipeline.RegressionTest.num_tasks>` is set to ``0``.
This is a requirement for flexible tests:

.. literalinclude:: ../tutorial/advanced/advanced_example9.py
  :lines: 13
  :dedent: 8

The sanity function of this test simply counts the host names and verifies that they are as many as expected:

.. literalinclude:: ../tutorial/advanced/advanced_example9.py
  :lines: 15-18
  :dedent: 8

Notice, however, that the sanity check does not use :attr:`num_tasks` for verification, but rather a different, custom attribute, the ``num_tasks_assigned``.
This happens for two reasons:

  a. At the time the sanity check expression is created, :attr:`num_tasks` is ``0``.
     So the actual number of tasks assigned must be a deferred expression as well.
  b. When ReFrame will determine and set the number of tasks of the test, it will not set the :attr:`num_tasks` attribute of the :class:`RegressionTest`.
     It will only set the corresponding attribute of the associated job instance.

Here is how the new deferred attribute is defined:

.. literalinclude:: ../tutorial/advanced/advanced_example9.py
  :lines: 22-25
  :dedent: 4


The behavior of the flexible task allocation is controlled by the ``--flex-alloc-tasks`` command line option.
See the corresponding `section <running.html#controlling-the-flexible-task-allocation>`__ for more information.
