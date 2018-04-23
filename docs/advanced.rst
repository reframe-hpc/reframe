=====================================
Customizing Further a Regression Test
=====================================

In this section, we are going to show some more elaborate use cases of ReFrame.
Through the use of more advanced examples, we will demonstrate further customization options which modify the default options of the ReFrame pipeline.
The corresponding scripts as well as the source code of the examples discussed here can be found in the directory ``tutorial/advanced``.

Leveraging Makefiles
--------------------

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

The important bit here is the ``compile()`` method.

.. literalinclude:: ../tutorial/advanced/advanced_example1.py
  :lines: 21-23
  :dedent: 4

As in the simple single source file examples we showed in the `tutorial <tutorial.html>`__, we use the current programming environment's flags for modifying the compilation.
ReFrame will then compile the regression test source code as by invoking ``make`` as follows:

.. code-block:: bash

  make CC=cc CXX=CC FC=ftn CPPFLAGS=-DMESSAGE

Notice, how ReFrame passes all the programming environment's variables to the ``make`` invocation.
It is important to note here that, if a set of flags is set to :class:`None` (the default, if not otherwise set in the `ReFrame's configuration <configure.html#environments-configuration>`__), these are not passed to ``make``.
You can also completely disable the propagation of any flags to ``make`` by setting :attr:`self.propagate = False <reframe.core.environments.ProgEnvironment.propagate>` in your regression test.

At this point it is useful also to note that you can also use a custom Makefile, not named ``Makefile`` or after any other standard Makefile name.
In this case, you can pass the custom Makefile name as an argument to the compile method of the base :class:`RegressionTest <reframe.core.pipeline.RegressionTest>` class as follows:

.. code-block:: python

  super().compile(makefile='Makefile_custom')


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

Finally, as already mentioned `previously <#leveraging-makefiles>`__, since the ``Makefile`` name is not one of the standard ones, it has to be passed as an argument to the :func:`compile <reframe.core.pipeline.RegressionTest.compile>` method of the base :class:`RegressionTest <reframe.core.pipeline.RegressionTest>` class as follows:

.. literalinclude:: ../tutorial/advanced/advanced_example4.py
  :lines: 24
  :dedent: 8

Setting a Time Limit for Regression Tests
-----------------------------------------

ReFrame gives you the option to limit the execution time of regression tests.
The following example (``tutorial/advanced/advanced_example5.py``) demonstrates how you can achieve this by limiting the execution time of a test that tries to sleep 100 seconds:

.. literalinclude:: ../tutorial/advanced/advanced_example5.py

The important bit here is the following line that sets the time limit for the test to one minute:


.. literalinclude:: ../tutorial/advanced/advanced_example5.py
  :lines: 16
  :dedent: 8

The :attr:`time_limit <reframe.core.pipeline.RegressionTest.time_limit>` attribute is a three-tuple in the form ``(HOURS, MINUTES, SECONDS)``.
Time limits are implemented for all the scheduler backends.

The sanity condition for this test verifies that associated job has been canceled due to the time limit.

.. literalinclude:: ../tutorial/advanced/advanced_example5.py
  :lines: 19-20
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
  :lines: 16-17
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
  :lines: 18-20
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

A key thing to note about the generated job script is that ReFrame submits it from the stage directory of the test, so that all relative paths are resolved against inside it.
