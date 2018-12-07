================
ReFrame Tutorial
================

This tutorial will guide you through writing your first regression tests with ReFrame.
We will start with the most common and simple case of a regression test that compiles a code, runs it and checks its output.
We will then expand this example gradually by adding functionality and more advanced sanity and performance checks.
By the end of the tutorial, you should be able to start writing your first regression tests with ReFrame.

If you just want to get a quick feeling of how it is like writing a regression test in ReFrame, you can start directly from here.
However, if you want to get a better understanding of what is happening behind the scenes, we recommend to have a look also in `"The Regression Test Pipeline" <pipeline.html>`__ section.

All the tutorial examples can be found in ``<reframe-install-prefix>/tutorial/``.

For the configuration of the system, we provide a minimal configuration file for Piz Daint, where we have tested all the tutorial examples.
The site configuration that we used for this tutorial is the following:

.. literalinclude:: ../tutorial/config/settings.py
  :lines: 12-75
  :dedent: 4

You can find the full ``settings.py`` file ready to be used by ReFrame in ``<reframe-install-prefix>/tutorial/config/settings.py``.
You may first need to go over the `"Configuring ReFrame For Your Site" <configure.html>`__ section, in order to prepare the framework for your systems.

The First Regression Test
-------------------------

The following is a simple regression test that compiles and runs a serial C program, which computes a matrix-vector product (``tutorial/src/example_matrix_multiplication.c``), and verifies its sane execution.
As a sanity check, it simply looks for a specific output in the output of the program.
Here is the full code for this test:

.. literalinclude:: ../tutorial/example1.py

A regression test written in ReFrame is essentially a Python class that must eventually derive from :class:`RegressionTest <reframe.core.pipeline.RegressionTest>`.
To make a test visible to the framework, you must decorate your final test class with one of the following decorators:

* ``@simple_test``: for registering a single parameterless instantiation of your test.
* ``@parameterized_test``: for registering multiple instantiations of your test.

Let's see in more detail how the ``Example1Test`` is defined:

.. literalinclude:: ../tutorial/example1.py
  :lines: 4-7

The ``__init__()`` method is the constructor of your test.
It is usually the only method you need to implement for your tests, especially if you don't want to customize any of the regression test pipeline stages.
The first statement in the ``Example1Test`` constructor calls the constructor of the base class.
This is essential for properly initializing your test.
When your test is instantiated, the framework assigns a default name to it.
This name is essentially a concatenation of the fully qualified name of the class and string representations of the constructor arguments, with any non-alphanumeric characters converted to underscores.
In this example, the auto-generated test name is simply ``Example1Test``.
You may change the name of the test later in the constructor by setting the :attr:`name <reframe.core.pipeline.RegressionTest.name>` attribute.

.. note::
      ReFrame requires that the names of all the tests it loads are unique.
      In case of name clashes, it will refuse to load the conflicting test.

   .. versionadded:: 2.12

The next line sets a more detailed description of the test:

.. literalinclude:: ../tutorial/example1.py
  :lines: 8
  :dedent: 8

This is optional and it defaults to the auto-generated test's name, if not specified.

.. note::
   If you explicitly set only the name of the test, the description will not be automatically updated and will still keep its default value.

The next two lines specify the systems and the programming environments that this test is valid for:

.. literalinclude:: ../tutorial/example1.py
  :lines: 9-10
  :dedent: 8

Both of these variables accept a list of system names or environment names, respectively.
The ``*`` symbol is a wildcard meaning any system or any programming environment.
The system and environment names listed in these variables must correspond to names of systems and environments defined in the ReFrame's `settings file <configure.html#the-configuration-file>`__.

.. ..note:: If a name specified in these lists does not appear in the settings file, it will be simply ignored.

When specifying system names you can always specify a partition name as well by appending ``:<partname>`` to the system's name.
For example, given the configuration for our tutorial, ``daint:gpu`` would refer specifically to the ``gpu`` virtual partition of the system ``daint``.
If only a system name (without a partition) is specified in the :attr:`self.valid_systems <reframe.core.pipeline.RegressionTest.valid_systems>` variable, e.g., ``daint``, it means that this test is valid for any partition of this system.

The next line specifies the source file that needs to be compiled:

.. literalinclude:: ../tutorial/example1.py
  :lines: 11
  :dedent: 8

ReFrame expects any source files, or generally resources, of the test to be inside an ``src/`` directory, which is at the same level as the regression test file.
If you inspect the directory structure of the ``tutorial/`` folder, you will notice that:

.. code-block:: none

  tutorial/
      example1.py
      src/
          example_matrix_vector_multiplication.c

Notice also that you need not specify the programming language of the file you are asking ReFrame to compile or the compiler to use.
ReFrame will automatically pick the correct compiler based on the extension of the source file.
The exact compiler that is going to be used depends on the programming environment that the test is running with.
For example, given our configuration, if it is run with ``PrgEnv-cray``, the Cray C compiler will be used, if it is run with ``PrgEnv-gnu``, the GCC compiler will be used etc.
A user can associate compilers with programming environments in the ReFrame's `settings file <configure.html#the-configuration-file>`__.

The next line in our first regression test specifies a list of options to be used for running the generated executable (the matrix dimension and the number of iterations in this particular example):

.. literalinclude:: ../tutorial/example1.py
  :lines: 12
  :dedent: 8

Notice that you do not need to specify the executable name.
Since ReFrame compiled it and generated it, it knows the name.
We will see in the `"Customizing Further A ReFrame Regression Test" <advanced.html>`__ section, how you can specify the name of the executable, in cases that ReFrame cannot guess its name.

The next lines specify what should be checked for assessing the sanity of the result of the test:

.. literalinclude:: ../tutorial/example1.py
  :lines: 13-14
  :dedent: 8

This expression simply asks ReFrame to look for ``time for single matrix vector multiplication`` in the standard output of the test.
The :attr:`sanity_patterns <reframe.core.pipeline.RegressionTest.sanity_patterns>` attribute can only be assigned the result of a special type of functions, called *sanity functions*.
`Sanity functions <deferrables.html>`__ are special in the sense that they are evaluated lazily.
You can generally treat them as normal Python functions inside a :attr:`sanity_patterns <reframe.core.pipeline.RegressionTest.sanity_patterns>` expression.
ReFrame provides already a wide range of useful sanity functions ranging from wrappers to the standard built-in functions of Python to functions related to parsing the output of a regression test.
For a complete listing of the available functions, please have a look at the `"Sanity Functions Reference" <sanity_functions_reference.html>`__.

In our example, the :func:`assert_found <reframe.utility.sanity.assert_found>` function accepts a regular expression pattern to be searched in a file and either returns :class:`True` on success or raises a :class:`SanityError <reframe.core.exceptions.SanityError>` in case of failure with a descriptive message.
This function uses internally the "`re <https://docs.python.org/3.6/library/re.html>`__" module of the Python standard library, so it may accept the same `regular expression syntax <https://docs.python.org/3.6/library/re.html#regular-expression-syntax>`__.
As a file argument, :func:`assert_found <reframe.utility.sanity.assert_found>` accepts any filename, which will be resolved against the stage directory of the test.
You can also use the :attr:`stdout <reframe.core.pipeline.RegressionTest.stdout>` and :attr:`stderr <reframe.core.pipeline.RegressionTest.stderr>` attributes to reference the standard output and standard error, respectively.

.. tip:: You need not to care about handling exceptions, and error handling in general, inside your test.
  The framework will automatically abort the execution of the test, report the error and continue with the next test case.

The last two lines of the regression test are optional, but serve a good role in a production environment:

.. literalinclude:: ../tutorial/example1.py
  :lines: 15-16
  :dedent: 8

In the :attr:`maintainers <reframe.core.pipeline.RegressionTest.maintainers>` attribute you may store a list of people responsible for the maintenance of this test.
In case of failure, this list will be printed in the failure summary.

The :attr:`tags <reframe.core.pipeline.RegressionTest.tags>` attribute is a set of tags that you can assign to this test.
This is useful for categorizing the tests and helps in quickly selecting the tests of interest.
More about test selection, you can find in the `"Running ReFrame" <running.html>`__ section.

.. note:: The values assigned to the attributes of a :class:`RegressionTest <reframe.core.pipeline.RegressionTest>` are validated and if they don't have the correct type, an error will be issued by ReFrame.
  For a list of all the attributes and their types, please refer to the `"Reference Guide" <reference.html>`__.

Running the Tutorial Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ReFrame offers a rich command-line interface that allows you to control several aspects of its executions.
A more detailed description can be found in the `"Running ReFrame" <running.html>`__ section.
Here we will only show you how to run a specific tutorial test:

.. code-block:: bash

  ./bin/reframe -C tutorial/config/settings.py -c tutorial/example1.py -r

If everything is configured correctly for your system, you should get an output similar to the following:

.. code-block:: none

   Command line: ./bin/reframe -C tutorial/config/settings.py -c tutorial/example1.py -r
   Reframe version: 2.13-dev0
   Launched by user: XXX
   Launched on host: daint104
   Reframe paths
   =============
       Check prefix      :
       Check search path : 'tutorial/example1.py'
       Stage dir prefix  : /current/working/dir/stage/
       Output dir prefix : /current/working/dir/output/
       Logging dir       : /current/working/dir/logs
   [==========] Running 1 check(s)
   [==========] Started on Fri May 18 13:19:12 2018

   [----------] started processing Example1Test (Simple matrix-vector multiplication example)
   [ RUN      ] Example1Test on daint:login using PrgEnv-cray
   [       OK ] Example1Test on daint:login using PrgEnv-cray
   [ RUN      ] Example1Test on daint:login using PrgEnv-gnu
   [       OK ] Example1Test on daint:login using PrgEnv-gnu
   [ RUN      ] Example1Test on daint:login using PrgEnv-intel
   [       OK ] Example1Test on daint:login using PrgEnv-intel
   [ RUN      ] Example1Test on daint:login using PrgEnv-pgi
   [       OK ] Example1Test on daint:login using PrgEnv-pgi
   [ RUN      ] Example1Test on daint:gpu using PrgEnv-cray
   [       OK ] Example1Test on daint:gpu using PrgEnv-cray
   [ RUN      ] Example1Test on daint:gpu using PrgEnv-gnu
   [       OK ] Example1Test on daint:gpu using PrgEnv-gnu
   [ RUN      ] Example1Test on daint:gpu using PrgEnv-intel
   [       OK ] Example1Test on daint:gpu using PrgEnv-intel
   [ RUN      ] Example1Test on daint:gpu using PrgEnv-pgi
   [       OK ] Example1Test on daint:gpu using PrgEnv-pgi
   [ RUN      ] Example1Test on daint:mc using PrgEnv-cray
   [       OK ] Example1Test on daint:mc using PrgEnv-cray
   [ RUN      ] Example1Test on daint:mc using PrgEnv-gnu
   [       OK ] Example1Test on daint:mc using PrgEnv-gnu
   [ RUN      ] Example1Test on daint:mc using PrgEnv-intel
   [       OK ] Example1Test on daint:mc using PrgEnv-intel
   [ RUN      ] Example1Test on daint:mc using PrgEnv-pgi
   [       OK ] Example1Test on daint:mc using PrgEnv-pgi
   [----------] finished processing Example1Test (Simple matrix-vector multiplication example)

   [  PASSED  ] Ran 12 test case(s) from 1 check(s) (0 failure(s))
   [==========] Finished on Fri May 18 13:20:17 2018


Notice how our regression test is run on every partition of the configured system and for every programming environment.

Now that you have got a first understanding of how a regression test is written in ReFrame, let's try to expand our example.

Inspecting the ReFrame Generated Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As described in the `regression test pipeline <pipeline>`__ section, ReFrame generates several files during the execution of a test.
When developing or debugging a regression test it is important to be able to locate them and inspect them.

As soon as the `setup` stage of the test is executed, a stage directory specific to this test is generated.
All the required resources for the test are copied to this directory, and this will be the working directory for the compilation, running, sanity and performance checking phases.
If the test is successful, this stage directory is removed, unless the ``--keep-stage-files`` option is passed in the command line.
Before removing this directory, ReFrame copies the following files to a dedicated output directory for this test:

- The generated build script and its standard output and standard error.
  This allows you to inspect exactly how your test was compiled.
- The generated run script and its standard output and standard error.
  This allows you to inspect exactly how your test was run and verify that the sanity checking was correct.
- Any other user-specified files.

If a regression test fails, its stage directory will not be removed.
This allows you to reproduce exactly what ReFrame was trying to perform and will help you debug the problem with your test.

Let's rerun our first example and instruct ReFrame to keep the stage directory of the test, so that we can inspect it.

.. code::

  ./bin/reframe -C tutorial/config/settings.py -c tutorial/example1.py -r --keep-stage-files

ReFrame creates a stage directory for each test case using the following pattern:

.. code::

  $STAGEDIR_PREFIX/<system>/<partition>/<prog-environ>/<test-name>

Let's pick the test case for the ``gpu`` partition and the ``PrgEnv-gnu`` programming environment from our first test to inspect.
The default ``STAGEDIR_PREFIX`` is ``./stage``:

.. code::

   cd stage/daint/gpu/PrgEnv-gnu/Example1Test/

If you do a listing in this directory, you will see all the files contained in the ``tutorial/src`` directory, as well as the following files:

.. code::

  rfm_Example1Test_build.err  rfm_Example1Test_job.err
  rfm_Example1Test_build.out  rfm_Example1Test_job.out
  rfm_Example1Test_build.sh   rfm_Example1Test_job.sh


The ``rfm_Example1Test_build.sh`` is the generated build script and the ``.out`` and ``.err`` are the compilation's standard output and standard error.
Here is the generated build script for our first test:

.. code:: shell

  #!/bin/bash

  _onerror()
  {
      exitcode=$?
      echo "-reframe: command \`$BASH_COMMAND' failed (exit code: $exitcode)"
      exit $exitcode
  }

  trap _onerror ERR

  module load daint-gpu
  module unload PrgEnv-cray
  module load PrgEnv-gnu
  cc example_matrix_vector_multiplication.c -o ./Example1Test


Similarly, the ``rfm_Example1Test_job.sh`` is the generated job script and the ``.out`` and ``.err`` files are the corresponding standard output and standard error.
The generated job script for the test case we are currently inspecting is the following:

.. code:: shell

  #!/bin/bash -l
  #SBATCH --job-name="rfm_Example1Test_job"
  #SBATCH --time=0:10:0
  #SBATCH --ntasks=1
  #SBATCH --output=/path/to/stage/daint/gpu/PrgEnv-gnu/Example1Test/rfm_Example1Test_job.out
  #SBATCH --error=/path/to/stage/daint/gpu/PrgEnv-gnu/Example1Test/rfm_Example1Test_job.err
  #SBATCH --constraint=gpu
  module load daint-gpu
  module unload PrgEnv-cray
  module load PrgEnv-gnu
  srun ./Example1Test 1024 100


It is interesting to check here the generated job script for the ``login`` partition of the example system, which does not use a workload manager:

.. code::

  cat stage/daint/login/PrgEnv-gnu/Example1Test/rfm_Example1Test_job.sh

.. code:: shell

  #!/bin/bash -l
  module unload PrgEnv-cray
  module load PrgEnv-gnu
   ./Example1Test 1024 100


This is one of the advantages in using ReFrame:
You do not have to care about the system-level details of the target system that your test is running.
Based on its configuration, ReFrame will generate the appropriate commands to run your test.


Customizing the Compilation Phase
---------------------------------

In this example, we write a regression test to compile and run the OpenMP version of the matrix-vector product program, that we have shown before.
The full code of this test follows:

.. literalinclude:: ../tutorial/example2.py
  :lines: 1-34

This example introduces two new concepts:

1. We need to set the ``OMP_NUM_THREADS`` environment variable, in order to specify the number of threads to use with our program.
2. We need to specify different flags for the different compilers provided by the programming environments we are testing.
   Notice also that we now restrict the validity of our test only to the programming environments that we know how to handle (see the :attr:`valid_prog_environs <reframe.core.pipeline.RegressionTest.valid_prog_environs>`).

To define environment variables to be set during the execution of a test, you should use the :attr:`variables <reframe.core.pipeline.RegressionTest.variables>` attribute of the :class:`RegressionTest <reframe.core.pipeline.RegressionTest>` class.
This is a dictionary, whose keys are the names of the environment variables and whose values are the values of the environment variables.
Notice that both the keys and the values must be strings.

From version 2.14, ReFrame manages compilation of tests through the concept of build systems.
Any customization of the build process should go through a build system.
For straightforward cases, as in our first example, where no customization is needed, ReFrame automatically picks the correct build system to build the code.
In this example, however, we want to set the flags for compiling the OpenMP code.
Assuming our test supported only GCC, we could simply add the following lines in the ``__init__()`` method of our test:

.. code:: python

        self.build_system = 'SingleSource'
        self.build_system.cflags = ['-fopenmp']

The :class:`SingleSource <reframe.core.buildsystems.SingleSource>` build system that we use here supports the compilation of a single file only.
Each build system type defines a set of variables that the user can set.
Based on the selected build system, ReFrame will generate a build script that will be used for building the code.
The generated build script can be found in `the stage or the output directory of the test <running.html#configuring-reframe-directories>`__, along with the output of the compilation.
This way, you may reproduce exactly what ReFrame does in case of any errors.
More on the build systems feature can be found `here <reference.html#build-systems>`__.

Getting back to our test, simply setting the ``cflags`` to ``-fopenmp`` globally in the test will make it fail for programming environments other than ``PrgEnv-gnu``, since the OpenMP flags vary for the different compilers.
Ideally, we need to set the ``cflags`` differently for each programming environment.
To achieve this we need to override the :func:`setup <reframe.core.pipeline.RegressionTest.setup>` method of the :class:`RegressionTest <reframe.core.pipeline.RegressionTest>`.
As described in `"The Regression Test Pipeline" <pipeline.html>`__ section, it is during the setup phase that a regression test is prepared for a new system partition and a new programming environment.
The following lines show the overriden ``setup()`` method:

.. literalinclude:: ../tutorial/example2.py
  :lines: 24-34
  :dedent: 4

The current environment is passed as argument by the framework to the ``setup()`` method, so we differentiate the build system's flags based on its name.
Finally, we need call the ``setup()`` method of the base class, in order to perform the actual setup of the test.

.. tip::

  The :class:`RegressionTest <reframe.core.pipeline.RegressionTest>` implements the six phases of the regression test pipeline in separate methods.
  Individual regression tests may override them to provide alternative implementations, but in most practical cases, only the :func:`setup <reframe.core.pipeline.RegressionTest.setup>` may need to be overriden.
  You will hardly ever need to override any of the other methods and, in fact, you should be very careful when doing it.


.. warning::
  Setting the compiler flags in the programming environment is now deprecated.
  Users are advised to use the build systems feature instead.

   .. versionadded:: 2.14


An alternative implementation using dictionaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we present an alternative implementation of the same test using a dictionary to hold the compilation flags for the different programming environments.
The advantage of this implementation is that you move the different compilation flags in the initialization phase, where also the rest of the test's specification is, thus making it more concise.

The ``setup()`` method is now very simple:
it gets the correct compilation flags from the ``prgenv_flags`` dictionary and applies them to the build system.

.. literalinclude:: ../tutorial/example2.py
  :lines: 1-4,37-64

.. tip::
  A regression test is like any other Python class, so you can freely define your own attributes.
  If you accidentally try to write on a reserved :class:`RegressionTest <reframe.core.pipeline.RegressionTest>` attribute that is not writeable, ReFrame will prevent this and it will throw an error.

Running on Multiple Nodes
-------------------------

So far, all our tests run on a single node.
Depending on the actual system that ReFrame is running, the test may run locally or be submitted to the system's job scheduler.
In this example, we write a regression test for the MPI+OpenMP version of the matrix-vector product.
The source code of this program is in ``tutorial/src/example_matrix_vector_multiplication_mpi_openmp.c``.
The regression test file follows:

.. literalinclude:: ../tutorial/example3.py

This test is pretty much similar to the `test example <#an-alternative-implementation-using-dictionaries>`__ for the OpenMP code we have shown before, except that it adds some information about the configuration of the distributed tasks.
It also restricts the valid systems only to those that support distributed execution.
Let's take the changes step-by-step:

First we need to specify for which partitions this test is meaningful by setting the :attr:`valid_systems <reframe.core.pipeline.RegressionTest.valid_systems>` attribute:

.. literalinclude:: ../tutorial/example3.py
  :lines: 10
  :dedent: 8

We only specify the partitions that are configured with a job scheduler.
If we try to run the generated executable on the login nodes, it will fail.
So we remove this partition from the list of the supported systems.

The most important addition to this check are the variables controlling the distributed execution:

.. literalinclude:: ../tutorial/example3.py
  :lines: 24-26
  :dedent: 8

By setting these variables, we specify that this test should run with 8 MPI tasks in total, using two tasks per node.
Each task may use four logical CPUs.
Based on these variables ReFrame will generate the appropriate scheduler flags to meet that requirement.
For example, for Slurm these variables will result in the following flags:
``--ntasks=8``, ``--ntasks-per-node=2`` and ``--cpus-per-task=4``.
ReFrame provides several more variables for configuring the job submission.
As shown in the following Table, they follow closely the corresponding Slurm options.
For schedulers that do not provide the same functionality, some of the variables may be ignored.

================================================ ===========================================
      :class:`RegressionTest` attribute                    Corresponding SLURM option
================================================ ===========================================
      ``time_limit = (0, 10, 30)``                         ``--time=00:10:30``
      ``use_multithreading = True``                        ``--hint=multithread``
      ``use_multithreading = False``                       ``--hint=nomultithread``
      ``exclusive = True``                                 ``--exclusive``
      ``num_tasks=72``                                     ``--ntasks=72``
      ``num_tasks_per_node=36``                            ``--ntasks-per-node=36``
      ``num_cpus_per_task=4``                              ``--cpus-per-task=4``
      ``num_tasks_per_core=2``                             ``--ntasks-per-core=2``
      ``num_tasks_per_socket=36``                          ``--ntasks-per-socket=36``
================================================ ===========================================

Testing a GPU Code
------------------

In this example, we will create two regression tests for two different GPU versions of our matrix-vector code:
OpenACC and CUDA.
Let's start with the OpenACC regression test:

.. literalinclude:: ../tutorial/example4.py

The things to notice in this test are the restricted list of system partitions and programming environments that this test supports and the use of the :attr:`modules <reframe.core.pipeline.RegressionTest.modules>` variable:

.. literalinclude:: ../tutorial/example4.py
  :lines: 15
  :dedent: 8

The :attr:`modules <reframe.core.pipeline.RegressionTest.modules>` variable takes a list of modules that should be loaded during the setup phase of the test.
In this particular test, we need to load the ``craype-accel-nvidia60`` module, which enables the generation of a GPU binary from an OpenACC code.

It is also important to note that in GPU-enabled tests the number of GPUs for each node have to be specified by setting the corresponding variable :attr:`num_gpus_per_node <reframe.core.pipeline.RegressionTest.num_gpus_per_node>`, as follows:

.. literalinclude:: ../tutorial/example4.py
  :lines: 16
  :dedent: 8

The regression test for the CUDA code is slightly simpler:

.. literalinclude:: ../tutorial/example5.py

ReFrame will recognize the ``.cu`` extension of the source file and it will try to invoke ``nvcc`` for compiling the code.
In this case, there is no need to differentiate across the programming environments, since the compiler will be eventually the same.
``nvcc`` in our example is provided by the ``cudatoolkit`` module, which we list it in the :attr:`modules <reframe.core.pipeline.RegressionTest.modules>` variable.

More Advanced Sanity Checking
-----------------------------

So far we have done a very simple sanity checking.
We are only looking if a specific line is present in the output of the test program.
In this example, we expand the regression test of the serial code, so as to check also if the printed norm of the result vector is correct.

.. literalinclude:: ../tutorial/example6.py

The only difference with our first example is actually the more complex expression to assess the sanity of the test.
Let's go over it line-by-line.
The first thing we do is to extract the norm printed in the standard output.

.. literalinclude:: ../tutorial/example6.py
  :lines: 19-21
  :dedent: 8

The :func:`extractsingle <reframe.utility.sanity.extractsingle>` sanity function extracts some information from a single occurrence (by default the first) of a pattern in a filename.
In our case, this function will extract the ``norm`` `capturing group <https://docs.python.org/3.6/library/re.html#regular-expression-syntax>`__ from the match of the regular expression ``r'The L2 norm of the resulting vector is:\s+(?P<norm>\S+)'`` in standard output, it will convert it to float and it will return it.
Unnamed capturing groups in regular expressions are also supported, which you can reference by their group number.
For example, we could have written the same statement as follows:

.. code-block:: python

          found_norm = sn.extractsingle(
              r'The L2 norm of the resulting vector is:\s+(\S+)',
              self.stdout, 1, float)

Notice that we replaced the ``'norm'`` argument with ``1``, which is the capturing group number.

.. note:: In regular expressions, capturing group ``0`` corresponds always to the whole match.
  In sanity functions dealing with regular expressions, this will yield the whole line that matched.

A useful counterpart of :func:`extractsingle <reframe.utility.sanity.extractsingle>` is the :func:`extractall <reframe.utility.sanity.extractall>` function, which instead of a single occurrence, returns a list of all the occurrences found.
For a more detailed description of this and other sanity functions, please refer to the `sanity function reference <sanity_functions_reference.html>`__.

The next four lines is the actual sanity check:

.. literalinclude:: ../tutorial/example6.py
  :lines: 22-26
  :dedent: 8

This expression combines two conditions that need to be true, in order for the sanity check to succeed:

1. Find in standard output the same line we were looking for already in the first example.
2. Verify that the printed norm does not deviate significantly from the expected value.

The :func:`all <reframe.utility.sanity.all>` function is responsible for combining the results of the individual subexpressions.
It is essentially the Python built-in `all() <https://docs.python.org/3.6/library/functions.html#all>`__ function, exposed as a sanity function, and requires that all the elements of the iterable it takes as an argument evaluate to :class:`True`.
As mentioned before, all the ``assert_*`` functions either return :class:`True` on success or raise :class:`SanityError <reframe.core.exceptions.SanityError>`.
So, if everything goes smoothly, ``sn.all()`` will evaluate to :class:`True` and sanity checking will succeed.

The expression for the second condition is more interesting.
Here, we want to assert that the absolute value of the difference between the expected and the found norm are below a certain value.
The important thing to mention here is that you can combine the results of sanity functions in arbitrary expressions, use them as arguments to other functions, return them from functions, assign them to variables etc.
Remember that sanity functions are not evaluated at the time you call them.
They will be evaluated later by the framework during the sanity checking phase.
If you include the result of a sanity function in an expression, the evaluation of the resulting expression will also be deferred.
For a detailed description of the mechanism behind the sanity functions, please have a look at `"Understanding The Mechanism Of Sanity Functions" <deferrables.html>`__ section.

Writing a Performance Test
--------------------------

An important aspect of regression testing is checking for performance regressions.
ReFrame offers a flexible way of extracting and manipulating performance data from the program output, as well as a comprehensive way of setting performance thresholds per system and system partitions.

In this example, we extend the CUDA test presented `previously <tutorial.html#testing-a-gpu-code>`__, so as to check also the performance of the matrix-vector multiplication.

.. literalinclude:: ../tutorial/example7.py

The are two new variables set in this test that basically enable the performance testing:

:attr:`perf_patterns <reframe.core.pipeline.RegressionTest.perf_patterns>`
  This variable defines which are the performance patterns we are looking for and how to extract the performance values.
:attr:`reference <reframe.core.pipeline.RegressionTest.reference>`
  This variable is a collection of reference values for different systems.

Let's have a closer look at each of them:

.. literalinclude:: ../tutorial/example7.py
  :lines: 20-23
  :dedent: 8

The :attr:`perf_patterns <reframe.core.pipeline.RegressionTest.perf_patterns>` attribute is a dictionary, whose keys are *performance variables* (i.e., arbitrary names assigned to the performance values we are looking for), and its values are *sanity expressions* that specify how to obtain these performance values from the output.
A sanity expression is a Python expression that uses the result of one or more *sanity functions*.
In our example, we name the performance value we are looking for simply as ``perf`` and we extract its value by converting to float the regex capturing group named ``Gflops`` from the line that was matched in the standard output.

Each of the performance variables defined in :attr:`perf_patterns <reframe.core.pipeline.RegressionTest.perf_patterns>` must be resolved in the :attr:`reference <reframe.core.pipeline.RegressionTest.reference>` dictionary of reference values.
When the framework obtains a performance value from the output of the test it searches for a reference value in the :attr:`reference <reframe.core.pipeline.RegressionTest.reference>` dictionary, and then it checks whether the user supplied tolerance is respected.
Let's go over the :attr:`reference <reframe.core.pipeline.RegressionTest.reference>` dictionary of our example and explain its syntax in more detail:

.. literalinclude:: ../tutorial/example7.py
  :lines: 24-28
  :dedent: 8

This is a special type of dictionary that we call ``scoped dictionary``, because it defines scopes for its keys.
We have already seen it being used in the ``environments`` section of the `configuration file <configure.html#environments-configuration>`__ of ReFrame.
In order to resolve a reference value for a performance variable, ReFrame creates the following key ``<current_sys>:<current_part>:<perf_variable>`` and looks it up inside the :attr:`reference <reframe.core.pipeline.RegressionTest.reference>` dictionary.
If our example, since this test is only allowed to run on the ``daint:gpu`` partition of our system, ReFrame will look for the ``daint:gpu:perf`` reference key.
The ``perf`` subkey will then be searched in the following scopes in this order:
``daint:gpu``, ``daint``, ``*``.
The first occurrence will be used as the reference value of the ``perf`` performance variable.
In our example, the ``perf`` key will be resolved in the ``daint:gpu`` scope giving us the reference value.

Reference values in ReFrame are specified as a three-tuple or four-tuple comprising the reference value, the lower and upper thresholds and, optionally, the measurement unit.
Thresholds are specified as decimal fractions of the reference value. For nonnegative reference values, the lower threshold must lie in the [-1,0], whereas the upper threshold may be any positive real number or zero.
In our example, the reference value for this test on ``daint:gpu`` is 50 Gflop/s ±10%. Setting a threshold value to :class:`None` disables the threshold.
If you specify a measurement unit as well, you will be able to log it the performance logs of the test; this is handy when you are inspecting or plotting the performance values.



Combining It All Together
-------------------------

As we have mentioned before and as you have already experienced with the examples in this tutorial, regression tests in ReFrame are written in pure Python.
As a result, you can leverage the language features and capabilities to organize better your tests and decrease the maintenance cost.
In this example, we are going to reimplement all the tests of the tutorial with much less code and in a single file.
Here is the final example code that combines all the tests discussed before:

.. literalinclude:: ../tutorial/example8.py

This test abstracts away the common functionality found in almost all of our tutorial tests (executable options, sanity checking, etc.) to a base class, from which all the concrete regression tests derive.
Each test then redefines only the parts that are specific to it.
Notice also that only the actual tests, i.e., the derived classes, are made visible to the framework through the ``@simple_test`` decorator.
Decorating the base class has now meaning, because it does not correspond to an actual test.

The total line count of this refactored example is less than half of that of the individual tutorial tests.
Another interesting thing to note here is the base class accepting additional additional parameters to its constructor, so that the concrete subclasses can initialize it based on their needs.

Summary
-------

This concludes our ReFrame tutorial.
We have covered all basic aspects of writing regression tests in ReFrame and you should now be able to start experimenting by writing your first useful tests.
The `next section <advanced.html>`__ covers further topics in customizing a regression test to your needs.
