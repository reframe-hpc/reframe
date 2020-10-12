==========================================
 Tutorial 1: Getting Started with ReFrame
==========================================

.. versionadded:: 3.1

This tutorial will give you a first overview of ReFrame and will acquaint you with its basic concepts.
We will start with a simple "Hello, World!" test running with the default configuration and we will expand the example along the way.
We will also explore performance tests and port our tests to an HPC cluster.
The examples of this tutorial can be found under :obj:`tutorials/basics/`.


Getting Ready
-------------

All you need to start off with this tutorial is to have `installed <started.html#getting-the-framework>`__ ReFrame.
If you haven't done so yet, all you need is Python 3.6 and above and to follow the steps below:


.. code:: bash

   git clone https://github.com/eth-cscs/reframe.git
   cd reframe
   ./bootstrap.sh
   ./bin/reframe -V

We're now good to go!


The "Hello, World!" test
------------------------

As simple as it may sound, a series of "naive" "Hello, World!" tests can reveal lots of regressions in the programming environment of HPC clusters, but the bare minimum of those also serves perfectly the purpose of starting this tutorial.
Here is its C version:

.. literalinclude:: ../tutorials/basics/hello/src/hello.c
   :language: c
   :lines: 6-


And here is the ReFrame version of it:

.. literalinclude:: ../tutorials/basics/hello/hello1.py
   :lines: 6-


Regression tests in ReFrame are specially decorated classes that ultimately derive from :class:`RegressionTest <reframe.core.pipeline.RegressionTest>`.
The :func:`@simple_test <reframe.core.decorators.simple_test>` decorator registers a test class with ReFrame and makes it available to the framework.
The test parameters are essentially attributes of the test class and are usually defined in the test class constructor (:func:`__init__` function).
Each test must always set the :attr:`valid_systems <reframe.core.pipeline.RegressionTest.valid_systems>` and :attr:`valid_prog_environs <reframe.core.pipeline.RegressionTest.valid_prog_environs>` attributes.
These define the systems and/or system partitions that this test is allowed to run on, as well as the programming environments that it is valid for.
A programming environment is essentially a compiler toolchain.
We will see later on in the tutorial how a programming environment can be defined.
The generic configuration of ReFrame assumes a single programming environment named ``builtin`` which comprises a C compiler that can be invoked with ``cc``.
In this particular test we set both these attributes to ``['*']``, essentially allowing this test to run everywhere.

Each regression test must always define the :attr:`sanity_patterns <reframe.core.pipeline.RegressionTest.sanity_patterns>` attribute.
This is a `lazily evaluated <deferrables.html>`__ expression that asserts the sanity of the test.
In this particular case, we ask ReFrame to check for the desired phrase in the test's standard output.
Note that ReFrame does not determine the success of a test by its exit code.
The assessment of success is responsibility of the test itself.

Finally, a test must either define an executable to execute or a source file (or source code) to be compiled.
In this example, it is enough to define the source file of our hello program.
ReFrame knows the executable that was produced and will use that to run the test.


Before running the test let's inspect the directory structure surrounding it:

.. code-block:: none

   tutorials/basics/hello
   ├── hello1.py
   └── src
       └── hello.c

Our test is ``hello1.py`` and its resources, i.e., the ``hello.c`` source file, are located inside the ``src/`` subdirectory.
If not specified otherwise, the :attr:`sourcepath <reframe.core.pipeline.RegressionTest.sourcepath>` attribute is always resolved relative to ``src/``.
There is full flexibility in organizing the tests.
Multiple tests may be defined in a single file or they may be split in multiple files.
Similarly, several tests may share the same resources directory or they can simply have their own.

Now it's time to run our first test:

.. code:: bash

   ./bin/reframe -c tutorials/basics/hello/hello1.py -r


.. code-block:: none

   [ReFrame Setup]
     version:           3.3-dev0 (rev: 5d246bff)
     command:           './bin/reframe -c tutorials/basics/hello/hello1.py -r'
     launched by:       user@tresa.local
     working directory: '/Users/user/Repositories/reframe'
     settings file:     '<builtin>'
     check search path: '/Users/user/Repositories/reframe/tutorials/basics/hello/hello1.py'
     stage directory:   '/Users/user/Repositories/reframe/stage'
     output directory:  '/Users/user/Repositories/reframe/output'

   [==========] Running 1 check(s)
   [==========] Started on Mon Oct 12 18:23:30 2020

   [----------] started processing HelloTest (HelloTest)
   [ RUN      ] HelloTest on generic:default using builtin
   [----------] finished processing HelloTest (HelloTest)

   [----------] waiting for spawned checks to finish
   [       OK ] (1/1) HelloTest on generic:default using builtin [compile: 0.389s run: 0.406s total: 0.811s]
   [----------] all spawned checks have finished

   [  PASSED  ] Ran 1 test case(s) from 1 check(s) (0 failure(s))
   [==========] Finished on Mon Oct 12 18:23:31 2020
   Log file(s) saved in: '/var/folders/h7/k7cgrdl13r996m4dmsvjq7v80000gp/T/rfm-00lf_tbi.log'


Perfect! We have verified that we have a functioning C compiler in our system.

When ReFrame runs a test, it copies all its resources to a stage directory and performs all test-related operations (compilation, run, sanity checking etc.) from that directory.
On successful outcome of the test, the stage directory is removed by default, but interesting files are copied to an output directory for archiving and later inspection.
The prefixes of these directories are printed in the first section of the output.
Let's inspect what files ReFrame produced for this test:

.. code-block:: console

   ls output/generic/default/builtin/HelloTest/

.. code-block:: none

   rfm_HelloTest_build.err rfm_HelloTest_build.sh  rfm_HelloTest_job.out
   rfm_HelloTest_build.out rfm_HelloTest_job.err   rfm_HelloTest_job.sh

ReFrame stores in the output directory of the test the build and run scripts it generated for building and running the code along with their standard output and error.
All these files are prefixed with ``rfm_``.

ReFrame also generates a detailed JSON report for the whole regression testing session.
By default, this is stored inside the ``${HOME}/.reframe/reports`` directory and a new report file is generated every time ReFrame is run, but you can control this through the :option:`--report-file` command-line option.

Here are the contents of the report file for our first ReFrame run:


.. code-block:: console

   cat ~/.reframe/reports/run-report-0.json

.. code-block:: javascript

   {
     "session_info": {
       "cmdline": "./bin/reframe -c tutorials/basics/hello/hello1.py -r",
       "config_file": "<builtin>",
       "data_version": "1.0",
       "hostname": "dhcp-133-44.cscs.ch",
       "prefix_output": "/Users/user/Repositories/reframe/output",
       "prefix_stage": "/Users/user/Repositories/reframe/stage",
       "user": "user",
       "version": "3.1-dev2 (rev: 272e1aae)",
       "workdir": "/Users/user/Repositories/reframe",
       "time_start": "2020-07-24T11:05:46+0200",
       "time_end": "2020-07-24T11:05:47+0200",
       "time_elapsed": 0.7293069362640381,
       "num_cases": 1,
       "num_failures": 0
     },
     "runs": [
       {
         "num_cases": 1,
         "num_failures": 0,
         "runid": 0,
         "testcases": [
           {
             "build_stderr": "rfm_HelloTest_build.err",
             "build_stdout": "rfm_HelloTest_build.out",
             "description": "HelloTest",
             "environment": "builtin",
             "fail_reason": null,
             "fail_phase": null,
             "jobid": 85063,
             "job_stderr": "rfm_HelloTest_job.err",
             "job_stdout": "rfm_HelloTest_job.out",
             "name": "HelloTest",
             "maintainers": [],
             "nodelist": [
               "dhcp-133-44.cscs.ch"
             ],
             "outputdir": "/Users/user/Repositories/reframe/output/generic/default/builtin/HelloTest",
             "perfvars": null,
             "result": "success",
             "stagedir": null,
             "scheduler": "local",
             "system": "generic:default",
             "tags": [],
             "time_compile": 0.3776402473449707,
             "time_performance": 4.506111145019531e-05,
             "time_run": 0.2992382049560547,
             "time_sanity": 0.0005609989166259766,
             "time_setup": 0.0031709671020507812,
             "time_total": 0.7213571071624756
           }
         ]
       }
     ]
   }


More of "Hello, World!"
-----------------------

We want to extend our test and run a C++ "Hello, World!" as well.
We could simply copy paste the ``hello1.py`` and change the source file extension to refer to the C++ source code.
But this duplication is something that we generally want to avoid.
ReFrame allows you to avoid this in several ways but the most compact is to define the new test as follows:


.. literalinclude:: ../tutorials/basics/hello/hello2.py
   :lines: 6-


This is exactly the same test as the ``hello1.py`` except that it is decorated with the :func:`@parameterized_test <reframe.core.decorators.parameterized_test>` decorator instead of the :func:`@simple_test <reframe.core.decorators.simple_test>`.
Also the constructor of the test now takes an argument.
The :func:`@parameterized_test <>` decorator instructs ReFrame to instantiate a test class with different parameters.
In this case the test will be instantiated for both C and C++ and then we use the ``lang`` parameter directly as the extension of the source file.
Let's run now the test:


.. code-block:: console

   ./bin/reframe -c tutorials/basics/hello/hello2.py -r

.. code-block:: none

   [ReFrame Setup]
     version:           3.3-dev0 (rev: 5d246bff)
     command:           './bin/reframe -c tutorials/basics/hello/hello2.py -r'
     launched by:       user@tresa.local
     working directory: '/Users/user/Repositories/reframe'
     settings file:     '<builtin>'
     check search path: '/Users/user/Repositories/reframe/tutorials/basics/hello/hello2.py'
     stage directory:   '/Users/user/Repositories/reframe/stage'
     output directory:  '/Users/user/Repositories/reframe/output'

   [==========] Running 2 check(s)
   [==========] Started on Mon Oct 12 18:24:31 2020

   [----------] started processing HelloMultiLangTest_c (HelloMultiLangTest_c)
   [ RUN      ] HelloMultiLangTest_c on generic:default using builtin
   [----------] finished processing HelloMultiLangTest_c (HelloMultiLangTest_c)

   [----------] started processing HelloMultiLangTest_cpp (HelloMultiLangTest_cpp)
   [ RUN      ] HelloMultiLangTest_cpp on generic:default using builtin
   [     FAIL ] (1/2) HelloMultiLangTest_cpp on generic:default using builtin [compile: 0.001s run: n/a total: 0.009s]
   [----------] finished processing HelloMultiLangTest_cpp (HelloMultiLangTest_cpp)

   [----------] waiting for spawned checks to finish
   [       OK ] (2/2) HelloMultiLangTest_c on generic:default using builtin [compile: 0.254s run: 0.286s total: 0.555s]
   [----------] all spawned checks have finished

   [  FAILED  ] Ran 2 test case(s) from 2 check(s) (1 failure(s))
   [==========] Finished on Mon Oct 12 18:24:32 2020

   ==============================================================================
   SUMMARY OF FAILURES
   ------------------------------------------------------------------------------
   FAILURE INFO for HelloMultiLangTest_cpp
     * Test Description: HelloMultiLangTest_cpp
     * System partition: generic:default
     * Environment: builtin
     * Stage directory: /Users/user/Repositories/reframe/stage/generic/default/builtin/HelloMultiLangTest_cpp
     * Node list: None
     * Job type: local (id=None)
     * Maintainers: []
     * Failing phase: compile
     * Rerun with '-n HelloMultiLangTest_cpp -p builtin --system generic:default'
     * Reason: build system error: I do not know how to compile a C++ program
   ------------------------------------------------------------------------------
   Log file(s) saved in: '/var/folders/h7/k7cgrdl13r996m4dmsvjq7v80000gp/T/rfm-lbpo8oan.log'


Oops! The C++ test has failed.
ReFrame complains that it does not know how to compile a C++ program.
Remember our discussion above that the default configuration of ReFrame defines a minimal programming environment named ``builtin`` which only knows of a ``cc`` compiler.
We will fix that in a moment, but before doing that it's worth looking into the failure information provided for the test.
For each failed test, ReFrame will print a short summary with information about the system partition and the programming environment that the test failed for, its job or process id (if any), the nodes it was running on, its stage directory, the phase that failed etc.

When a test fails its stage directory is kept intact, so that users can inspect the failure and try to reproduce it manually.
In this case, the stage directory contains only the "Hello, World" source files, since ReFrame could not produce a build script for the C++ test, as it doesn't know to compile a C++ program for the moment.

.. code-block:: console

   ls stage/generic/default/builtin/HelloMultiLangTest_cpp


.. code-block:: none

   hello.c   hello.cpp


Let's go on and fix this failure by defining a new system and programming environments for the machine we are running on.
We start off by copying the generic configuration file that ReFrame uses.
Note that you should *not* edit this configuration file in place.

.. code-block:: console

   cp reframe/core/settings.py tutorials/config/mysettings.py


Here is how the new configuration file looks like with the needed additions highlighted:

.. literalinclude:: ../tutorials/config/settings.py
   :lines: 10-24,64-85,118-
   :emphasize-lines: 3-15,31-42

Here we define a system named ``catalina`` that has one partition named ``default``.
This partition makes no use of any workload manager, but instead launches any jobs locally as OS processes.
Two programming environments are relevant for that partition, namely ``gnu`` and ``clang``, which are defined in the section :js:attr:`environments` of the configuration file.
The ``gnu`` programming environment provides GCC 9, whereas the ``clang`` one provides the Clang compiler from the system.
Notice, how you can define the actual commands for invoking the C, C++ and Fortran compilers in each programming environment.
Finally, the new system that we defined may be identified by the hostname ``tresa`` (see the :js:attr:`hostnames` configuration parameter).
This will help ReFrame to automatically pick the right configuration when running on it.
Notice, how the ``generic`` system matches any hostname, so that it acts as a fallback system.

.. note::

   The different systems in the configuration file are tried in order and the first match is picked.
   This practically means that the more general the selection pattern for a system is, the lower in the list of systems it should be.

The :doc:`configure` page describes the configuration file in more detail and the :doc:`config_reference` provides a complete reference guide of all the configuration options of ReFrame.

Let's now rerun our "Hello, World!" tests:


.. code-block:: console

   ./bin/reframe -C tutorials/config/mysettings.py -c tutorials/basics/hello/hello2.py -r


.. code-block:: none

   [ReFrame Setup]
     version:           3.3-dev0 (rev: 5d246bff)
     command:           './bin/reframe -C tutorials/config/settings.py -c tutorials/basics/hello/hello2.py -r'
     launched by:       user@tresa.local
     working directory: '/Users/user/Repositories/reframe'
     settings file:     'tutorials/config/settings.py'
     check search path: '/Users/user/Repositories/reframe/tutorials/basics/hello/hello2.py'
     stage directory:   '/Users/user/Repositories/reframe/stage'
     output directory:  '/Users/user/Repositories/reframe/output'

   [==========] Running 2 check(s)
   [==========] Started on Mon Oct 12 18:28:48 2020

   [----------] started processing HelloMultiLangTest_c (HelloMultiLangTest_c)
   [ RUN      ] HelloMultiLangTest_c on catalina:default using gnu
   [ RUN      ] HelloMultiLangTest_c on catalina:default using clang
   [----------] finished processing HelloMultiLangTest_c (HelloMultiLangTest_c)

   [----------] started processing HelloMultiLangTest_cpp (HelloMultiLangTest_cpp)
   [ RUN      ] HelloMultiLangTest_cpp on catalina:default using gnu
   [ RUN      ] HelloMultiLangTest_cpp on catalina:default using clang
   [----------] finished processing HelloMultiLangTest_cpp (HelloMultiLangTest_cpp)

   [----------] waiting for spawned checks to finish
   [       OK ] (1/4) HelloMultiLangTest_cpp on catalina:default using gnu [compile: 1.077s run: 1.475s total: 2.566s]
   [       OK ] (2/4) HelloMultiLangTest_c on catalina:default using gnu [compile: 4.128s run: 2.860s total: 7.004s]
   [       OK ] (3/4) HelloMultiLangTest_c on catalina:default using clang [compile: 0.241s run: 2.741s total: 2.998s]
   [       OK ] (4/4) HelloMultiLangTest_cpp on catalina:default using clang [compile: 1.399s run: 0.356s total: 1.770s]
   [----------] all spawned checks have finished

   [  PASSED  ] Ran 4 test case(s) from 2 check(s) (0 failure(s))
   [==========] Finished on Mon Oct 12 18:28:56 2020
   Log file(s) saved in: '/var/folders/h7/k7cgrdl13r996m4dmsvjq7v80000gp/T/rfm-a_dt6nro.log'


Notice how the same tests are now tried with both the ``gnu`` and ``clang`` programming environments, without having to touch them at all!
That's one of the powerful features of ReFrame and we shall see later on, how easily we can port our tests to an HPC cluster with minimal changes.
In order to instruct ReFrame to use our configuration file, we use the ``-C`` command line option.
Since we don't want to type it throughout the tutorial, we will now set it in the environment:

.. code-block:: console

   export RFM_CONFIG_FILE=$(pwd)/tutorials/config/mysettings.py


A Multithreaded "Hello, World!"
-------------------------------

We extend our C++ "Hello, World!" example to print the greetings from multiple threads:


.. literalinclude:: ../tutorials/basics/hellomp/src/hello_threads.cpp
   :language: cpp
   :lines: 6-

This program takes as argument the number of threads it will create and it uses ``std::thread``, which is C++11 addition, meaning that we will need to pass ``-std=c++11`` to our compilers.
Here is the corresponding ReFrame test, where the new concepts introduced are highlighted:

.. literalinclude:: ../tutorials/basics/hellomp/hellomp1.py
   :lines: 6-
   :emphasize-lines: 11-13

ReFrame delegates the compilation of a test to a *build system*, which is an abstraction of the steps needed to compile the test.
Build systems take also care of interactions with the programming environment if necessary.
Compilation flags are a property of the build system.
If not explicitly specified, ReFrame will try to pick the correct build system (e.g., CMake, Autotools etc.) by inspecting the test resources, but in cases as the one presented here where we need to set the compilation flags, we need to specify a build system explicitly.
In this example, we instruct ReFrame to compile a single source file using the ``-std=c++11 -Wall`` compilation flags.
Finally, we set the arguments to be passed to the generated executable in :attr:`executable_opts <reframe.core.pipeline.RegressionTest.executable_opts>`.


.. code-block:: none

   [ReFrame Setup]
     version:           3.3-dev0 (rev: 5d246bff)
     command:           './bin/reframe -c tutorials/basics/hellomp/hellomp1.py -r'
     launched by:       user@tresa.local
     working directory: '/Users/user/Repositories/reframe'
     settings file:     '/Users/user/Repositories/reframe/tutorials/config/settings.py'
     check search path: '/Users/user/Repositories/reframe/tutorials/basics/hellomp/hellomp1.py'
     stage directory:   '/Users/user/Repositories/reframe/stage'
     output directory:  '/Users/user/Repositories/reframe/output'

   [==========] Running 1 check(s)
   [==========] Started on Mon Oct 12 20:02:37 2020

   [----------] started processing HelloThreadedTest (HelloThreadedTest)
   [ RUN      ] HelloThreadedTest on catalina:default using gnu
   [ RUN      ] HelloThreadedTest on catalina:default using clang
   [----------] finished processing HelloThreadedTest (HelloThreadedTest)

   [----------] waiting for spawned checks to finish
   [       OK ] (1/2) HelloThreadedTest on catalina:default using gnu [compile: 1.591s run: 1.205s total: 2.816s]
   [       OK ] (2/2) HelloThreadedTest on catalina:default using clang [compile: 1.141s run: 0.309s total: 1.465s]
   [----------] all spawned checks have finished

   [  PASSED  ] Ran 2 test case(s) from 1 check(s) (0 failure(s))
   [==========] Finished on Mon Oct 12 20:02:40 2020
   Log file(s) saved in: '/var/folders/h7/k7cgrdl13r996m4dmsvjq7v80000gp/T/rfm-h_itoc1k.log'


Everything looks fine, but let's inspect the actual output of one of the tests:


.. code-block:: console

   cat output/catalina/default/clang/HelloThreadedTest/rfm_HelloThreadedTest_job.out


.. code-block:: none

   [[[[    8] Hello, World!
   1] Hello, World!
   5[[0[ 7] Hello, World!
   ] ] Hello, World!
   [ Hello, World!
   6[] Hello, World!
   9] Hello, World!
    2 ] Hello, World!
   4] [[10 3] Hello, World!
   ] Hello, World!
   [Hello, World!
   11] Hello, World!
   [12] Hello, World!
   [13] Hello, World!
   [14] Hello, World!
   [15] Hello, World!


Not exactly what we were looking for!
In the following we write a more robust sanity check that can catch this havoc.


-----------------------------
More advanced sanity checking
-----------------------------

Sanity checking of a test's outcome is quite powerful in ReFrame.
So far, we have seen only a ``grep``-like search for a string in the output, but ReFrame's ``sanity_patterns`` are much more capable than this.
In fact, you can practically do almost any operation in the output and process it as you would like before assessing the test's sanity.
The syntax feels also quite natural since it is fully integrated in Python.

In the following we extend the sanity checking of the multithreaded "Hello, World!", such that not only the output pattern we are looking for is more restrictive, but also we check that all the threads produce a greetings line.

.. literalinclude:: ../tutorials/basics/hellomp/hellomp2.py
   :lines: 6-
   :emphasize-lines: 14-16

The sanity checking is straightforward.
We find all the matches of the required pattern, we count them and finally we check their number.
Both statements here are lazily evaluated.
They will not be executed where they appear, but rather at the sanity checking phase.
ReFrame provides lazily evaluated counterparts for most of the builtin Python functions, such the :func:`len` function here.
Also whole expressions can be lazily evaluated if one of the operands is deferred, as is the case in this example with the assignment to ``num_messages``.
This makes the sanity checking mechanism quite powerful and straightforward to reason about, without having to rely on complex pattern matching techniques.
:doc:`sanity_functions_reference` provides a complete reference of the sanity functions provided by ReFrame, but users can also define their own, as described in :doc:`deferrables`.


Let's run this version of the test now and see if it fails:

.. code-block:: console

   ./bin/reframe -c tutorials/basics/hellomp/hellomp2.py -r

.. code-block:: none

   [ReFrame Setup]
     version:           3.3-dev0 (rev: 5d246bff)
     command:           './bin/reframe -c tutorials/basics/hellomp/hellomp2.py -r'
     launched by:       user@tresa.local
     working directory: '/Users/user/Repositories/reframe'
     settings file:     '/Users/user/Repositories/reframe/tutorials/config/settings.py'
     check search path: '/Users/user/Repositories/reframe/tutorials/basics/hellomp/hellomp2.py'
     stage directory:   '/Users/user/Repositories/reframe/stage'
     output directory:  '/Users/user/Repositories/reframe/output'

   [==========] Running 1 check(s)
   [==========] Started on Mon Oct 12 20:04:59 2020

   [----------] started processing HelloThreadedExtendedTest (HelloThreadedExtendedTest)
   [ RUN      ] HelloThreadedExtendedTest on catalina:default using gnu
   [ RUN      ] HelloThreadedExtendedTest on catalina:default using clang
   [----------] finished processing HelloThreadedExtendedTest (HelloThreadedExtendedTest)

   [----------] waiting for spawned checks to finish
   [     FAIL ] (1/2) HelloThreadedExtendedTest on catalina:default using gnu [compile: 1.222s run: 0.891s total: 2.130s]
   [     FAIL ] (2/2) HelloThreadedExtendedTest on catalina:default using clang [compile: 0.835s run: 0.167s total: 1.018s]
   [----------] all spawned checks have finished

   [  FAILED  ] Ran 2 test case(s) from 1 check(s) (2 failure(s))
   [==========] Finished on Mon Oct 12 20:05:02 2020

   ==============================================================================
   SUMMARY OF FAILURES
   ------------------------------------------------------------------------------
   FAILURE INFO for HelloThreadedExtendedTest
     * Test Description: HelloThreadedExtendedTest
     * System partition: catalina:default
     * Environment: gnu
     * Stage directory: /Users/user/Repositories/reframe/stage/catalina/default/gnu/HelloThreadedExtendedTest
     * Node list: tresa.local
     * Job type: local (id=60355)
     * Maintainers: []
     * Failing phase: sanity
     * Rerun with '-n HelloThreadedExtendedTest -p gnu --system catalina:default'
     * Reason: sanity error: 12 != 16
   ------------------------------------------------------------------------------
   FAILURE INFO for HelloThreadedExtendedTest
     * Test Description: HelloThreadedExtendedTest
     * System partition: catalina:default
     * Environment: clang
     * Stage directory: /Users/user/Repositories/reframe/stage/catalina/default/clang/HelloThreadedExtendedTest
     * Node list: tresa.local
     * Job type: local (id=60366)
     * Maintainers: []
     * Failing phase: sanity
     * Rerun with '-n HelloThreadedExtendedTest -p clang --system catalina:default'
     * Reason: sanity error: 6 != 16
   ------------------------------------------------------------------------------
   Log file(s) saved in: '/var/folders/h7/k7cgrdl13r996m4dmsvjq7v80000gp/T/rfm-zz7x_5c8.log'


As expected, only some of lines are printed correctly which makes the test fail.
To fix this test, we need to compile with ``-DSYNC_MESSAGES``, which will synchronize the printing of messages.

.. literalinclude:: ../tutorials/basics/hellomp/hellomp3.py
   :lines: 6-
   :emphasize-lines: 13


Writing A Performance Test
--------------------------

An important aspect of regression testing is checking for performance regressions.
In this example, we will write a test that downloads the `STREAM <http://www.cs.virginia.edu/stream/ref.html>`__ benchmark, compiles it, runs it and records its performance.
In the test below, we highlight the lines that introduce new concepts.

.. literalinclude:: ../tutorials/basics/stream/stream1.py
   :lines: 6-
   :emphasize-lines: 10-12,17-20,23-32

First of all, notice that we restrict the programming environments to ``gnu`` only, since this test requires OpenMP, which our installation of Clang does not have.
The next thing to notice is the :attr:`prebuild_cmds <reframe.core.pipeline.RegressionTest.prebuild_cmds>` attribute, which provides a list of commands to be executed before the build step.
These commands will be executed from the test's stage directory.
In this case, we just fetch the source code of the benchmark.
For running the benchmark, we need to set the OpenMP number of threads and pin them to the right CPUs through the ``OMP_NUM_THREADS`` and ``OMP_PLACES`` environment variables.
You can set environment variables in a ReFrame test through the :attr:`variables <reframe.core.pipeline.RegressionTest.variables>` dictionary.

What makes a ReFrame test a performance test is the definition of the :attr:`perf_patterns <reframe.core.pipeline.RegressionTest.perf_patterns>` attribute.
This is a dictionary where the keys are *performance variables* and the values are lazily evaluated expressions for extracting the performance variable values from the test's output.
In this example, we extract four performance variables, namely the memory bandwidth values for each of the "Copy", "Scale", "Add" and "Triad" sub-benchmarks of STREAM and we do so by using the :func:`extractsingle <reframe.utility.sanity.extractsingle>` sanity function.
For each of the sub-benchmarks we extract the "Best Rate MB/s" column of the output (see below) and wee convert that to a float.

.. code-block:: none

   Function    Best Rate MB/s  Avg time     Min time     Max time
   Copy:           24939.4     0.021905     0.021527     0.022382
   Scale:          16956.3     0.031957     0.031662     0.032379
   Add:            18648.2     0.044277     0.043184     0.046349
   Triad:          19133.4     0.042935     0.042089     0.044283


Let's run the test now:


.. code-block:: console

   ./bin/reframe -c tutorials/basics/stream/stream1.py -r --performance-report

The :option:`--performance-report` will generate a short report at the end for each performance test that has run.


.. code-block:: none

   [ReFrame Setup]
     version:           3.3-dev0 (rev: 5d246bff)
     command:           './bin/reframe -c tutorials/basics/stream/stream1.py -r --performance-report'
     launched by:       user@tresa.local
     working directory: '/Users/user/Repositories/reframe'
     settings file:     '/Users/user/Repositories/reframe/tutorials/config/settings.py'
     check search path: '/Users/user/Repositories/reframe/tutorials/basics/stream/stream1.py'
     stage directory:   '/Users/user/Repositories/reframe/stage'
     output directory:  '/Users/user/Repositories/reframe/output'

   [==========] Running 1 check(s)
   [==========] Started on Mon Oct 12 20:06:09 2020

   [----------] started processing StreamTest (StreamTest)
   [ RUN      ] StreamTest on catalina:default using gnu
   [----------] finished processing StreamTest (StreamTest)

   [----------] waiting for spawned checks to finish
   [       OK ] (1/1) StreamTest on catalina:default using gnu [compile: 1.386s run: 2.377s total: 3.780s]
   [----------] all spawned checks have finished

   [  PASSED  ] Ran 1 test case(s) from 1 check(s) (0 failure(s))
   [==========] Finished on Mon Oct 12 20:06:13 2020
   ==============================================================================
   PERFORMANCE REPORT
   ------------------------------------------------------------------------------
   StreamTest
   - catalina:default
      - gnu
         * num_tasks: 1
         * Copy: 24326.7 None
         * Scale: 16664.2 None
         * Add: 18398.7 None
         * Triad: 18930.6 None
   ------------------------------------------------------------------------------
   Log file(s) saved in: '/var/folders/h7/k7cgrdl13r996m4dmsvjq7v80000gp/T/rfm-gczplnic.log'


-----------------------
Adding reference values
-----------------------

A performance test would not be so meaningful, if we couldn't test the obtained performance against a reference value.
ReFrame offers the possibility to set references for each of the performance variables defined in a test and also set different references for different systems.
In the following example, we set the reference values for all the STREAM sub-benchmarks for the system we are currently running on.

.. note::

   Optimizing STREAM benchmark performance is outside the scope of this tutorial.


.. literalinclude:: ../tutorials/basics/stream/stream2.py
   :lines: 6-
   :emphasize-lines: 33-


The performance reference tuple consists of the reference value, the lower and upper thresholds expressed as fractional numbers relative to the reference value, and the unit of measurement.
If any of the thresholds is not relevant, :class:`None` may be used instead.

If any obtained performance value is beyond its respective thresholds, the test will fail with a summary as shown below:

.. code-block:: none

   FAILURE INFO for StreamWithRefTest
     * Test Description: StreamWithRefTest
     * System partition: catalina:default
     * Environment: gnu
     * Stage directory: /Users/user/Repositories/reframe/stage/catalina/default/gnu/StreamWithRefTest
     * Node list: tresa.local
     * Job type: local (id=62114)
     * Maintainers: []
     * Failing phase: performance
     * Rerun with '-n StreamWithRefTest -p gnu --system catalina:default'
     * Reason: performance error: failed to meet reference: Copy=24586.5, expected 55200 (l=52440.0, u=57960.0)



------------------------------
Examining the performance logs
------------------------------

ReFrame has a powerful mechanism for logging its activities as well as performance data.
It supports different types of log channels and it can send data simultaneously in any number of them.
For example, performance data might be logged in files and the same time being send to Syslog or to a centralized log management server.
By default (i.e., starting off from the builtin configuration file), ReFrame sends performance data to files per test under the ``perflogs/`` directory:

.. code-block:: none

   perflogs
   └── catalina
       └── default
           ├── StreamTest.log
           └── StreamWithRefTest.log

ReFrame creates a log file per test per system and per partition and appends to it every time the test is run on that system/partition combination.
Let's inspect the log file from our last test:

.. code-block:: console

   tail perflogs/catalina/default/StreamWithRefTest.log


.. code-block:: none

   2020-06-24T00:27:06|reframe 3.1-dev0 (rev: 9d92d0ec)|StreamWithRefTest on catalina:default using gnu|jobid=58384|Copy=24762.2|ref=25200 (l=-0.05, u=0.05)|MB/s
   2020-06-24T00:27:06|reframe 3.1-dev0 (rev: 9d92d0ec)|StreamWithRefTest on catalina:default using gnu|jobid=58384|Scale=16784.6|ref=16800 (l=-0.05, u=0.05)|MB/s
   2020-06-24T00:27:06|reframe 3.1-dev0 (rev: 9d92d0ec)|StreamWithRefTest on catalina:default using gnu|jobid=58384|Add=18553.8|ref=18500 (l=-0.05, u=0.05)|MB/s
   2020-06-24T00:27:06|reframe 3.1-dev0 (rev: 9d92d0ec)|StreamWithRefTest on catalina:default using gnu|jobid=58384|Triad=18679.0|ref=18800 (l=-0.05, u=0.05)|MB/s
   2020-06-24T12:42:07|reframe 3.1-dev0 (rev: 138cbd68)|StreamWithRefTest on catalina:default using gnu|jobid=62114|Copy=24586.5|ref=55200 (l=-0.05, u=0.05)|MB/s
   2020-06-24T12:42:07|reframe 3.1-dev0 (rev: 138cbd68)|StreamWithRefTest on catalina:default using gnu|jobid=62114|Scale=16880.6|ref=16800 (l=-0.05, u=0.05)|MB/s
   2020-06-24T12:42:07|reframe 3.1-dev0 (rev: 138cbd68)|StreamWithRefTest on catalina:default using gnu|jobid=62114|Add=18570.4|ref=18500 (l=-0.05, u=0.05)|MB/s
   2020-06-24T12:42:07|reframe 3.1-dev0 (rev: 138cbd68)|StreamWithRefTest on catalina:default using gnu|jobid=62114|Triad=19048.3|ref=18800 (l=-0.05, u=0.05)|MB/s

Several information are printed for each run, such as the performance variables, their value, their references and thresholds etc.
The default format is in a form suitable for easy parsing, but you may fully control not only the format, but also what is being logged from the configuration file.
:doc:`configure` and :doc:`config_reference` cover logging in ReFrame in much more detail.



Porting The Tests to an HPC cluster
-----------------------------------

It's now time to port our tests to an HPC cluster.
Obviously, HPC clusters are much more complex than our laptop or PC.
Usually there are many more compilers, the user environment is handled in a different way, and the way to launch the tests varies significantly, since you have to go through a workload manager in order to acces the actual compute nodes.
Besides that, there might be multiple types of compute nodes that we would like to run our tests on, but each type might be accessed in a different way.
It is already apparent that porting even an as simple as a "Hello, World" test to such a system is not that straightforward.
As we shall see in this section, ReFrame makes that pretty easy.

--------------------------
Adapting the configuration
--------------------------

Our target system is the `Piz Daint <https://www.cscs.ch/computers/piz-daint/>`__ supercomputer at CSCS, but you can adapt the process to your target HPC system.
In ReFrame, all the details of the various interactions of a test with the system environment are handled transparently and are set up in its configuration file.
Let's extend our configuration file for Piz Daint.


.. literalinclude:: ../tutorials/config/settings.py
   :lines: 10-46,53-
   :emphasize-lines: 16-48,70-101


First of all, we need to define a new system and set the list of hostnames that will help ReFrame identify it.
We also set the :js:attr:`modules_system` configuration parameter to instruct ReFrame that this system makes use of the `environment modules <http://modules.sourceforge.net/>`__ for managing the user environment.
Then we define the system partitions that we want to test.
In this case, we define three partitions:

1. the login nodes,
2. the multicore partition (2x Broadwell CPUs per node) and
3. the hybrid partition (1x Haswell CPU + 1x Pascal GPU).

.. |srun| replace:: :obj:`srun`
.. _srun: https://slurm.schedmd.com/srun.html

The login nodes are pretty much similar to the ``catalina:default`` partition which corresponded to our laptop: tests will be launched and run locally.
The other two partitions are handled by `Slurm <https://slurm.schedmd.com/>`__ and parallel jobs are launched using the |srun|_ command.
Additionally, in order to access the different types of nodes represented by those partitions, users have to specify either ``-C mc`` or ``-C gpu`` options along with their account.
This is what we do exactly with the :js:attr:`access` partition configuration option.

.. note::

   System partitions in ReFrame do not necessarily correspond to real job scheduler partitions.

Piz Daint's programming environment offers four compilers: Cray, GNU, Intel and PGI.
We want to test all of them, so we include them in the :js:attr:`environs` lists.
Notice that we do not include Clang in the list, since there is no such compiler on this particular system.

Before looking into the definition of the new environments for the four compilers, it is worth mentioning the :js:attr:`max_jobs` parameter.
This parameter specifies the maximum number of ReFrame test jobs that can be simultaneously in flight.
ReFrame will try to keep concurrency close to this limit (but not exceeding it).
By default, this is set to ``8``, so you are advised to set it to a higher number if you want to increase the throughput of completed tests.

The new environments are defined similarly to the ones we had for our local system, except that now we set two more parameters: the :js:attr:`modules` and the :js:attr:`target_systems`.
The :js:attr:`modules` parameter is a list of environment modules that needs to be loaded, in order to make available this compiler.
The :js:attr:`target_systems` parameter restricts the environment definition to a list of specific systems or system partitions.
This allows us to redefine environments for different systems, as for example the ``gnu`` environment in this case.
ReFrame will always pick the definition that is a closest match for the current system.
In this example, it will pick the second definition for ``gnu`` whenever it runs on the system named ``daint``, and the first in every other occasion.

-----------------
Running the tests
-----------------

We are now ready to run our tests on Piz Daint.
We will only do so with the final versions of the tests from the previous section, which we will select using :option:`-n` option.

.. code-block:: console

   ./bin/reframe -C tutorials/config/settings.py -c tutorials/basics/ -R -n 'HelloMultiLangTest|HelloThreadedExtended2Test|StreamWithRefTest' --performance-report -r


..  code-block:: none

   [ReFrame Setup]
     version:           3.3-dev0 (rev: cb974c13)
     command:           './bin/reframe -C tutorials/config/settings.py -c tutorials/basics/ -R -n HelloMultiLangTest|HelloThreadedExtended2Test|StreamWithRefTest --performance-report -r'
     launched by:       user@dom101
     working directory: '/users/user/Devel/reframe'
     settings file:     'tutorials/config/settings.py'
     check search path: (R) '/users/user/Devel/reframe/tutorials/basics'
     stage directory:   '/users/user/Devel/reframe/stage'
     output directory:  '/users/user/Devel/reframe/output'

   [==========] Running 4 check(s)
   [==========] Started on Mon Oct 12 20:12:34 2020

   [----------] started processing HelloMultiLangTest_c (HelloMultiLangTest_c)
   [ RUN      ] HelloMultiLangTest_c on daint:login using gnu
   [ RUN      ] HelloMultiLangTest_c on daint:login using intel
   [ RUN      ] HelloMultiLangTest_c on daint:login using pgi
   [ RUN      ] HelloMultiLangTest_c on daint:login using cray
   [ RUN      ] HelloMultiLangTest_c on daint:gpu using gnu
   [ RUN      ] HelloMultiLangTest_c on daint:gpu using intel
   [ RUN      ] HelloMultiLangTest_c on daint:gpu using pgi
   [ RUN      ] HelloMultiLangTest_c on daint:gpu using cray
   [ RUN      ] HelloMultiLangTest_c on daint:mc using gnu
   [ RUN      ] HelloMultiLangTest_c on daint:mc using intel
   [ RUN      ] HelloMultiLangTest_c on daint:mc using pgi
   [ RUN      ] HelloMultiLangTest_c on daint:mc using cray
   [----------] finished processing HelloMultiLangTest_c (HelloMultiLangTest_c)

   [----------] started processing HelloMultiLangTest_cpp (HelloMultiLangTest_cpp)
   [ RUN      ] HelloMultiLangTest_cpp on daint:login using gnu
   [ RUN      ] HelloMultiLangTest_cpp on daint:login using intel
   [ RUN      ] HelloMultiLangTest_cpp on daint:login using pgi
   [ RUN      ] HelloMultiLangTest_cpp on daint:login using cray
   [ RUN      ] HelloMultiLangTest_cpp on daint:gpu using gnu
   [ RUN      ] HelloMultiLangTest_cpp on daint:gpu using intel
   [ RUN      ] HelloMultiLangTest_cpp on daint:gpu using pgi
   [ RUN      ] HelloMultiLangTest_cpp on daint:gpu using cray
   [ RUN      ] HelloMultiLangTest_cpp on daint:mc using gnu
   [ RUN      ] HelloMultiLangTest_cpp on daint:mc using intel
   [ RUN      ] HelloMultiLangTest_cpp on daint:mc using pgi
   [ RUN      ] HelloMultiLangTest_cpp on daint:mc using cray
   [----------] finished processing HelloMultiLangTest_cpp (HelloMultiLangTest_cpp)

   [----------] started processing HelloThreadedExtended2Test (HelloThreadedExtended2Test)
   [ RUN      ] HelloThreadedExtended2Test on daint:login using gnu
   [ RUN      ] HelloThreadedExtended2Test on daint:login using intel
   [ RUN      ] HelloThreadedExtended2Test on daint:login using pgi
   [ RUN      ] HelloThreadedExtended2Test on daint:login using cray
   [ RUN      ] HelloThreadedExtended2Test on daint:gpu using gnu
   [ RUN      ] HelloThreadedExtended2Test on daint:gpu using intel
   [ RUN      ] HelloThreadedExtended2Test on daint:gpu using pgi
   [ RUN      ] HelloThreadedExtended2Test on daint:gpu using cray
   [ RUN      ] HelloThreadedExtended2Test on daint:mc using gnu
   [ RUN      ] HelloThreadedExtended2Test on daint:mc using intel
   [ RUN      ] HelloThreadedExtended2Test on daint:mc using pgi
   [ RUN      ] HelloThreadedExtended2Test on daint:mc using cray
   [----------] finished processing HelloThreadedExtended2Test (HelloThreadedExtended2Test)

   [----------] started processing StreamWithRefTest (StreamWithRefTest)
   [ RUN      ] StreamWithRefTest on daint:login using gnu
   [ RUN      ] StreamWithRefTest on daint:gpu using gnu
   [ RUN      ] StreamWithRefTest on daint:mc using gnu
   [----------] finished processing StreamWithRefTest (StreamWithRefTest)

   [----------] waiting for spawned checks to finish
   [       OK ] ( 1/39) HelloThreadedExtended2Test on daint:gpu using intel [compile: 2.271s run: 26.769s total: 29.055s]
   [       OK ] ( 2/39) HelloMultiLangTest_c on daint:gpu using cray [compile: 0.154s run: 75.494s total: 75.661s]
   [       OK ] ( 3/39) HelloThreadedExtended2Test on daint:mc using cray [compile: 0.778s run: 11.878s total: 12.671s]
   [       OK ] ( 4/39) HelloThreadedExtended2Test on daint:mc using gnu [compile: 2.012s run: 19.736s total: 21.763s]
   [       OK ] ( 5/39) HelloMultiLangTest_c on daint:mc using cray [compile: 0.139s run: 68.148s total: 68.302s]
   [       OK ] ( 6/39) StreamWithRefTest on daint:login using gnu [compile: 4.640s run: 6.752s total: 11.859s]
   [       OK ] ( 7/39) HelloThreadedExtended2Test on daint:login using intel [compile: 2.248s run: 36.810s total: 39.072s]
   [       OK ] ( 8/39) HelloMultiLangTest_c on daint:login using cray [compile: 0.140s run: 82.924s total: 83.080s]
   [       OK ] ( 9/39) HelloThreadedExtended2Test on daint:gpu using pgi [compile: 2.556s run: 23.535s total: 26.108s]
   [       OK ] (10/39) HelloThreadedExtended2Test on daint:gpu using gnu [compile: 2.065s run: 29.922s total: 32.002s]
   [       OK ] (11/39) HelloMultiLangTest_cpp on daint:gpu using intel [compile: 1.976s run: 55.048s total: 57.039s]
   [       OK ] (12/39) HelloMultiLangTest_c on daint:gpu using intel [compile: 1.802s run: 78.871s total: 80.685s]
   [       OK ] (13/39) HelloThreadedExtended2Test on daint:mc using intel [compile: 2.286s run: 16.886s total: 19.186s]
   [       OK ] (14/39) HelloMultiLangTest_cpp on daint:mc using cray [compile: 0.462s run: 42.545s total: 43.022s]
   [       OK ] (15/39) HelloMultiLangTest_cpp on daint:mc using intel [compile: 1.976s run: 46.582s total: 48.572s]
   [       OK ] (16/39) HelloMultiLangTest_c on daint:mc using intel [compile: 1.736s run: 71.563s total: 73.316s]
   [       OK ] (17/39) HelloThreadedExtended2Test on daint:login using pgi [compile: 2.611s run: 33.608s total: 36.235s]
   [       OK ] (18/39) HelloThreadedExtended2Test on daint:login using gnu [compile: 2.020s run: 39.888s total: 42.523s]
   [       OK ] (19/39) HelloMultiLangTest_cpp on daint:login using intel [compile: 2.013s run: 63.789s total: 65.815s]
   [       OK ] (20/39) HelloMultiLangTest_c on daint:login using intel [compile: 1.809s run: 86.147s total: 87.970s]
   [       OK ] (21/39) HelloMultiLangTest_cpp on daint:gpu using pgi [compile: 1.938s run: 52.444s total: 54.397s]
   [       OK ] (22/39) HelloMultiLangTest_cpp on daint:gpu using gnu [compile: 1.788s run: 57.910s total: 59.714s]
   [       OK ] (23/39) HelloMultiLangTest_c on daint:gpu using pgi [compile: 1.512s run: 76.705s total: 78.234s]
   [       OK ] (24/39) HelloMultiLangTest_c on daint:gpu using gnu [compile: 1.543s run: 81.567s total: 83.138s]
   [       OK ] (25/39) HelloMultiLangTest_cpp on daint:mc using gnu [compile: 1.778s run: 49.430s total: 51.224s]
   [       OK ] (26/39) HelloMultiLangTest_c on daint:mc using pgi [compile: 1.564s run: 69.324s total: 70.910s]
   [       OK ] (27/39) HelloMultiLangTest_c on daint:mc using gnu [compile: 1.531s run: 74.238s total: 75.795s]
   [       OK ] (28/39) HelloMultiLangTest_cpp on daint:login using pgi [compile: 2.147s run: 60.996s total: 63.157s]
   [       OK ] (29/39) HelloMultiLangTest_cpp on daint:login using gnu [compile: 1.813s run: 66.590s total: 68.418s]
   [       OK ] (30/39) HelloMultiLangTest_c on daint:login using pgi [compile: 1.523s run: 84.018s total: 85.554s]
   [       OK ] (31/39) HelloMultiLangTest_c on daint:login using gnu [compile: 1.514s run: 88.777s total: 90.327s]
   [       OK ] (32/39) HelloMultiLangTest_cpp on daint:gpu using cray [compile: 0.451s run: 52.389s total: 52.865s]
   [       OK ] (33/39) HelloMultiLangTest_cpp on daint:mc using pgi [compile: 2.002s run: 44.985s total: 47.000s]
   [       OK ] (34/39) HelloMultiLangTest_cpp on daint:login using cray [compile: 0.474s run: 60.884s total: 61.375s]
   [       OK ] (35/39) HelloThreadedExtended2Test on daint:gpu using cray [compile: 0.871s run: 23.620s total: 24.510s]
   [       OK ] (36/39) HelloThreadedExtended2Test on daint:mc using pgi [compile: 2.535s run: 15.193s total: 17.744s]
   [       OK ] (37/39) HelloThreadedExtended2Test on daint:login using cray [compile: 0.773s run: 33.721s total: 34.508s]
   [       OK ] (38/39) StreamWithRefTest on daint:gpu using gnu [compile: 2.112s run: 6.108s total: 8.242s]
   [       OK ] (39/39) StreamWithRefTest on daint:mc using gnu [compile: 2.062s run: 6.873s total: 8.948s]
   [----------] all spawned checks have finished

   [  PASSED  ] Ran 39 test case(s) from 4 check(s) (0 failure(s))
   [==========] Finished on Mon Oct 12 20:14:10 2020
   ==============================================================================
   PERFORMANCE REPORT
   ------------------------------------------------------------------------------
   StreamWithRefTest
   - daint:login
      - gnu
         * num_tasks: 1
         * Copy: 73243.6 MB/s
         * Scale: 45575.6 MB/s
         * Add: 49194.6 MB/s
         * Triad: 49308.0 MB/s
   - daint:gpu
      - gnu
         * num_tasks: 1
         * Copy: 51542.8 MB/s
         * Scale: 35071.5 MB/s
         * Add: 38618.2 MB/s
         * Triad: 39003.5 MB/s
   - daint:mc
      - gnu
         * num_tasks: 1
         * Copy: 19129.6 MB/s
         * Scale: 10490.3 MB/s
         * Add: 11116.9 MB/s
         * Triad: 11065.5 MB/s
   ------------------------------------------------------------------------------
   Log file(s) saved in: '/tmp/rfm-6yaunqqp.log'


There it is!
Without any change in our tests, we could simply run them in a HPC cluster with all of its intricacies.
Notice how our original four tests expanded to almost 40 test cases on that particular HPC cluster!
One reason we could run immediately our tests on a new system was that we have not been restricting neither the valid system they can run nor the valid programming environments they can run with (except for the STREAM test).
Otherwise we would have to add ``daint`` and its corresponding programming environments in :attr:`valid_systems` and :attr:`valid_prog_environs` lists respectively.

.. tip::

   A quick way to try a test on a new system, if it's not generic, is to pass the :option:`--skip-system-check` and the :option:`--skip-prgenv-check` command line options which will cause ReFrame to skip any test validity checks for systems or programming environments.

Although the tests remain the same, ReFrame has generated completely different job scripts for each test depending on where it was going to run.
Let's check the job script generated for the ``StreamWithRefTest``:

.. code-block:: console

   cat output/daint/gpu/gnu/StreamWithRefTest/rfm_StreamWithRefTest_job.sh

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name="rfm_StreamWithRefTest_job"
   #SBATCH --ntasks=1
   #SBATCH --output=rfm_StreamWithRefTest_job.out
   #SBATCH --error=rfm_StreamWithRefTest_job.err
   #SBATCH --time=0:10:0
   #SBATCH -A csstaff
   #SBATCH --constraint=gpu
   module unload PrgEnv-cray
   module load PrgEnv-gnu
   export OMP_NUM_THREADS=4
   export OMP_PLACES=cores
   srun ./StreamWithRefTest

Whereas the exact same test running on our laptop was as simple as the following:

.. code-block:: bash

   #!/bin/bash
   export OMP_NUM_THREADS=4
   export OMP_PLACES=cores
    ./StreamWithRefTest

In ReFrame, you don't have to care about all the system interaction details, but rather about the logic of your tests as we shall see in the next section.


-----------------------------------------------------------
Adapting a test to new systems and programming environments
-----------------------------------------------------------

Unless a test is rather generic, you will need to make some adaptations for the system that you port it to.
In this case, we will adapt the STREAM benchmark so as to run it with multiple compiler and adjust its execution parameters based on the target architecture of each partition.
Let's see and comment the changes:

.. literalinclude:: ../tutorials/basics/stream/stream3.py
   :lines: 6-
   :emphasize-lines: 9,37-

First of all, we need to add the new programming environments in the list of the supported ones.
Now there is the problem that each compiler has its own flags for enabling OpenMP, so we need to differentiate the behavior of the test based on the programming environment.
For this reason, we define the flags for each compiler in a separate dictionary (``self.flags``) and we set them in the :func:`setflags` pipeline hook.
Let's explain what is this all about.
When ReFrame loads a test file, it instantiates all the tests it finds in it.
Based on the system ReFrame runs on and the supported environments of the tests, it will generate different test cases for each system partition and environment combination and it will finally send the test cases for execution.
During its execution, a test case goes through the *regression test pipeline*, which is a series of well defined phases.
Users can attach arbitrary functions to run before or after any pipeline stage and this is exactly what the :func:`setflags` function is.
We instruct ReFrame to run this function before the test enters the ``compile`` stage and set accordingly the compilation flags.
The system partition and the programming environment of the currently running test case are available to a ReFrame test through the :attr:`current_partition <reframe.core.pipeline.RegressionTest.current_partition>` and :attr:`current_environ <reframe.core.pipeline.RegressionTest.current_environ>` attributes respectively.
These attributes, however, are only set after the first stage (``setup``) of the pipeline is executed, so we can't use them inside the test's constructor.

We do exactly the same for setting the ``OMP_NUM_THREADS`` environment variables depending on the system partition we are running on, by attaching the :func:`set_num_threads` pipeline hook to the ``run`` phase of the test.
In that same hook we also set the :attr:`num_cpus_per_task <reframe.core.pipeline.RegressionTest.num_cpus_per_task>` attribute of the test, so as to instruct the backend job scheduler to properly assign CPU cores to the test.
In ReFrame tests you can set a series of task allocation attributes that will be used by the backend schedulers to emit the right job submission script.
The section :ref:`scheduler_options` of the :doc:`regression_test_api` summarizes these attributes and the actual backend scheduler options that they correspond to.

For more information about the regression test pipeline and how ReFrame executes the tests in general, have a look at :doc:`pipeline`.

.. note::

   ReFrame tests are ordinary Python classes so you can define your own attributes as we do with :attr:`flags` and :attr:`cores` in this example.

Let's run our adapted test now:

.. code-block:: console

   ./bin/reframe -C tutorials/config/settings.py -c tutorials/basics/stream/stream3.py -r --performance-report


.. code-block:: none

   [ReFrame Setup]
     version:           3.3-dev0 (rev: cb974c13)
     command:           './bin/reframe -C tutorials/config/settings.py -c tutorials/basics/stream/stream3.py -r --performance-report'
     launched by:       user@dom101
     working directory: '/users/user/Devel/reframe'
     settings file:     'tutorials/config/settings.py'
     check search path: '/users/user/Devel/reframe/tutorials/basics/stream/stream3.py'
     stage directory:   '/users/user/Devel/reframe/stage'
     output directory:  '/users/user/Devel/reframe/output'

   [==========] Running 1 check(s)
   [==========] Started on Mon Oct 12 20:16:03 2020

   [----------] started processing StreamMultiSysTest (StreamMultiSysTest)
   [ RUN      ] StreamMultiSysTest on daint:login using gnu
   [ RUN      ] StreamMultiSysTest on daint:login using intel
   [ RUN      ] StreamMultiSysTest on daint:login using pgi
   [ RUN      ] StreamMultiSysTest on daint:login using cray
   [ RUN      ] StreamMultiSysTest on daint:gpu using gnu
   [ RUN      ] StreamMultiSysTest on daint:gpu using intel
   [ RUN      ] StreamMultiSysTest on daint:gpu using pgi
   [ RUN      ] StreamMultiSysTest on daint:gpu using cray
   [ RUN      ] StreamMultiSysTest on daint:mc using gnu
   [ RUN      ] StreamMultiSysTest on daint:mc using intel
   [ RUN      ] StreamMultiSysTest on daint:mc using pgi
   [ RUN      ] StreamMultiSysTest on daint:mc using cray
   [----------] finished processing StreamMultiSysTest (StreamMultiSysTest)

   [----------] waiting for spawned checks to finish
   [       OK ] ( 1/12) StreamMultiSysTest on daint:gpu using pgi [compile: 2.092s run: 11.201s total: 13.307s]
   [       OK ] ( 2/12) StreamMultiSysTest on daint:gpu using gnu [compile: 2.349s run: 17.140s total: 19.509s]
   [       OK ] ( 3/12) StreamMultiSysTest on daint:login using pgi [compile: 2.230s run: 20.946s total: 23.189s]
   [       OK ] ( 4/12) StreamMultiSysTest on daint:login using gnu [compile: 2.161s run: 27.093s total: 29.266s]
   [       OK ] ( 5/12) StreamMultiSysTest on daint:mc using gnu [compile: 1.954s run: 7.904s total: 9.870s]
   [       OK ] ( 6/12) StreamMultiSysTest on daint:gpu using intel [compile: 2.286s run: 14.686s total: 16.984s]
   [       OK ] ( 7/12) StreamMultiSysTest on daint:login using intel [compile: 2.520s run: 24.427s total: 26.960s]
   [       OK ] ( 8/12) StreamMultiSysTest on daint:mc using intel [compile: 2.312s run: 5.350s total: 7.678s]
   [       OK ] ( 9/12) StreamMultiSysTest on daint:gpu using cray [compile: 0.672s run: 10.791s total: 11.476s]
   [       OK ] (10/12) StreamMultiSysTest on daint:login using cray [compile: 0.706s run: 20.505s total: 21.229s]
   [       OK ] (11/12) StreamMultiSysTest on daint:mc using cray [compile: 0.674s run: 2.763s total: 3.453s]
   [       OK ] (12/12) StreamMultiSysTest on daint:mc using pgi [compile: 2.088s run: 5.124s total: 7.224s]
   [----------] all spawned checks have finished

   [  PASSED  ] Ran 12 test case(s) from 1 check(s) (0 failure(s))
   [==========] Finished on Mon Oct 12 20:16:36 2020
   ==============================================================================
   PERFORMANCE REPORT
   ------------------------------------------------------------------------------
   StreamMultiSysTest
   - daint:login
      - gnu
         * num_tasks: 1
         * Copy: 95784.6 MB/s
         * Scale: 73747.3 MB/s
         * Add: 79138.3 MB/s
         * Triad: 81253.3 MB/s
      - intel
         * num_tasks: 1
         * Copy: 103540.5 MB/s
         * Scale: 109257.6 MB/s
         * Add: 112189.8 MB/s
         * Triad: 113440.8 MB/s
      - pgi
         * num_tasks: 1
         * Copy: 99071.7 MB/s
         * Scale: 74721.3 MB/s
         * Add: 81206.4 MB/s
         * Triad: 78328.9 MB/s
      - cray
         * num_tasks: 1
         * Copy: 96664.5 MB/s
         * Scale: 75637.4 MB/s
         * Add: 74759.3 MB/s
         * Triad: 73450.6 MB/s
   - daint:gpu
      - gnu
         * num_tasks: 1
         * Copy: 42293.7 MB/s
         * Scale: 38095.1 MB/s
         * Add: 43080.7 MB/s
         * Triad: 43719.2 MB/s
      - intel
         * num_tasks: 1
         * Copy: 52563.0 MB/s
         * Scale: 54316.5 MB/s
         * Add: 59044.5 MB/s
         * Triad: 59165.5 MB/s
      - pgi
         * num_tasks: 1
         * Copy: 50710.5 MB/s
         * Scale: 39639.5 MB/s
         * Add: 44104.5 MB/s
         * Triad: 44143.7 MB/s
      - cray
         * num_tasks: 1
         * Copy: 51159.8 MB/s
         * Scale: 39176.0 MB/s
         * Add: 43588.8 MB/s
         * Triad: 43866.8 MB/s
   - daint:mc
      - gnu
         * num_tasks: 1
         * Copy: 48744.5 MB/s
         * Scale: 38774.7 MB/s
         * Add: 43760.0 MB/s
         * Triad: 44143.1 MB/s
      - intel
         * num_tasks: 1
         * Copy: 52707.0 MB/s
         * Scale: 49011.8 MB/s
         * Add: 57513.3 MB/s
         * Triad: 57678.3 MB/s
      - pgi
         * num_tasks: 1
         * Copy: 46274.3 MB/s
         * Scale: 40628.6 MB/s
         * Add: 44352.4 MB/s
         * Triad: 44630.2 MB/s
      - cray
         * num_tasks: 1
         * Copy: 46912.5 MB/s
         * Scale: 40076.9 MB/s
         * Add: 43639.0 MB/s
         * Triad: 44068.3 MB/s
   ------------------------------------------------------------------------------
   Log file(s) saved in: '/tmp/rfm-odx7qewe.log'


Notice the improved performance of the benchmark in all partitions and the differences in performance between the different compilers.

This concludes our introductory tutorial to ReFrame!
