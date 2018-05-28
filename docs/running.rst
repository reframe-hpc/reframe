===============
Running ReFrame
===============

Before getting into any details, the simplest way to invoke ReFrame is the following:

.. code-block:: bash

  ./bin/reframe -c /path/to/checks -R --run

This will search recursively for test files in ``/path/to/checks`` and will start running them on the current system.

ReFrame's front-end goes through three phases:

1. Load tests
2. Filter tests
3. Act on tests

In the following, we will elaborate on these phases and the key command-line options controlling them.
A detailed listing of all the command-line options grouped by phase is given by ``./bin/reframe -h``.

Supported Actions
-----------------

Even though an action is the last phase that the front-end goes through, we are listing it first since an action is always required.
Currently there are only two available actions:

1. Listing of the selected checks
2. Execution of the selected checks

Listing of the regression tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To retrieve a listing of the selected checks, you must specify the ``-l`` or ``--list`` options.
An example listing of checks is the following that lists all the tests found under the ``tutorial/`` folder:

.. code-block:: bash

  ./bin/reframe -c tutorial -l

The output looks like:

.. code-block:: none

   Command line: ./bin/reframe -c tutorial/ -l
   Reframe version: 2.13-dev0
   Launched by user: USER
   Launched on host: daint103
   Reframe paths
   =============
       Check prefix      :
       Check search path : 'tutorial/'
       Stage dir prefix  : /path/to/reframe/stage/
       Output dir prefix : /path/to/reframe/output/
       Logging dir       : /path/to/reframe/logs
   List of matched checks
   ======================
     * Example5Test (found in /path/to/reframe/tutorial/example5.py)
           descr: Matrix-vector multiplication example with CUDA
           tags: {'tutorial'}, maintainers: ['you-can-type-your-email-here']
     * Example1Test (found in /path/to/reframe/tutorial/example1.py)
           descr: Simple matrix-vector multiplication example
           tags: {'tutorial'}, maintainers: ['you-can-type-your-email-here']
     * Example4Test (found in /path/to/reframe/tutorial/example4.py)
           descr: Matrix-vector multiplication example with OpenACC
           tags: {'tutorial'}, maintainers: ['you-can-type-your-email-here']
     * SerialTest (found in /path/to/reframe/tutorial/example8.py)
           descr: Serial matrix-vector multiplication
           tags: {'tutorial'}, maintainers: ['you-can-type-your-email-here']
     * OpenMPTest (found in /path/to/reframe/tutorial/example8.py)
           descr: OpenMP matrix-vector multiplication
           tags: {'tutorial'}, maintainers: ['you-can-type-your-email-here']
     * MPITest (found in /path/to/reframe/tutorial/example8.py)
           descr: MPI matrix-vector multiplication
           tags: {'tutorial'}, maintainers: ['you-can-type-your-email-here']
     * OpenACCTest (found in /path/to/reframe/tutorial/example8.py)
           descr: OpenACC matrix-vector multiplication
           tags: {'tutorial'}, maintainers: ['you-can-type-your-email-here']
     * CudaTest (found in /path/to/reframe/tutorial/example8.py)
           descr: CUDA matrix-vector multiplication
           tags: {'tutorial'}, maintainers: ['you-can-type-your-email-here']
     * Example3Test (found in /path/to/reframe/tutorial/example3.py)
           descr: Matrix-vector multiplication example with MPI
           tags: {'tutorial'}, maintainers: ['you-can-type-your-email-here']
     * Example7Test (found in /path/to/reframe/tutorial/example7.py)
           descr: Matrix-vector multiplication (CUDA performance test)
           tags: {'tutorial'}, maintainers: ['you-can-type-your-email-here']
     * Example6Test (found in /path/to/reframe/tutorial/example6.py)
           descr: Matrix-vector multiplication with L2 norm check
           tags: {'tutorial'}, maintainers: ['you-can-type-your-email-here']
     * Example2aTest (found in /path/to/reframe/tutorial/example2.py)
           descr: Matrix-vector multiplication example with OpenMP
           tags: {'tutorial'}, maintainers: ['you-can-type-your-email-here']
     * Example2bTest (found in /path/to/reframe/tutorial/example2.py)
           descr: Matrix-vector multiplication example with OpenMP
           tags: {'tutorial'}, maintainers: ['you-can-type-your-email-here']
   Found 13 check(s).


The listing contains the name of the check, its description, the tags associated with it and a list of its maintainers.
Note that this listing may also contain checks that are not supported by the current system.
These checks will be just skipped if you try to run them.

Execution of the regression tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run the regression tests you should specify the *run* action though the ``-r`` or ``--run`` options.

.. note:: The listing action takes precedence over the execution, meaning that if you specify both ``-l -r``, only the listing action will be performed.


.. code-block:: bash

  ./reframe.py -C tutorial/config/settings.py -c tutorial/example1.py -r

The output of the regression run looks like the following:

.. code-block:: none

  Command line: ./reframe.py -C tutorial/config/settings.py -c tutorial/example1.py -r
  Reframe version: 2.13-dev0
  Launched by user: USER
  Launched on host: daint103
  Reframe paths
  =============
      Check prefix      :
      Check search path : 'tutorial/example1.py'
      Stage dir prefix  : /path/to/reframe/stage/
      Output dir prefix : /path/to/reframe/output/
      Logging dir       : /path/to/reframe/logs
  [==========] Running 1 check(s)
  [==========] Started on Sat May 26 00:34:34 2018

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
  [==========] Finished on Sat May 26 00:35:39 2018


Discovery of Regression Tests
-----------------------------

When ReFrame is invoked, it tries to locate regression tests in a predefined path.
By default, this path is the ``<reframe-install-dir>/checks``.
You can also retrieve this path as follows:

.. code-block:: bash

  ./bin/reframe -l | grep 'Check search path'

If the path line is prefixed with ``(R)``, every directory in that path will be searched recursively for regression tests.

As described extensively in the `"ReFrame Tutorial" <tutorial.html>`__, regression tests in ReFrame are essentially Python source files that provide a special function, which returns the actual regression test instances.
A single source file may also provide multiple regression tests.
ReFrame loads the python source files and tries to call this special function;
if this function cannot be found, the source file will be ignored.
At the end of this phase, the front-end will have instantiated all the tests found in the path.

You can override the default search path for tests by specifying the ``-c`` or ``--checkpath`` options.
We have already done that already when listing all the tutorial tests:

.. code-block:: bash

  ./bin/reframe -c tutorial/ -l

ReFrame the does not search recursively into directories specified with the ``-c`` option, unless you explicitly specify the ``-R`` or ``--recurse`` options.

The ``-c`` option completely overrides the default path.
Currently, there is no option to prepend or append to the default regression path.
However, you can build your own check path by specifying multiple times the ``-c`` option.
The ``-c``\ option accepts also regular files. This is very useful when you are implementing new regression tests, since it allows you to run only your test:

.. code-block:: bash

  ./bin/reframe -c /path/to/my/new/test.py -r

.. important::
   The names of the loaded tests must be unique.
   Trying to load two or more tests with the same name will produce an error.
   You may ignore the error by using the ``--ignore-check-conflicts`` option.
   In this case, any conflicting test will not be loaded and a warning will be issued.

   .. versionadded:: 2.12


Filtering of Regression Tests
-----------------------------

At this phase you can select which regression tests should be run or listed.
There are several ways to select regression tests, which we describe in more detail here:

Selecting tests by programming environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To select tests by the programming environment, use the ``-p`` or ``--prgenv`` options:

.. code-block:: bash

  ./bin/reframe -p PrgEnv-gnu -l

This will select all the checks that support the ``PrgEnv-gnu`` environment.

You can also specify multiple times the ``-p`` option, in which case a test will be selected if it support all the programming environments specified in the command line.
For example the following will select all the checks that can run with both ``PrgEnv-cray`` and ``PrgEnv-gnu``:

.. code-block:: bash

  ./bin/reframe -p PrgEnv-gnu -p PrgEnv-cray -l

If you are going to run a set of tests selected by programming environment, they will run only for the selected programming environment(s).

Selecting tests by tags
^^^^^^^^^^^^^^^^^^^^^^^

As we have seen in the `"ReFrame tutorial" <tutorial.html>`__, every regression test may be associated with a set of tags. Using the ``-t`` or ``--tag`` option you can select the regression tests associated with a specific tag.
For example the following will list all the tests that have a ``maintenance`` tag:

.. code-block:: bash

  ./bin/reframe -t maintenance -l

Similarly to the ``-p`` option, you can chain multiple ``-t`` options together, in which case a regression test will be selected if it is associated with all the tags specified in the command line.
The list of tags associated with a check can be viewed in the listing output when specifying the ``-l`` option.

Selecting tests by name
^^^^^^^^^^^^^^^^^^^^^^^

It is possible to select or exclude tests by name through the ``--name`` or ``-n`` and ``--exclude`` or ``-x`` options.
For example, you can select only the ``Example7Test`` from the tutorial as follows:

.. code-block:: bash

  ./bin/reframe -c tutorial/ -n Example7Test -l

.. code-block:: none

  Command line: ./bin/reframe -c tutorial/ -n Example7Test -l
  Reframe version: 2.13-dev0
  Launched by user: USER
  Launched on host: daint103
  Reframe paths
  =============
      Check prefix      :
      Check search path : 'tutorial'
      Stage dir prefix  : /path/to/reframe/stage/
      Output dir prefix : /path/to/reframe/output/
      Logging dir       : /path/to/reframe/logs
  List of matched checks
  ======================
    * Example7Test (found in /path/to/reframe/tutorial/example7.py)
          descr: Matrix-vector multiplication (CUDA performance test)
          tags: {'tutorial'}, maintainers: ['you-can-type-your-email-here']
  Found 1 check(s).


Similarly, you can exclude this test by passing the ``-x Example7Test`` option:

.. code-block:: none

  Command line: ./bin/reframe -c tutorial -x Example7Test -l
  Reframe version: 2.13-dev0
  Launched by user: USER
  Launched on host: daint103
  Reframe paths
  =============
      Check prefix      :
      Check search path : 'tutorial'
      Stage dir prefix  : /path/to/reframe/stage/
      Output dir prefix : /path/to/reframe/output/
      Logging dir       : /path/to/reframe/logs
  List of matched checks
  ======================
    * Example5Test (found in /path/to/reframe/tutorial/example5.py)
          descr: Matrix-vector multiplication example with CUDA
          tags: {'tutorial'}, maintainers: ['you-can-type-your-email-here']
    * Example1Test (found in /path/to/reframe/tutorial/example1.py)
          descr: Simple matrix-vector multiplication example
          tags: {'tutorial'}, maintainers: ['you-can-type-your-email-here']
    * Example4Test (found in /path/to/reframe/tutorial/example4.py)
          descr: Matrix-vector multiplication example with OpenACC
          tags: {'tutorial'}, maintainers: ['you-can-type-your-email-here']
    * SerialTest (found in /path/to/reframe/tutorial/example8.py)
          descr: Serial matrix-vector multiplication
          tags: {'tutorial'}, maintainers: ['you-can-type-your-email-here']
    * OpenMPTest (found in /path/to/reframe/tutorial/example8.py)
          descr: OpenMP matrix-vector multiplication
          tags: {'tutorial'}, maintainers: ['you-can-type-your-email-here']
    * MPITest (found in /path/to/reframe/tutorial/example8.py)
          descr: MPI matrix-vector multiplication
          tags: {'tutorial'}, maintainers: ['you-can-type-your-email-here']
    * OpenACCTest (found in /path/to/reframe/tutorial/example8.py)
          descr: OpenACC matrix-vector multiplication
          tags: {'tutorial'}, maintainers: ['you-can-type-your-email-here']
    * CudaTest (found in /path/to/reframe/tutorial/example8.py)
          descr: CUDA matrix-vector multiplication
          tags: {'tutorial'}, maintainers: ['you-can-type-your-email-here']
    * Example3Test (found in /path/to/reframe/tutorial/example3.py)
          descr: Matrix-vector multiplication example with MPI
          tags: {'tutorial'}, maintainers: ['you-can-type-your-email-here']
    * Example6Test (found in /path/to/reframe/tutorial/example6.py)
          descr: Matrix-vector multiplication with L2 norm check
          tags: {'tutorial'}, maintainers: ['you-can-type-your-email-here']
    * Example2aTest (found in /path/to/reframe/tutorial/example2.py)
          descr: Matrix-vector multiplication example with OpenMP
          tags: {'tutorial'}, maintainers: ['you-can-type-your-email-here']
    * Example2bTest (found in /path/to/reframe/tutorial/example2.py)
          descr: Matrix-vector multiplication example with OpenMP
          tags: {'tutorial'}, maintainers: ['you-can-type-your-email-here']
  Found 12 check(s).


Controlling the Execution of Regression Tests
---------------------------------------------

There are several options for controlling the execution of regression tests.
Keep in mind that these options will affect all the tests that will run with the current invocation.
They are summarized below:

* ``-A ACCOUNT``, ``--account ACCOUNT``: Submit regression test jobs using ``ACCOUNT``.
* ``-P PART``, ``--partition PART``: Submit regression test jobs in the *scheduler partition* ``PART``.
* ``--reservation RES``: Submit regression test jobs in reservation ``RES``.
* ``--nodelist NODELIST``: Run regression test jobs on the nodes specified in ``NODELIST``.
* ``--exclude-nodes NODELIST``: Do not run the regression test jobs on any of the nodes specified in ``NODELIST``.
* ``--job-option OPT``: Pass option ``OPT`` directly to the back-end job scheduler. This option *must* be used with care, since you may break the submission mechanism.
  All of the above job submission related options could be expressed with this option.
  For example, the ``-n NODELIST`` is equivalent to ``--job-option='--nodelist=NODELIST'`` for a Slurm job scheduler.
  If you pass an option that is already defined by the framework, the framework will *not* explicitly override it; this is up to scheduler.
  All extra options defined from the command line are appended to the automatically generated options in the generated batch script file.
  So if you redefine one of them, e.g., ``--output`` for the Slurm scheduler, it is up the job scheduler on how to interpret multiple definitions of the same options.
  In this example, Slurm's policy is that later definitions of options override previous ones.
  So, in this case, way you would override the standard output for all the submitted jobs!

* ``--force-local``: Force the local execution of the selected tests.
  No jobs will be submitted.
* ``--skip-sanity-check``: Skip sanity checking phase.
* ``--skip-performance-check``: Skip performance verification phase.
* ``--strict``: Force strict performance checking. Some tests may set their :attr:`strict_check <reframe.core.pipeline.RegressionTest.strick_check>` attribute to :class:`False` (see `"Reference Guide" <reference.html>`__) in order to just let their performance recorded but not yield an error.
  This option overrides this behavior and forces all tests to be strict.
* ``--skip-system-check``: Skips the system check and run the selected tests even if they do not support the current system.
  This option is sometimes useful when you need to quickly verify if a regression test supports a new system.
* ``--skip-prgenv-check``: Skips programming environment check and run the selected tests for even if they do not support a programming environment.
  This option is useful when you need to quickly verify if a regression check supports another programming environment.
  For example, if you know that a tests supports only ``PrgEnv-cray`` and you need to check if it also works with ``PrgEnv-gnu``, you can test is as follows:

  .. code-block:: bash

    ./bin/reframe -c /path/to/my/check.py -p PrgEnv-gnu --skip-prgenv-check -r

* ``--max-retries NUM``: Specify the maximum number of times a failed regression test may be retried (default: 0).

Configuring ReFrame Directories
-------------------------------

ReFrame uses three basic directories during the execution of tests:

1. The stage directory

  * Each regression test is executed in a "sandbox";
    all of its resources (source files, input data etc.) are copied over to a stage directory (if the directory preexists, it will be wiped out) and executed from there.
    This will also be the working directory for the test.

2. The output directory

  * After a regression test finishes some important files will be copied from the stage directory to the output directory (if the directory preexists, it will be wiped out).
    By default these are the standard output, standard error and the generated job script file.
    A regression test may also specify to keep additional files.

3. The log directory

  * This is where the performance log files of the individual performance tests are placed (see `Logging <#logging>`__ for more information)

By default, all these directories are placed under a common prefix, which defaults to ``.``.
The rest of the directories are organized as follows:

* Stage directory: ``${prefix}/stage/<timestamp>``
* Output directory: ``${prefix}/output/<timestamp>``
* Performance log directory: ``${prefix}/logs``

You can optionally append a timestamp directory component to the above paths (except the logs directory), by using the ``--timestamp`` option.
This options takes an optional argument to specify the timestamp format.
The default `time format <http://man7.org/linux/man-pages/man3/strftime.3.html>`__ is ``%FT%T``, which results into timestamps of the form ``2017-10-24T21:10:29``.

You can override either the default global prefix or any of the default individual directories using the corresponding options.

* ``--prefix DIR``: set prefix to ``DIR``.
* ``--output DIR``: set output directory to ``DIR``.
* ``--stage DIR``: set stage directory to ``DIR``.
* ``--logdir DIR``: set performance log directory to ``DIR``.

The stage and output directories are created only when you run a regression test.
However you can view the directories that will be created even when you do a listing of the available checks with the ``-l`` option.
This is useful if you want to check the directories that ReFrame will create.

.. code-block:: bash

  ./bin/reframe -C tutorial/config/settings.py --prefix /foo -l

.. code-block:: none

  Command line: ./bin/reframe -C tutorial/config/settings.py --prefix /foo -l
  Reframe version: 2.13-dev0
  Launched by user: USER
  Launched on host: daint103
  Reframe paths
  =============
      Check prefix      : /path/to/reframe
  (R) Check search path : 'checks/'
      Stage dir prefix  : /foo/stage/
      Output dir prefix : /foo/output/
      Logging dir       : /foo/logs
  List of matched checks
  ======================
  Found 0 check(s).


You can also define different default directories per system by specifying them in the `site configuration <configure.html#the-configuration-file>`__ settings file.
The command line options, though, take always precedence over any default directory.

Logging
-------

From version 2.4 onward, ReFrame supports logging of its actions.
ReFrame creates two files inside the current working directory every time it is run:

* ``reframe.out``: This file stores the output of a run as it was printed in the standard output.
* ``reframe.log``: This file stores more detailed of information on ReFrame's actions.

By default, the output in ``reframe.log`` looks like the following:

.. code-block:: none

  2018-05-26T00:30:39] info: reframe: [ RUN      ] Example7Test on daint:gpu using PrgEnv-cray
  [2018-05-26T00:30:39] debug: Example7Test: entering stage: setup
  [2018-05-26T00:30:39] debug: Example7Test: loading environment for the current partition
  [2018-05-26T00:30:39] debug: Example7Test: executing OS command: modulecmd python show daint-gpu
  [2018-05-26T00:30:39] debug: Example7Test: executing OS command: modulecmd python load daint-gpu
  [2018-05-26T00:30:39] debug: Example7Test: loading test's environment
  [2018-05-26T00:30:39] debug: Example7Test: executing OS command: modulecmd python show PrgEnv-cray
  [2018-05-26T00:30:39] debug: Example7Test: executing OS command: modulecmd python unload PrgEnv-gnu
  [2018-05-26T00:30:39] debug: Example7Test: executing OS command: modulecmd python load PrgEnv-cray
  [2018-05-26T00:30:39] debug: Example7Test: executing OS command: modulecmd python show cudatoolkit
  [2018-05-26T00:30:39] debug: Example7Test: executing OS command: modulecmd python load cudatoolkit
  [2018-05-26T00:30:39] debug: Example7Test: setting up paths
  [2018-05-26T00:30:40] debug: Example7Test: setting up the job descriptor
  [2018-05-26T00:30:40] debug: Example7Test: job scheduler backend: local
  [2018-05-26T00:30:40] debug: Example7Test: setting up performance logging
  [2018-05-26T00:30:40] debug: Example7Test: entering stage: compile
  [2018-05-26T00:30:40] debug: Example7Test: copying /path/to/reframe/tutorial/src to stage directory (/path/to/reframe/stage/gpu/Example7Test/PrgEnv-cray)
  [2018-05-26T00:30:40] debug: Example7Test: symlinking files: []
  [2018-05-26T00:30:40] debug: Example7Test: Staged sourcepath: /path/to/reframe/stage/gpu/Example7Test/PrgEnv-cray/example_matrix_vector_multiplication_cuda.cu
  [2018-05-26T00:30:40] debug: Example7Test: executing OS command: nvcc  -O3 -I/path/to/reframe/stage/gpu/Example7Test/PrgEnv-cray /path/to/reframe/stage/gpu/Example7Test/PrgEnv-cray/e
  xample_matrix_vector_multiplication_cuda.cu -o /path/to/reframe/stage/gpu/Example7Test/PrgEnv-cray/./Example7Test
  [2018-05-26T00:30:40] debug: Example7Test: compilation stdout:

  [2018-05-26T00:30:40] debug: Example7Test: compilation stderr:
  nvcc warning : The 'compute_20', 'sm_20', and 'sm_21' architectures are deprecated, and may be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).

  [2018-05-26T00:30:40] debug: Example7Test: compilation finished
  [2018-05-26T00:30:40] debug: Example7Test: entering stage: run
  [2018-05-26T00:30:40] debug: Example7Test: executing OS command: sbatch /path/to/reframe/stage/gpu/Example7Test/PrgEnv-cray/Example7Test_daint_gpu_PrgEnv-cray.sh
  [2018-05-26T00:30:40] debug: Example7Test: spawned job (jobid=746641)
  [2018-05-26T00:30:40] debug: Example7Test: entering stage: wait
  [2018-05-26T00:30:40] debug: Example7Test: executing OS command: sacct -S 2018-05-26 -P -j 746641 -o jobid,state,exitcode
  [2018-05-26T00:30:40] debug: Example7Test: job state not matched (stdout follows)
  JobID|State|ExitCode

  [2018-05-26T00:30:41] debug: Example7Test: executing OS command: sacct -S 2018-05-26 -P -j 746641 -o jobid,state,exitcode
  [2018-05-26T00:30:44] debug: Example7Test: executing OS command: sacct -S 2018-05-26 -P -j 746641 -o jobid,state,exitcode
  [2018-05-26T00:30:47] debug: Example7Test: executing OS command: sacct -S 2018-05-26 -P -j 746641 -o jobid,state,exitcode
  [2018-05-26T00:30:47] debug: Example7Test: spawned job finished
  [2018-05-26T00:30:47] debug: Example7Test: entering stage: sanity
  [2018-05-26T00:30:47] debug: Example7Test: entering stage: performance
  [2018-05-26T00:30:47] debug: Example7Test: entering stage: cleanup
  [2018-05-26T00:30:47] debug: Example7Test: copying interesting files to output directory
  [2018-05-26T00:30:47] debug: Example7Test: removing stage directory
  [2018-05-26T00:30:47] info: reframe: [       OK ] Example7Test on daint:gpu using PrgEnv-cray


Each line starts with a timestamp, the level of the message (``info``, ``debug`` etc.), the context in which the framework is currently executing (either ``reframe`` or the name of the current test and, finally, the actual message.

Every time ReFrame is run, both ``reframe.out`` and ``reframe.log`` files will be rewritten.
However, you can ask ReFrame to copy them to the output directory before exiting by passing it the ``--save-log-files`` option.

Configuring logging
^^^^^^^^^^^^^^^^^^^

You can configure several aspects of logging in ReFrame and even how the output will look like.
ReFrame's logging mechanism is built upon Python's `logging <https://docs.python.org/3.6/library/logging.html>`__ framework adding extra logging levels and more formatting capabilities.

Logging in ReFrame is configured by the ``_logging_config`` variable in the ``reframe/settings.py`` file.
The default configuration looks as follows:

.. code-block:: python

  _logging_config = {
      'level': 'DEBUG',
      'handlers': {
          'reframe.log' : {
              'level'     : 'DEBUG',
              'format'    : '[%(asctime)s] %(levelname)s: '
                            '%(check_info)s: %(message)s',
              'append'    : False,
          },

          # Output handling
          '&1': {
              'level'     : 'INFO',
              'format'    : '%(message)s'
          },
          'reframe.out' : {
              'level'     : 'INFO',
              'format'    : '%(message)s',
              'append'    : False,
          }
      }
  }

Note that this configuration dictionary is not the same as the one used by Python's logging framework.
It is a simplified version adapted to the needs of ReFrame.

The ``_logging_config`` dictionary has two main key entries:

* ``level`` (default: ``'INFO'``): This is the lowest level of messages that will be passed down to the different log record handlers.
  Any message with a lower level than that, it will be filtered out immediately and will not be passed to any handler.
  ReFrame defines the following logging levels with a decreasing severity: ``CRITICAL``, ``ERROR``, ``WARNING``, ``INFO``, ``VERBOSE`` and ``DEBUG``.
  Note that the level name is *not* case sensitive in ReFrame.
* ``handlers``: A dictionary defining the properties of the handlers that are attached to ReFrame's logging mechanism.
  The key is either a filename or a special character combination denoting standard output (``&1``) or standard error (``&2``).
  You can attach as many handlers as you like.
  The value of each handler key is another dictionary that holds the properties of the corresponding handler as key/value pairs.

The configurable properties of a log record handler are the following:

* ``level`` (default: ``'debug'``): The lowest level of log records that this handler can process.
* ``format`` (default: ``'%(message)s'``): Format string for the printout of the log record.
  ReFrame supports all the `format strings <https://docs.python.org/3.6/library/logging.html#logrecord-attributes>`__ from Python's logging library and provides the following additional ones:

  * ``check_name``: Prints the name of the regression test on behalf of which ReFrame is currently executing.
    If ReFrame is not in the context of regression test, ``reframe`` will be printed.
  * ``check_jobid``: Prints the job or process id of the job or process associated with currently executing regression test.
    If a job or process is not yet created, ``-1`` will be printed.
  * ``check_info``: Print live information of the currently executing check.
    By default this field has the form ``<check_name> on <current_partition> using <current_environment>``.
    It can be configured on a per test basis by overriding the :func:`info <reframe.core.pipeline.RegressionTest.info>` method in your regression test.

* ``datefmt`` (default: ``'%FT%T'``) The format that will be used for outputting timestamps (i.e., the ``%(asctime)s`` field).
  Acceptable formats must conform to standard library's `time.strftime() <https://docs.python.org/3.6/library/time.html#time.strftime>`__ function.
* ``append`` (default: :class:`False`) Controls whether ReFrame should append to this file or not.
  This is ignored for the standard output/error handlers.
* ``timestamp`` (default: :class:`None`): Append a timestamp to this log filename.
  This property may accept any date format as the ``datefmt`` property.
  If set for a ``filename.log`` handler entry, the resulting log file name will be ``filename_<timestamp>.log``.
  This property is ignored for the standard output/error handlers.

.. caution::
      The ``testcase_name`` logging attribute was replaced with the ``check_info``, which is now also configurable

   .. versionchanged:: 2.10


Performance Logging
^^^^^^^^^^^^^^^^^^^

ReFrame supports additional logging for performance tests specifically, in order to record historical performance data.
For each performance test, a log file of the form ``<test-name>.log`` is created under the ReFrame's `log directory <#configuring-reframe-directories>`__ where the test's performance is recorded.
The default format used for this file is ``'[%(asctime)s] %(check_info)s (jobid=%(check_jobid)s): %(message)s'`` and ReFrame always appends to this file.
Currently, it is not possible for users to configure performance logging.

The resulting log file looks like the following:

.. code-block:: none

  [2018-05-26T00:30:47] reframe 2.13-dev0: Example7Test on daint:gpu using PrgEnv-cray (jobid=746641): value: 49.246694, reference: (50.0, -0.1, 0.1)
  [2018-05-26T00:30:54] reframe 2.13-dev0: Example7Test on daint:gpu using PrgEnv-gnu (jobid=746642): value: 48.781683, reference: (50.0, -0.1, 0.1)
  [2018-05-26T00:31:02] reframe 2.13-dev0: Example7Test on daint:gpu using PrgEnv-pgi (jobid=746643): value: 49.139091, reference: (50.0, -0.1, 0.1)


The interpretation of the performance values depends on the individual tests.
The above output is from the CUDA performance test we presented in the `tutorial <tutorial.html#writing-a-performance-test>`__, so the value refers to the achieved Gflop/s.
The reference value is a three-element tuple of the form ``(<reference>, <lower-threshold>, <upper-threshold>)``, where the ``lower-threshold`` and ``upper-threshold`` are the acceptable tolerance thresholds expressed in percentages.
For example, the performance check shown above has a reference value of 50 Gflop/s ± 10%.

Asynchronous Execution of Regression Checks
-------------------------------------------

From version `2.4 <https://github.com/eth-cscs/reframe/releases/tag/v2.4>`__, ReFrame supports asynchronous execution of regression tests.
This execution policy can be enabled by passing the option ``--exec-policy=async`` to the command line.
The default execution policy is ``serial`` which enforces a sequential execution of the selected regression tests.
The asynchronous execution policy parallelizes only the `running phase <pipeline.html#the-run-phase>`__ of the tests.
The rest of the phases remain sequential.

A limit of concurrent jobs (pending and running) may be `configured <configure.html#partition-configuration>`__ for each virtual system partition.
As soon as the concurrency limit of a partition is reached, ReFrame will hold the execution of new regression tests until a slot is released in that partition.

When executing in asynchronous mode, ReFrame's output differs from the sequential execution.
The final result of the tests will be printed at the end and additional messages may be printed to indicate that a test is held.
Here is an example output of ReFrame using asynchronous execution policy:

.. code-block:: none

  Command line: ./bin/reframe -C tutorial/config/settings.py -c tutorial/ --exec-policy=async -r
  Reframe version: 2.13-dev0
  Launched by user: USER
  Launched on host: daint103
  Reframe paths
  =============
      Check prefix      :
      Check search path : 'tutorial/'
      Stage dir prefix  : /path/to/reframe/stage/
      Output dir prefix : /path/to/reframe/output/
      Logging dir       : /path/to/reframe/logs
  [==========] Running 13 check(s)
  [==========] Started on Sat May 26 00:48:03 2018

  [----------] started processing Example1Test (Simple matrix-vector multiplication example)
  [ RUN      ] Example1Test on daint:login using PrgEnv-cray
  [ RUN      ] Example1Test on daint:login using PrgEnv-gnu
  [ RUN      ] Example1Test on daint:login using PrgEnv-intel
  [ RUN      ] Example1Test on daint:login using PrgEnv-pgi
  [ RUN      ] Example1Test on daint:gpu using PrgEnv-cray
  [ RUN      ] Example1Test on daint:gpu using PrgEnv-gnu
  [ RUN      ] Example1Test on daint:gpu using PrgEnv-intel
  [ RUN      ] Example1Test on daint:gpu using PrgEnv-pgi
  [ RUN      ] Example1Test on daint:mc using PrgEnv-cray
  [ RUN      ] Example1Test on daint:mc using PrgEnv-gnu
  [ RUN      ] Example1Test on daint:mc using PrgEnv-intel
  [ RUN      ] Example1Test on daint:mc using PrgEnv-pgi
  [----------] finished processing Example1Test (Simple matrix-vector multiplication example)

  [----------] started processing Example2aTest (Matrix-vector multiplication example with OpenMP)
  [ RUN      ] Example2aTest on daint:login using PrgEnv-cray
  [ RUN      ] Example2aTest on daint:login using PrgEnv-gnu
  [ RUN      ] Example2aTest on daint:login using PrgEnv-intel
  [ RUN      ] Example2aTest on daint:login using PrgEnv-pgi
  [ RUN      ] Example2aTest on daint:gpu using PrgEnv-cray
  [ RUN      ] Example2aTest on daint:gpu using PrgEnv-gnu
  [ RUN      ] Example2aTest on daint:gpu using PrgEnv-intel
  [ RUN      ] Example2aTest on daint:gpu using PrgEnv-pgi
  [ RUN      ] Example2aTest on daint:mc using PrgEnv-cray
  [ RUN      ] Example2aTest on daint:mc using PrgEnv-gnu
  [ RUN      ] Example2aTest on daint:mc using PrgEnv-intel
  [ RUN      ] Example2aTest on daint:mc using PrgEnv-pgi
  [----------] finished processing Example2aTest (Matrix-vector multiplication example with OpenMP)
  <output omitted>
  [----------] waiting for spawned checks to finish
  [       OK ] MPITest on daint:gpu using PrgEnv-pgi
  [       OK ] MPITest on daint:gpu using PrgEnv-gnu
  [       OK ] OpenMPTest on daint:mc using PrgEnv-pgi
  [       OK ] OpenMPTest on daint:mc using PrgEnv-gnu
  [       OK ] OpenMPTest on daint:gpu using PrgEnv-pgi
  [       OK ] OpenMPTest on daint:gpu using PrgEnv-gnu
  <output omitted>
  [       OK ] Example1Test on daint:login using PrgEnv-cray
  [       OK ] MPITest on daint:mc using PrgEnv-cray
  [       OK ] MPITest on daint:gpu using PrgEnv-cray
  [       OK ] OpenMPTest on daint:mc using PrgEnv-cray
  [       OK ] OpenMPTest on daint:gpu using PrgEnv-cray
  [       OK ] SerialTest on daint:login using PrgEnv-pgi
  [       OK ] MPITest on daint:mc using PrgEnv-gnu
  [       OK ] OpenMPTest on daint:mc using PrgEnv-intel
  [       OK ] OpenMPTest on daint:login using PrgEnv-gnu
  [       OK ] OpenMPTest on daint:gpu using PrgEnv-intel
  [       OK ] MPITest on daint:gpu using PrgEnv-intel
  [       OK ] CudaTest on daint:gpu using PrgEnv-gnu
  [       OK ] OpenACCTest on daint:gpu using PrgEnv-pgi
  [       OK ] MPITest on daint:mc using PrgEnv-intel
  [       OK ] CudaTest on daint:gpu using PrgEnv-cray
  [       OK ] MPITest on daint:mc using PrgEnv-pgi
  [       OK ] OpenACCTest on daint:gpu using PrgEnv-cray
  [       OK ] CudaTest on daint:gpu using PrgEnv-pgi
  [----------] all spawned checks have finished

  [  PASSED  ] Ran 101 test case(s) from 13 check(s) (0 failure(s))
  [==========] Finished on Sat May 26 00:52:02 2018


The asynchronous execution policy may provide significant overall performance benefits for run-only regression tests.
For compile-only and normal tests that require a compilation, the execution time will be bound by the total compilation time of the test.


Manipulating modules
--------------------

.. versionadded:: 2.11

ReFrame allows you to change the modules loaded by a regression test on-the-fly without having to edit the regression test file.
This feature is extremely useful when you need to quickly test a newer version of a module, but it also allows you to completely decouple the module names used in your regression tests from the real module names in a system, thus making your test even more portable.
This is achieved by defining *module mappings*.

There are two ways to pass module mappings to ReFrame.
The first is to use the ``--map-module`` command-line option, which accepts a module mapping.
For example, the following line maps the module ``test_module`` to the module ``real_module``:

.. code-block:: none

  --map-module='test_module: real_module'

In this case, whenever ReFrame is asked to load ``test_module``, it will load ``real_module``.
Any string without spaces may be accepted in place of ``test_module`` and ``real_module``.
You can also define multiple module mappings at once by repeating the ``--map-module``.
If more than one mapping is specified for the same module, then the last mapping will take precedence.
It is also possible to map a single module to more than one target.
This can be done by listing the target modules separated by spaces in the order that they should be loaded.
In the following example, ReFrame will load ``real_module0`` and ``real_module1`` whenever the ``test_module`` is encountered:

.. code-block:: none

  --map-module 'test_module: real_module0 real_module1'

The second way of defining mappings is by listing them on a file, which you can then pass to ReFrame through the command-line option ``--module-mappings``.
Each line on the file corresponds to the definition of a mapping for a single module.
The syntax of the individual mappings in the file is the same as with the option ``--map-module`` and the same rules apply regarding repeated definitions.
Text starting with ``#`` is considered a comment and is ignored until the end of line is encountered.
Empty lines are ignored.
The following block shows an example of module mapping file:

.. code-block:: none

  module-1: module-1a  # an inline comment
  module-2: module-2a module-2b module-2c

  # This is a full line comment
  module-4: module-4a module-4b

If both ``--map-module`` and ``--module-mappings`` are passed, ReFrame will first create a mapping from the definitions on the file and it will then process the definitions passed with the ``--map-module`` options.
As usual, later definitions will override the former.

A final note on module mappings.
Module mappings can be arbitrarily deep as long as they do not form a cycle.
In this case, ReFrame will issue an error (denoting the offending cyclic dependency).
For example, suppose having the following mapping file:

.. code-block:: none

   cudatoolkit: foo
   foo: bar
   bar: foobar
   foobar: cudatoolkit

If you now try to run a test that loads the module `cudatoolkit`, the following error will be yielded:

.. code-block:: none

   ------------------------------------------------------------------------------
   FAILURE INFO for Example7Test
     * System partition: daint:gpu
     * Environment: PrgEnv-gnu
     * Stage directory: None
     * Job type: batch job (id=-1)
     * Maintainers: ['you-can-type-your-email-here']
     * Failing phase: setup
     * Reason: caught framework exception: module cyclic dependency: cudatoolkit->foo->bar->foobar->cudatoolkit
   ------------------------------------------------------------------------------
