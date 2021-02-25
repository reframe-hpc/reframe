===========================
Tutorial 4: Tips and Tricks
===========================

.. versionadded:: 3.4

This tutorial focuses on some less known aspects of ReFrame's command line interface that can be helpful.


Debugging
---------

ReFrame tests are Python classes inside Python source files, so the usual debugging techniques for Python apply, but the ReFrame frontend will filter some errors and stack traces by default in order to keep the output clean.
ReFrame test files are imported, so any error that appears during import time will cause the test loading process to fail and print a stack trace pointing to the offending line.
In the following, we have inserted a small typo in the ``hello2.py`` tutorial example:

.. code:: bash

   ./bin/reframe -c tutorials/basics/hello -R -l

.. code-block:: none

   ./bin/reframe: name error: name 'rm' is not defined
   ./bin/reframe: Traceback (most recent call last):
     File "/Users/karakasv/Repositories/reframe/reframe/frontend/cli.py", line 668, in main
       checks_found = loader.load_all()
     File "/Users/karakasv/Repositories/reframe/reframe/frontend/loader.py", line 204, in load_all
       checks.extend(self.load_from_dir(d, self._recurse))
     File "/Users/karakasv/Repositories/reframe/reframe/frontend/loader.py", line 189, in load_from_dir
       checks.extend(self.load_from_file(entry.path))
     File "/Users/karakasv/Repositories/reframe/reframe/frontend/loader.py", line 174, in load_from_file
       return self.load_from_module(util.import_module_from_file(filename))
     File "/Users/karakasv/Repositories/reframe/reframe/utility/__init__.py", line 96, in import_module_from_file
       return importlib.import_module(module_name)
     File "/usr/local/Cellar/python/3.7.7/Frameworks/Python.framework/Versions/3.7/lib/python3.7/importlib/__init__.py", line 127, in import_module
       return _bootstrap._gcd_import(name[level:], package, level)
     File "<frozen importlib._bootstrap>", line 1006, in _gcd_import
     File "<frozen importlib._bootstrap>", line 983, in _find_and_load
     File "<frozen importlib._bootstrap>", line 967, in _find_and_load_unlocked
     File "<frozen importlib._bootstrap>", line 677, in _load_unlocked
     File "<frozen importlib._bootstrap_external>", line 728, in exec_module
     File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
     File "/Users/karakasv/Repositories/reframe/tutorials/basics/hello/hello2.py", line 10, in <module>
       @rm.parameterized_test(['c'], ['cpp'])
   NameError: name 'rm' is not defined


However, if there is a Python error inside your test's constructor, ReFrame will issue a warning and keep on loading and initializing the rest of the tests.

.. code-block:: none

   ./bin/reframe: skipping test due to errors: HelloMultiLangTest: use `-v' for more information
     FILE: /Users/karakasv/Repositories/reframe/tutorials/basics/hello/hello2.py:13
   ./bin/reframe: skipping test due to errors: HelloMultiLangTest: use `-v' for more information
     FILE: /Users/karakasv/Repositories/reframe/tutorials/basics/hello/hello2.py:13
   [List of matched checks]
   - HelloTest (found in '/Users/karakasv/Repositories/reframe/tutorials/basics/hello/hello1.py')
   Found 1 check(s)


As suggested by the warning message, passing :option:`-v` will give you the stack trace for each of the failing tests, as well as some more information about what is going on during the loading.

.. code:: bash

   ./bin/reframe -c tutorials/basics/hello -R -lv

.. code-block:: none

   ./bin/reframe: skipping test due to errors: HelloMultiLangTest: use `-v' for more information
     FILE: /Users/karakasv/Repositories/reframe/tutorials/basics/hello/hello2.py:13
   Traceback (most recent call last):
     File "/Users/karakasv/Repositories/reframe/reframe/core/decorators.py", line 49, in _instantiate_all
       ret.append(_instantiate(cls, args))
     File "/Users/karakasv/Repositories/reframe/reframe/core/decorators.py", line 32, in _instantiate
       return cls(*args)
     File "/Users/karakasv/Repositories/reframe/tutorials/basics/hello/hello2.py", line 13, in __init__
       foo
   NameError: name 'foo' is not defined

   ./bin/reframe: skipping test due to errors: HelloMultiLangTest: use `-v' for more information
     FILE: /Users/karakasv/Repositories/reframe/tutorials/basics/hello/hello2.py:13
   Traceback (most recent call last):
     File "/Users/karakasv/Repositories/reframe/reframe/core/decorators.py", line 49, in _instantiate_all
       ret.append(_instantiate(cls, args))
     File "/Users/karakasv/Repositories/reframe/reframe/core/decorators.py", line 32, in _instantiate
       return cls(*args)
     File "/Users/karakasv/Repositories/reframe/tutorials/basics/hello/hello2.py", line 13, in __init__
       foo
   NameError: name 'foo' is not defined

   Loaded 1 test(s)
   Generated 1 test case(s)
   Filtering test cases(s) by name: 1 remaining
   Filtering test cases(s) by tags: 1 remaining
   Filtering test cases(s) by other attributes: 1 remaining
   Final number of test cases: 1
   [List of matched checks]
   - HelloTest (found in '/Users/karakasv/Repositories/reframe/tutorials/basics/hello/hello1.py')
   Found 1 check(s)
   Log file(s) saved in: '/var/folders/h7/k7cgrdl13r996m4dmsvjq7v80000gp/T/rfm-ckymcl44.log'


.. tip::
   The :option:`-v` option can be given multiple times to increase the verbosity level further.


Debugging deferred expressions
==============================

Although deferred expression that are used in :attr:`sanity_patterns` and :attr:`perf_patterns` behave similarly to normal Python expressions, you need to understand their `implicit evaluation rules <sanity_functions_reference.html#implicit-evaluation-of-sanity-functions>`__.
One of the rules is that :func:`str` triggers the implicit evaluation, so trying to use the standard :func:`print` function with a deferred expression, you might get unexpected results if that expression is not yet to be evaluated.
For this reason, ReFrame offers a sanity function counterpart of :func:`print`, which allows you to safely print deferred expressions.

Let's see that in practice, by printing the filename of the standard output for :class:`HelloMultiLangTest` test.
The :attr:`stdout <reframe.core.pipeline.RegressionTest.stdout>` is a deferred expression and it will get its value later on while the test executes.
Trying to use the standard print here :func:`print` function here would be of little help, since it would simply give us :obj:`None`, which is the value of :attr:`stdout` when the test is created.


.. code-block:: python
   :emphasize-lines: 11

   import reframe as rfm
   import reframe.utility.sanity as sn


   @rfm.parameterized_test(['c'], ['cpp'])
   class HelloMultiLangTest(rfm.RegressionTest):
       def __init__(self, lang):
           self.valid_systems = ['*']
           self.valid_prog_environs = ['*']
           self.sourcepath = f'hello.{lang}'
           self.sanity_patterns = sn.assert_found(r'Hello, World\!', sn.print(self.stdout))


If we run the test, we can see that the correct standard output filename will be printed after sanity:

.. code:: bash

   ./bin/reframe -C tutorials/config/settings.py -c tutorials/basics/hello/hello2.py -r

.. code-block:: none

   [----------] waiting for spawned checks to finish
   rfm_HelloMultiLangTest_cpp_job.out
   [       OK ] (1/4) HelloMultiLangTest_cpp on catalina:default using gnu [compile: 0.677s run: 0.700s total: 1.394s]
   rfm_HelloMultiLangTest_c_job.out
   [       OK ] (2/4) HelloMultiLangTest_c on catalina:default using gnu [compile: 0.451s run: 1.788s total: 2.258s]
   rfm_HelloMultiLangTest_c_job.out
   [       OK ] (3/4) HelloMultiLangTest_c on catalina:default using clang [compile: 0.329s run: 1.585s total: 1.934s]
   rfm_HelloMultiLangTest_cpp_job.out
   [       OK ] (4/4) HelloMultiLangTest_cpp on catalina:default using clang [compile: 0.609s run: 0.373s total: 1.004s]
   [----------] all spawned checks have finished

   [  PASSED  ] Ran 4 test case(s) from 2 check(s) (0 failure(s))
   [==========] Finished on Wed Jan 20 17:19:01 2021


Debugging test loading
======================

If you are new to ReFrame, you might wonder sometimes why your tests are not loading or why your tests are not running on the partition they were supposed to run.
This can be due to ReFrame picking the wrong configuration entry or that your test is not written properly (not decorated, no :attr:`valid_systems` etc.).
If you try to load a test file and list its tests by increasing twice the verbosity level, you will get enough output to help you debug such issues.
Let's try loading the ``tutorials/basics/hello/hello2.py`` file:

.. code:: bash

   ./bin/reframe -C tutorials/config/settings.py -c tutorials/basics/hello/hello2.py -lvv


.. code-block:: none

   Loading user configuration
   Loading configuration file: 'tutorials/config/settings.py'
   Detecting system
   Looking for a matching configuration entry for system 'dhcp-133-191.cscs.ch'
   Configuration found: picking system 'generic'
   Selecting subconfig for 'generic'
   Initializing runtime
   Selecting subconfig for 'generic:default'
   Initializing system partition 'default'
   Selecting subconfig for 'generic'
   Initializing system 'generic'
   Initializing modules system 'nomod'
   [ReFrame Environment]
     RFM_CHECK_SEARCH_PATH=<not set>
     RFM_CHECK_SEARCH_RECURSIVE=<not set>
     RFM_CLEAN_STAGEDIR=<not set>
     RFM_COLORIZE=<not set>
     RFM_CONFIG_FILE=/Users/user/Repositories/reframe/tutorials/config/settings.py
     RFM_GRAYLOG_ADDRESS=<not set>
     RFM_IGNORE_CHECK_CONFLICTS=<not set>
     RFM_IGNORE_REQNODENOTAVAIL=<not set>
     RFM_INSTALL_PREFIX=/Users/user/Repositories/reframe
     RFM_KEEP_STAGE_FILES=<not set>
     RFM_MODULE_MAPPINGS=<not set>
     RFM_MODULE_MAP_FILE=<not set>
     RFM_NON_DEFAULT_CRAYPE=<not set>
     RFM_OUTPUT_DIR=<not set>
     RFM_PERFLOG_DIR=<not set>
     RFM_PREFIX=<not set>
     RFM_PURGE_ENVIRONMENT=<not set>
     RFM_REPORT_FILE=<not set>
     RFM_SAVE_LOG_FILES=<not set>
     RFM_STAGE_DIR=<not set>
     RFM_SYSLOG_ADDRESS=<not set>
     RFM_SYSTEM=<not set>
     RFM_TIMESTAMP_DIRS=<not set>
     RFM_UNLOAD_MODULES=<not set>
     RFM_USER_MODULES=<not set>
     RFM_USE_LOGIN_SHELL=<not set>
     RFM_VERBOSE=<not set>
   [ReFrame Setup]
     version:           3.4-dev2 (rev: 33a97c81)
     command:           './bin/reframe -C tutorials/config/settings.py -c tutorials/basics/hello/hello2.py -lvv'
     launched by:       user@dhcp-133-191.cscs.ch
     working directory: '/Users/user/Repositories/reframe'
     settings file:     'tutorials/config/settings.py'
     check search path: '/Users/user/Repositories/reframe/tutorials/basics/hello/hello2.py'
     stage directory:   '/Users/user/Repositories/reframe/stage'
     output directory:  '/Users/user/Repositories/reframe/output'

   Looking for tests in '/Users/user/Repositories/reframe/tutorials/basics/hello/hello2.py'
   Validating '/Users/user/Repositories/reframe/tutorials/basics/hello/hello2.py': OK
     > Loaded 2 test(s)
   Loaded 2 test(s)
   Generated 2 test case(s)
   Filtering test cases(s) by name: 2 remaining
   Filtering test cases(s) by tags: 2 remaining
   Filtering test cases(s) by other attributes: 2 remaining
   Building and validating the full test DAG
   Full test DAG:
     ('HelloMultiLangTest_c', 'generic:default', 'builtin') -> []
     ('HelloMultiLangTest_cpp', 'generic:default', 'builtin') -> []
   Final number of test cases: 2
   [List of matched checks]
   - HelloMultiLangTest_c (found in '/Users/user/Repositories/reframe/tutorials/basics/hello/hello2.py')
   - HelloMultiLangTest_cpp (found in '/Users/user/Repositories/reframe/tutorials/basics/hello/hello2.py')
   Found 2 check(s)
   Log file(s) saved in: '/var/folders/h7/k7cgrdl13r996m4dmsvjq7v80000gp/T/rfm-3956_dlu.log'

You can see all the different phases ReFrame's frontend goes through when loading a test.
The first "strange" thing to notice in this log is that ReFrame picked the generic system configuration.
This happened because it couldn't find a system entry with a matching hostname pattern.
However, it did not impact the test loading, because these tests are valid for any system, but it will affect the tests when running (see :doc:`tutorial_basics`) since the generic system does not define any C++ compiler.

After loading the configuration, ReFrame will print out its relevant environment variables and will start examining the given files in order to find and load ReFrame tests.
Before attempting to load a file, it will validate it and check if it looks like a ReFrame test.
If it does, it will load that file by importing it.
This is where any ReFrame tests are instantiated and initialized (see ``Loaded 2 test(s)``), as well as the actual test cases (combination of tests, system partitions and environments) are generated.
Then the test cases are filtered based on the various `filtering command line options <manpage.html#test-filtering>`__ as well as the programming environments that are defined for the currently selected system.
Finally, the test case dependency graph is built and everything is ready for running (or listing).

Try passing a specific system or partition with the :option:`--system` option or modify the test (e.g., removing the decorator that registers it) and see how the logs change.


Execution modes
---------------

ReFrame allows you to create pre-defined ways of running it, which you can invoke from the command line.
These are called *execution modes* and are essentially named groups of command line options that will be passed to ReFrame whenever you request them.
These are defined in the configuration file and can be requested with the :option:`--mode` command-line option.
The following configuration defines an execution mode named ``maintenance`` and sets up ReFrame in a certain way (selects tests to run, sets up stage and output paths etc.)

.. code-block:: python

   'modes': [
       {
           'name': 'maintenance',
           'options': [
               '--unload-module=reframe',
               '--exec-policy=async',
               '--strict',
               '--output=/path/to/$USER/regression/maintenance',
               '--perflogdir=/path/to/$USER/regression/maintenance/logs',
               '--stage=$SCRATCH/regression/maintenance/stage',
               '--report-file=/path/to/$USER/regression/maintenance/reports/maint_report_{sessionid}.json',
               '-Jreservation=maintenance',
               '--save-log-files',
               '--tag=maintenance',
               '--timestamp=%F_%H-%M-%S'
           ]
       },
  ]

The execution modes come handy in situations that you have a standardized way of running ReFrame and you don't want to create and maintain shell scripts around it.
In this example, you can simply run ReFrame with

.. code:: bash

  ./bin/reframe --mode=maintenance -r

and it will be equivalent to passing explicitly all the above options.
You can still pass any additional command line option and it will supersede or be combined (depending on the behaviour of the option) with those defined in the execution mode.
In this particular example, we could change just the reservation name by running

.. code:: bash

  ./bin/reframe --mode=maintenance -J reservation=maint -r

There are two options that you can't use inside execution modes and these are the :option:`-C` and :option:`--system`.
The reason is that these option select the configuration file and the configuration entry to load.


Manipulating ReFrame's environment
----------------------------------

ReFrame runs the selected tests in the same environment as the one that it executes.
It does not unload any environment modules nor sets or unsets any environment variable.
Nonetheless, it gives you the opportunity to modify the environment that the tests execute.
You can either purge completely all environment modules by passing the :option:`--purge-env` option or ask ReFrame to load or unload some environment modules before starting running any tests by using the :option:`-m` and :option:`-u` options respectively.
Of course you could manage the environment manually, but it's more convenient if you do that directly through ReFrame's command-line.
If you used an environment module to load ReFrame, e.g., ``reframe``, you can use the :option:`-u` to have ReFrame unload it before running any tests, so that the tests start in a clean environment:

.. code:: bash

   ./bin/reframe -u reframe [...]


Environment Modules Mappings
----------------------------

ReFrame allows you to replace environment modules used in tests with other modules on the fly.
This is quite useful if you want to test a new version of a module or another combination of modules.
Assume you have a test that loads a ``gromacs`` module:

.. code-block:: python

   class GromacsTest(rfm.RunOnlyRegressionTest):
       def __init__(self):
           ...
           self.modules = ['gromacs']


This test would the default version of the module in the system, but you might want to test another version, before making that new one the default.
You can ask ReFrame to temporarily replace the ``gromacs`` module with another one as follows:


.. code-block:: bash

   ./bin/reframe -n GromacsTest -M 'gromacs:gromacs/2020.5' -r


Every time ReFrame tries to load the ``gromacs`` module, it will replace it with ``gromacs/2020.5``.
You can specify multiple mappings at once or provide a file with mappings using the :option:`--module-mappings` option.
You can also replace a single module with multiple modules.

A very convenient feature of ReFrame in dealing with modules is that you do not have to care about module conflicts at all, regardless of the modules system backend.
ReFrame will take care of unloading any conflicting modules, if the underlying modules system cannot do that automatically.
In case of module mappings, it will also respect the module order of the replacement modules and will produce the correct series of "load" and "unload" commands needed by the modules system backend used.


Retrying and Rerunning Tests
----------------------------

If you are running ReFrame regularly as part of a continuous testing procedure you might not want it to generate alerts for transient failures.
If a ReFrame test fails, you might want to retry a couple of times before marking it as a failure.
You can achieve this with the :option:`--max-retries`.
ReFrame will then retry the failing test cases a maximum number of times before reporting them as actual failures.
The failed test cases will not be retried immediately after they have failed, but rather at the end of the run session.
This is done to give more chances of success in case the failures have been transient.

Another interesting feature introduced in ReFrame 3.4 is the ability to restore a previous test session.
Whenever it runs, ReFrame stores a detailed JSON report of the last run under ``$HOME/.reframe`` (see :option:`--report-file`).
Using that file, ReFrame can restore a previous run session using the :option:`--restore-session`.
This option is useful when you combine it with the various test filtering options.
For example, you might want to rerun only the failed tests or just a specific test in a dependency chain.
Let's see an artificial example that uses the following test dependency graph.

.. _fig-deps-complex:

.. figure:: _static/img/deps-complex.svg
   :align: center

   :sub:`Complex test dependency graph. Nodes in red are set to fail.`



Tests :class:`T2` and :class:`T8` are set to fail.
Let's run the whole test DAG:

.. code-block:: bash

   ./bin/reframe -c unittests/resources/checks_unlisted/deps_complex.py -r

.. code-block:: none

   <output omitted>

   [----------] waiting for spawned checks to finish
   [       OK ] ( 1/10) T0 on generic:default using builtin [compile: 0.014s run: 0.297s total: 0.337s]
   [       OK ] ( 2/10) T4 on generic:default using builtin [compile: 0.010s run: 0.171s total: 0.207s]
   [       OK ] ( 3/10) T5 on generic:default using builtin [compile: 0.010s run: 0.192s total: 0.225s]
   [       OK ] ( 4/10) T1 on generic:default using builtin [compile: 0.008s run: 0.198s total: 0.226s]
   [     FAIL ] ( 5/10) T8 on generic:default using builtin [compile: n/a run: n/a total: 0.003s]
   ==> test failed during 'setup': test staged in '/Users/user/Repositories/reframe/stage/generic/default/builtin/T8'
   [     FAIL ] ( 6/10) T9 [compile: n/a run: n/a total: n/a]
   ==> test failed during 'startup': test staged in '<not available>'
   [       OK ] ( 7/10) T6 on generic:default using builtin [compile: 0.007s run: 0.224s total: 0.262s]
   [       OK ] ( 8/10) T3 on generic:default using builtin [compile: 0.007s run: 0.211s total: 0.235s]
   [     FAIL ] ( 9/10) T2 on generic:default using builtin [compile: 0.011s run: 0.318s total: 0.389s]
   ==> test failed during 'sanity': test staged in '/Users/user/Repositories/reframe/stage/generic/default/builtin/T2'
   [     FAIL ] (10/10) T7 [compile: n/a run: n/a total: n/a]
   ==> test failed during 'startup': test staged in '<not available>'
   [----------] all spawned checks have finished

   [  FAILED  ] Ran 10 test case(s) from 10 check(s) (4 failure(s))
   [==========] Finished on Thu Jan 21 13:58:43 2021

   <output omitted>

You can restore the run session and run only the failed test cases as follows:


.. code-block:: bash

   ./bin/reframe --restore-session --failed -r


Of course, as expected, the run will fail again, since these tests were designed to fail.

Instead of running the failed test cases of a previous run, you might simply want to rerun a specific test.
This has little meaning if you don't use dependencies, because it would be equivalent to running it separately using the :option:`-n` option.
However, if a test was part of a dependency chain, using :option:`--restore-session` will not rerun its dependencies, but it will rather restore them.
This is useful in cases where the test that we want to rerun depends on time-consuming tests.
There is a little tweak, though, for this to work:
you need to have run with :option:`--keep-stage-files` in order to keep the stage directory even for tests that have passed.
This is due to two reasons:
(a) if a test needs resources from its parents, it will look into their stage directories and
(b) ReFrame stores the state of a finished test case inside its stage directory and it will need that state information in order to restore a test case.

Let's try to rerun the :class:`T6` test from the previous test dependency chain:


.. code-block:: bash

   ./bin/reframe -c unittests/resources/checks_unlisted/deps_complex.py --keep-stage-files -r

.. code-block:: bash

   ./bin/reframe --restore-session --keep-stage-files -n T6 -r


Notice how only the :class:`T6` test was rerun and none of its dependencies, since they were simply restored:

.. code-block:: none

   [==========] Running 1 check(s)
   [==========] Started on Thu Jan 21 14:27:18 2021

   [----------] started processing T6 (T6)
   [ RUN      ] T6 on generic:default using builtin
   [----------] finished processing T6 (T6)

   [----------] waiting for spawned checks to finish
   [       OK ] (1/1) T6 on generic:default using builtin [compile: 0.012s run: 0.428s total: 0.464s]
   [----------] all spawned checks have finished

   [  PASSED  ] Ran 1 test case(s) from 1 check(s) (0 failure(s))
   [==========] Finished on Thu Jan 21 14:27:19 2021


If we tried to run :class:`T6` without restoring the session, we would have to rerun also the whole dependency chain, i.e., also :class:`T5`, :class:`T1`, :class:`T4` and :class:`T0`.

.. code-block:: bash

   ./bin/reframe -c unittests/resources/checks_unlisted/deps_complex.py -n T6 -r

.. code-block:: none

   [----------] waiting for spawned checks to finish
   [       OK ] (1/5) T0 on generic:default using builtin [compile: 0.012s run: 0.424s total: 0.464s]
   [       OK ] (2/5) T4 on generic:default using builtin [compile: 0.011s run: 0.348s total: 0.381s]
   [       OK ] (3/5) T5 on generic:default using builtin [compile: 0.007s run: 0.225s total: 0.248s]
   [       OK ] (4/5) T1 on generic:default using builtin [compile: 0.009s run: 0.235s total: 0.267s]
   [       OK ] (5/5) T6 on generic:default using builtin [compile: 0.010s run: 0.265s total: 0.297s]
   [----------] all spawned checks have finished


   [  PASSED  ] Ran 5 test case(s) from 5 check(s) (0 failure(s))
   [==========] Finished on Thu Jan 21 14:32:09 2021


.. _generate-ci-pipeline:

Integrating into a CI pipeline
------------------------------

.. versionadded:: 3.4.1

Instead of running your tests, you can ask ReFrame to generate a `child pipeline <https://docs.gitlab.com/ee/ci/parent_child_pipelines.html>`__ specification for the Gitlab CI.
This will spawn a CI job for each ReFrame test respecting test dependencies.
You could run your tests in a single job of your Gitlab pipeline, but you would not take advantage of the parallelism across different CI jobs.
Having a separate CI job per test makes it also easier to spot the failing tests.

As soon as you have set up a `runner <https://docs.gitlab.com/ee/ci/quick_start/>`__ for your repository, it is fairly straightforward to use ReFrame to automatically generate the necessary CI steps.
The following is an example of ``.gitlab-ci.yml`` file that does exactly that:

.. code-block:: yaml

   stages:
     - generate
     - test

   generate-pipeline:
     stage: generate
     script:
       - reframe --ci-generate=${CI_PROJECT_DIR}/pipeline.yml -c ${CI_PROJECT_DIR}/path/to/tests
     artifacts:
       paths:
         - ${CI_PROJECT_DIR}/pipeline.yml

   test-jobs:
     stage: test
     trigger:
       include:
         - artifact: pipeline.yml
           job: generate-pipeline
       strategy: depend


It defines two stages.
The first one, called ``generate``, will call ReFrame to generate the pipeline specification for the desired tests.
All the usual `test selection options <manpage.html#test-filtering>`__ can be used to select specific tests.
ReFrame will process them as usual, but instead of running the selected tests, it will generate the correct steps for running each test individually as a Gitlab job.
We then pass the generated CI pipeline file to second phase as an artifact and we are done!

The following figure shows one part of the automatically generated pipeline for the test graph depicted `above <#fig-deps-complex>`__.

.. figure:: _static/img/gitlab-ci.png
   :align: center

   :sub:`Snapshot of a Gitlab pipeline generated automatically by ReFrame.`


.. note::

   The ReFrame executable must be available in the Gitlab runner that will run the CI jobs.
