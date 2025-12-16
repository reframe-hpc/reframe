======================
Command Line Reference
======================


Synopsis
========

.. option:: reframe [OPTION]... ACTION


Description
===========

ReFrame provides both a :doc:`programming interface <regression_test_api>` for writing regression tests and a command-line interface for managing and running the tests, which is detailed here.
The ``reframe`` command is part of ReFrame's frontend.
This frontend is responsible for loading and running regression tests written in ReFrame.
ReFrame executes tests by sending them down to a well defined pipeline.
The implementation of the different stages of this pipeline is part of ReFrame's core architecture, but the frontend is responsible for driving this pipeline and executing tests through it.
Usually, ReFrame processes tests in three phases:

1. It :ref:`discovers and loads tests <test-discovery>` from the filesystem.
2. It :ref:`filters <test-filtering>` the loaded tests based on the current system and any other criteria specified by the user.
3. It :ref:`acts <commands>` upon the selected tests.

There are also ReFrame commands that do not operate on a set of tests.


.. _commands:

Commands
--------

ReFrame commands are mutually exclusive and one of them must always be specified.
There are commands that act upon the selected tests and others that have a helper function, such as querying the configuration, querying the results database etc.

.. versionchanged:: 4.7

   ReFrame commands are now mutually exclusive and only one can be specified every time.


Test commands
^^^^^^^^^^^^^

.. option:: --ci-generate=FILE

   Generate a Gitlab `child pipeline <https://docs.gitlab.com/ee/ci/parent_child_pipelines.html>`__ specification in ``FILE`` that will run the selected tests.

   You can set up your Gitlab CI to use the generated file to run every test as a separate Gitlab job respecting test dependencies.
   For more information, have a look in :ref:`generate-ci-pipeline`.

   .. note::
      This option will not work with the :ref:`test generation options <test-generators>`.


   .. versionadded:: 3.4.1

.. option:: --describe

   Print a detailed description of the `selected tests <#test-filtering>`__ in JSON format and exit.

   .. note::
      The generated test description corresponds to its state after it has been initialized.
      If any of its attributes are changed or set during its execution, their updated values will not be shown by this listing.

   .. versionadded:: 3.10.0


.. option:: --dry-run

   Dry run the selected tests.

   The dry-run mode will try to execute as much of the test pipeline as possible.
   More specifically, the tests will not be submitted and will not be run for real,
   but their stage directory will be prepared and the corresponding job script will be emitted.
   Similarly, the sanity and performance functions will not be evaluated but all the preparation will happen.
   Tests run in dry-run mode will not fail unless there is a programming error in the test or if the test tries to use a resource that is not produced in dry run mode (e.g., access the standard output or a resource produced by a dependency outside any sanity or performance function).
   In this case, users can call the :func:`~reframe.core.pipeline.RegressionTest.is_dry_run` method in their test and take a specific action if the test is run in dry-run mode.

   .. versionadded:: 4.1

.. option:: -L, --list-detailed[=T|C]

   List selected tests providing more details for each test.

   The unique id of each test (see also :attr:`~reframe.core.pipeline.RegressionTest.unique_name`) as well as the file where each test is defined are printed.

   This option accepts optionally a single argument denoting what type of listing is requested.
   Please refer to :option:`-l` for an explanation of this argument.

   .. versionadded:: 3.10.0
      Support for different types of listing is added.

   .. versionchanged:: 4.0.5
      The variable names to which fixtures are bound are also listed.
      See :ref:`test_naming_scheme` for more information.

.. option:: -l, --list[=T|C]

   List selected tests and their dependencies.

   This option accepts optionally a single argument denoting what type of listing is requested.
   There are two types of possible listings:

   - *Regular test listing* (``T``, the default): This type of listing lists the tests and their dependencies or fixtures using their :attr:`~reframe.core.pipeline.RegressionTest.display_name`. A test that is listed as a dependency of another test will not be listed separately.
   - *Concretized test case listing* (``C``): This type of listing lists the exact test cases and their dependencies as they have been concretized for the current system and environment combinations.
     This listing shows practically the exact test DAG that will be executed.

   .. versionadded:: 3.10.0
      Support for different types of listing is added.

   .. versionchanged:: 4.0.5
      The variable names to which fixtures are bound are also listed.
      See :ref:`test_naming_scheme` for more information.

.. option:: --list-tags

   List the unique tags of the selected tests.

   The tags are printed in alphabetical order.

   .. versionadded:: 3.6.0

.. option:: -r, --run

   Run the selected tests.


Result storage commands
^^^^^^^^^^^^^^^^^^^^^^^

.. option:: --delete-stored-sessions=SELECT_SPEC

   Delete the stored sessions matching the given selection criteria.

   Check :ref:`session-selection` for information on the exact syntax of ``SELECT_SPEC``.

   .. versionadded:: 4.7

.. option:: --describe-stored-sessions=SELECT_SPEC

   Get detailed information of the sessions matching the given selection criteria.

   The output is in JSON format.
   Check :ref:`session-selection` for information on the exact syntax of ``SELECT_SPEC``.

   .. versionadded:: 4.7

.. option:: --describe-stored-testcases=SELECT_SPEC

   Get detailed information of the test cases matching the given selection criteria.

   This option can be combined with :option:`--name` and :option:`--filter-expr` to restrict further the test cases.

   Check :ref:`session-selection` for information on the exact syntax of ``SELECT_SPEC``.

   .. versionadded:: 4.7

.. _--list-stored-sessions:

.. option:: --list-stored-sessions[=SELECT_SPEC|all]

   List sessions stored in the results database matching the given selection criteria.

   If ``all`` is given instead of ``SELECT_SPEC``, all stored sessions will be listed.
   This is equivalent to ``19700101T0000+0000:now``.
   If the ``SELECT_SPEC`` is not specified, only the sessions of last week will be listed (equivalent to ``now-1w:now``).

   Check :ref:`session-selection` for information on the exact syntax of ``SELECT_SPEC``.

   .. versionadded:: 4.7

.. option:: --list-stored-testcases=CMPSPEC

   Select and list information of stored testcases.

   The ``CMPSPEC`` argument specifies how testcases will be selected, aggregated and presented.
   This option can be combined with :option:`--name` and :option:`--filter-expr` to restrict the listed tests.

   Check the :ref:`querying-past-results` section for the exact syntax of ``CMPSPEC``.

   .. versionadded:: 4.7

.. option:: --performance-compare=CMPSPEC

   Compare the performance of test cases that have run in the past.

   The ``CMPSPEC`` argument specifies how testcases will be selected, aggregated and presented.
   This option can be combined with :option:`--name` and :option:`--filter-expr` to restrict the listed tests.
   The :option:`--filter-expr` option specifically can be specified twice, in which case the first expression will be used the to filter the first set of test cases, and the second one will filter the second set.

   Check the :ref:`querying-past-results` section for the exact syntax of ``CMPSPEC``.

   .. versionadded:: 4.7

   .. versionchanged:: 4.8

      The :option:`--filter-expr` can now be passed twice with :option:`--performance-compare`.


.. option:: --term-lhs=NAME
.. option:: --term-rhs=NAME

   Change the default suffix for columns in performance comparisons.

   These options are relevant only in conjunction with the :option:`--performance-compare` and :option:`--performance-report` options

   .. versionadded:: 4.9


Other commands
^^^^^^^^^^^^^^

.. _--detect-host-topology:

.. option:: --detect-host-topology[=FILE]

   Detect the local host processor topology, store it to ``FILE`` and exit.

   If no ``FILE`` is specified, the standard output will be used.

   .. versionadded:: 3.7.0

.. option:: --show-config [PARAM]

   Show the value of configuration parameter ``PARAM`` as this is defined for the currently selected system and exit.

   The parameter value is printed in JSON format.
   If ``PARAM`` is not specified or if it set to ``all``, the whole configuration for the currently selected system will be shown.
   Configuration parameters are formatted as a path navigating from the top-level configuration object to the actual parameter.
   The ``/`` character acts as a selector of configuration object properties or an index in array objects.
   The ``@`` character acts as a selector by name for configuration objects that have a ``name`` property.
   Here are some example queries:

   - Retrieve all the partitions of the current system:

     .. code:: bash

        reframe --show-config=systems/0/partitions

   - Retrieve the job scheduler of the partition named ``default``:

     .. code:: bash

        reframe --show-config=systems/0/partitions/@default/scheduler

   - Retrieve the check search path for system ``foo``:

     .. code:: bash

        reframe --system=foo --show-config=general/0/check_search_path

.. option:: -V, --version

   Print version and exit.


.. _test-discovery:

Test discovery and test loading
-------------------------------

This is the very first phase of the frontend.
ReFrame will search for tests in its *check search path* and will load them.
When ReFrame loads a test, it actually *instantiates* it, meaning that it will call its :func:`__init__` method unconditionally whether this test is meant to run on the selected system or not.
This is something that test developers should bear in mind.

.. option:: -c, --checkpath=PATH

   A filesystem path where ReFrame should search for tests.

   ``PATH`` can be a directory or a single test file.
   If it is a directory, ReFrame will search for test files inside this directory load all tests found in them.
   This option can be specified multiple times, in which case each ``PATH`` will be searched in order.

   The check search path can also be set using the :envvar:`RFM_CHECK_SEARCH_PATH` environment variable or the :attr:`~config.general.check_search_path` general configuration parameter.

.. option:: -R, --recursive

   Search for test files recursively in directories found in the check search path.

   This option can also be set using the :envvar:`RFM_CHECK_SEARCH_RECURSIVE` environment variable or the :attr:`~config.general.check_search_recursive` general configuration parameter.

.. note::
   ReFrame will fail to load a test with a relative import unless *any* of the following holds true:

   1. The test is located under ReFrame's installation prefix.
   2. The parent directory of the test contains an ``__init__.py`` file.

   For versions prior to 4.6, relative imports are supported only for case (1).


.. _test-filtering:

Test filtering
--------------

After all tests in the search path have been loaded, they are first filtered by the selected system.
Any test that is not valid for the current system, it will be filtered out.
The current system is either auto-selected or explicitly specified with the :option:`--system` option.
Tests can be filtered by different attributes and there are specific command line options for achieving this.
A common characteristic of all test filtering options is that if a test is selected, then all its dependencies will be selected, too, regardless if they match the filtering criteria or not.
This happens recursively so that if test ``T1`` depends on ``T2`` and ``T2`` depends on ``T3``, then selecting ``T1`` would also select ``T2`` and ``T3``.

.. option:: --cpu-only

   Select tests that do not target GPUs.

   These are all tests with :attr:`num_gpus_per_node` equals to zero
   This option and :option:`--gpu-only` are mutually exclusive.

   The :option:`--gpu-only` and :option:`--cpu-only` check only the value of the :attr:`num_gpus_per_node` attribute of tests.
   The value of this attribute is not required to be non-zero for GPU tests.
   Tests may or may not make use of it.

   .. deprecated:: 4.4

      Please use ``-E 'not num_gpus_per_node'`` instead.

.. option:: -E, --filter-expr=EXPR

   Select only tests that satisfy the given expression.

   The expression ``EXPR`` can be any valid Python expression on the test variables or parameters.
   For example, ``-E num_tasks > 10`` will select all tests, whose :attr:`~reframe.core.pipeline.RegressionTest.num_tasks` exceeds ``10``.
   You may use any test variable in expression, even user-defined.
   Multiple variables can also be included such as ``-E num_tasks >= my_param``, where ``my_param`` is user-defined parameter.

   .. versionadded:: 4.4

.. option:: --failed

   Select only the failed test cases for a previous run.

   This option can only be used in combination with the :option:`--restore-session`.
   To rerun the failed cases from the last run, you can use ``reframe --restore-session --failed -r``.

   .. versionadded:: 3.4


.. option:: --gpu-only

   Select tests that can run on GPUs.

   These are all tests with :attr:`num_gpus_per_node` greater than zero.
   This option and :option:`--cpu-only` are mutually exclusive.

   .. deprecated:: 4.4

      Please use ``-E num_gpus_per_node`` instead.

.. option:: --maintainer=MAINTAINER

   Filter tests by maintainer.

   ``MAINTAINER`` is interpreted as a `Python Regular Expression <https://docs.python.org/3/library/re.html>`__; all tests that have at least a matching maintainer will be selected.
   ``MAINTAINER`` being a regular expression has the implication that ``--maintainer 'foo'`` will select also tests that define ``'foobar'`` as a maintainer.
   To restrict the selection to tests defining only ``'foo'``, you should use ``--maintainer 'foo$'``.

   This option may be specified multiple times, in which case only tests defining or matching *all* maintainers will be selected.

   .. versionadded:: 3.9.1

   .. versionchanged:: 4.1.0

      The ``MAINTAINER`` pattern is matched anywhere in the maintainer's name and not at its beginning.
      If you want to match at the beginning of the name, you should prepend ``^``.


.. option:: -n, --name=NAME

   Filter tests by name.

   ``NAME`` is interpreted as a `Python Regular Expression <https://docs.python.org/3/library/re.html>`__;
   any test whose *display name* matches ``NAME`` will be selected.
   The display name of a test encodes also any parameterization information.
   See :ref:`test_naming_scheme` for more details on how the tests are automatically named by the framework.

   Before matching, any whitespace will be removed from the display name of the test.

   This option may be specified multiple times, in which case tests with *any* of the specified names will be selected:
   ``-n NAME1 -n NAME2`` is therefore equivalent to ``-n 'NAME1|NAME2'``.

   If the special notation ``<test_name>@<variant_num>`` is passed as the ``NAME`` argument, then an exact match will be performed selecting the variant ``variant_num`` of the test ``test_name``.

   You may also select a test by its hash code using the notation ``/<test-hash>`` for the ``NAME`` argument.

   .. note::

      Fixtures cannot be selected.

   .. versionchanged:: 3.10.0

      The option's behaviour was adapted and extended in order to work with the updated test naming scheme.

   .. versionchanged:: 4.0.0

      Support selecting tests by their hash code.

   .. versionchanged:: 4.1.0

      The ``NAME`` pattern is matched anywhere in the test name and not at its beginning.
      If you want to match at the beginning of a test name, you should prepend ``^``.


.. option:: -p, --prgenv=NAME

   Filter tests by programming environment.

   ``NAME`` is interpreted as a `Python Regular Expression <https://docs.python.org/3/library/re.html>`__;
   any test for which at least one valid programming environment is matching ``NAME`` will be selected.

   This option may be specified multiple times, in which case only tests matching all of the specified programming environments will be selected.

.. option:: --skip-prgenv-check

   Do not filter tests against programming environments.

   Even if the :option:`-p` option is not specified, ReFrame will filter tests based on the programming environments defined for the currently selected system.
   This option disables that filter completely.


.. option:: --skip-system-check

   Do not filter tests against the selected system.

.. option:: -T, --exclude-tag=TAG

   Exclude tests by tags.

   ``TAG`` is interpreted as a `Python Regular Expression <https://docs.python.org/3/library/re.html>`__;
   any test with tags matching ``TAG`` will be excluded.

   This option may be specified multiple times, in which case tests with *any* of the specified tags will be excluded:
   ``-T TAG1 -T TAG2`` is therefore equivalent to ``-T 'TAG1|TAG2'``.

   .. versionchanged:: 4.1.0

      The ``TAG`` pattern is matched anywhere in the tag name and not at its beginning.
      If you want to match at the beginning of a tag, you should prepend ``^``.

.. option:: -t, --tag=TAG

   Filter tests by tag.

   ``TAG`` is interpreted as a `Python Regular Expression <https://docs.python.org/3/library/re.html>`__; all tests that have at least a matching tag will be selected.
   ``TAG`` being a regular expression has the implication that ``-t 'foo'`` will select also tests that define ``'foobar'`` as a tag.
   To restrict the selection to tests defining only ``'foo'``, you should use ``-t 'foo$'``.

   This option may be specified multiple times, in which case only tests defining or matching *all* tags will be selected.

   .. versionchanged:: 4.1.0

      The ``TAG`` pattern is matched anywhere in the tag name and not at its beginning.
      If you want to match at the beginning of a tag, you should prepend ``^``.

.. option:: -x, --exclude=NAME

   Exclude tests by name.

   ``NAME`` is interpreted as a `Python Regular Expression <https://docs.python.org/3/library/re.html>`__;
   any test whose name matches ``NAME`` will be excluded.

   This option may be specified multiple times, in which case tests with *any* of the specified names will be excluded:
   ``-x NAME1 -x NAME2`` is therefore equivalent to ``-x 'NAME1|NAME2'``.

   .. versionchanged:: 4.1.0

      The ``NAME`` pattern is matched anywhere in the test name and not at its beginning.
      If you want to match at the beginning of a test name, you should prepend ``^``.


Options controlling ReFrame output
----------------------------------

.. option:: --compress-report

   Compress the generated run report (see :option:`--report-file`).
   The generated report is a JSON file formatted in a human readable form.
   If this option is enabled, the generated JSON file will be a single stream of text without additional spaces or new lines.

   This option can also be set using the :envvar:`RFM_COMPRESS_REPORT` environment variable or the :attr:`~config.general.compress_report` general configuration parameter.

   .. versionadded:: 3.12.0

.. option:: --dont-restage

   Do not restage a test if its stage directory exists.
   Normally, if the stage directory of a test exists, ReFrame will remove it and recreate it.
   This option disables this behavior.

   This option can also be set using the :envvar:`RFM_CLEAN_STAGEDIR` environment variable or the :attr:`~config.general.clean_stagedir` general configuration parameter.

   .. versionadded:: 3.1

   .. warning::

      Running a test with :option:`--dont-restage` on a stage directory that was created with a different ReFrame version is undefined behaviour.

.. option:: --keep-stage-files

   Keep test stage directories even for tests that finish successfully.

   This option can also be set using the :envvar:`RFM_KEEP_STAGE_FILES` environment variable or the :attr:`~config.general.keep_stage_files` general configuration parameter.

.. option:: -o, --output=DIR

   Directory prefix for test output files.

   When a test finishes successfully, ReFrame copies important output files to a test-specific directory for future reference.
   This test-specific directory is of the form ``{output_prefix}/{system}/{partition}/{environment}/{test_name}``,
   where ``output_prefix`` is set by this option.
   The test files saved in this directory are the following:

   - The ReFrame-generated build script, if not a run-only test.
   - The standard output and standard error of the build phase, if not a run-only test.
   - The ReFrame-generated job script, if not a compile-only test.
   - The standard output and standard error of the run phase, if not a compile-only test.
   - Any additional files specified by the :attr:`keep_files` regression test attribute.

   This option can also be set using the :envvar:`RFM_OUTPUT_DIR` environment variable or the :attr:`~systems.outputdir` system configuration parameter.

.. option:: --perflogdir=DIR

   Directory prefix for logging performance data.

   This option is relevant only to the ``filelog`` :ref:`logging handler <filelog-handler>`.

   This option can also be set using the :envvar:`RFM_PERFLOG_DIR` environment variable or the :attr:`~config.logging.handlers_perflog..filelog..basedir` logging handler configuration parameter.

.. option:: --prefix=DIR

   General directory prefix for ReFrame-generated directories.

   The base stage and output directories (see below) will be specified relative to this prefix if not specified explicitly.

   This option can also be set using the :envvar:`RFM_PREFIX` environment variable or the :attr:`~config.systems.prefix` system configuration parameter.

.. option:: --report-file=FILE

   The file where ReFrame will store its report.

   The ``FILE`` argument may contain the special placeholder ``{sessionid}``, in which case ReFrame will generate a new report each time it is run by appending a counter to the report file.
   If the report is generated in the default location (see the :attr:`~config.general.report_file` configuration option), a symlink to the latest report named ``latest.json`` will also be created.

   This option can also be set using the :envvar:`RFM_REPORT_FILE` environment variable or the :attr:`~config.general.report_file` general configuration parameter.

   .. versionadded:: 3.1

   .. versionadded:: 4.2
      Symlink to the latest report is now created.

.. option:: --report-junit=FILE

   Instruct ReFrame to generate a JUnit XML report in ``FILE``.

   The generated report adheres to the XSD schema `here <https://github.com/windyroad/JUnit-Schema/blob/master/JUnit.xsd>`__ where each retry is treated as an individual testsuite.

   This option can also be set using the :envvar:`RFM_REPORT_JUNIT` environment variable or the :attr:`~config.general.report_junit` general configuration parameter.

   .. versionadded:: 3.6.0

   .. versionchanged:: 3.6.1
      Added support for retries in the JUnit XML report.

.. option:: -s, --stage=DIR

   Directory prefix for staging test resources.

   ReFrame does not execute tests from their original source directory.
   Instead it creates a test-specific stage directory and copies all test resources there.
   It then changes to that directory and executes the test.
   This test-specific directory is of the form ``{stage_prefix}/{system}/{partition}/{environment}/{test_name}``,
   where ``stage_prefix`` is set by this option.
   If a test finishes successfully, its stage directory will be removed.

   This option can also be set using the :envvar:`RFM_STAGE_DIR` environment variable or the :attr:`~config.systems.stagedir` system configuration parameter.

.. option:: --save-log-files

   Save ReFrame log files in the output directory before exiting.

   Only log files generated by ``file`` :ref:`log handlers <file-handler>` will be copied.

   This option can also be set using the :envvar:`RFM_SAVE_LOG_FILES` environment variable or the :attr:`~config.general.save_log_files` general configuration parameter.

.. option:: --timestamp [TIMEFMT]

   Append a timestamp to the output and stage directory prefixes.

   ``TIMEFMT`` can be any valid :manpage:`strftime(3)` time format.
   If not specified, ``TIMEFMT`` is set to ``%FT%T``.

   This option can also be set using the :envvar:`RFM_TIMESTAMP_DIRS` environment variable or the :attr:`~config.general.timestamp_dirs` general configuration parameter.


Options controlling ReFrame execution
-------------------------------------

.. option:: --disable-hook=HOOK

   Disable the pipeline hook named ``HOOK`` from all the tests that will run.

   This feature is useful when you have implemented test workarounds as pipeline hooks, in which case you can quickly disable them from the command line.
   This option may be specified multiple times in order to disable multiple hooks at the same time.

   .. versionadded:: 3.2

.. option:: --duration=TIMEOUT

   Run the test session repeatedly until the specified timeout expires.

   ``TIMEOUT`` can be specified in one of the following forms:

   - ``<int>`` or ``<float>``: number of seconds
   - ``<days>d<hours>h<minutes>m<seconds>s``: a string denoting days, hours, minutes and/or seconds.

   At the end, failures from every run will be reported and, similarly, the failure statistics printed by the :option:`--failure-stats` option will include all runs.

   .. versionadded:: 4.2


.. option:: --exec-order=ORDER

   Impose an execution order for the independent tests.
   The ``ORDER`` argument can take one of the following values:

   - ``name``: Order tests by their display name.
   - ``rname``: Order tests by their display name in reverse order.
   - ``uid``: Order tests by their unique name.
   - ``ruid``: Order tests by their unique name in reverse order.
   - ``random``: Randomize the order of execution.

   If this option is not specified the order of execution of independent tests is implementation defined.
   This option can be combined with any of the listing options (:option:`-l` or :option:`-L`) to list the tests in the order.

   .. versionadded:: 4.0.0

.. option:: --exec-policy=POLICY

   The execution policy to be used for running tests.

   There are two policies defined:

   - ``serial``: Tests will be executed sequentially.
   - ``async``: Tests will be executed asynchronously.
     This is the default policy.

     The ``async`` execution policy executes the build and run phases of tests asynchronously by submitting their associated jobs in a non-blocking way.
     ReFrame's runtime monitors the progress of each test and will resume the pipeline execution of an asynchronously spawned test as soon as its build or run phase have finished.
     Note that the rest of the pipeline stages are still executed sequentially in this policy.

     Concurrency can be controlled by setting the :attr:`~config.systems.partitions.max_jobs` system partition configuration parameter.
     As soon as the concurrency limit is reached, ReFrame will first poll the status of all its pending tests to check if any execution slots have been freed up.
     If there are tests that have finished their build or run phase, ReFrame will keep pushing tests for execution until the concurrency limit is reached again.
     If no execution slots are available, ReFrame will throttle job submission.

.. option:: --max-retries=NUM

   The maximum number of times a failing test can be retried.

   The test stage and output directories will receive a ``_retry<N>`` suffix every time the test is retried.

.. option:: --maxfail=NUM

   The maximum number of failing test cases before the execution is aborted.

   After ``NUM`` failed test cases the rest of the test cases will be aborted.
   The counter of the failed test cases is reset to 0 in every retry.

.. option:: --mode=MODE

   ReFrame execution mode to use.

   An execution mode is simply a predefined set of options that is set in the :attr:`~modes` :ref:`configuration parameter <exec-mode-config>`.
   Additional options can be passed to the command line, in which case they will be combined with the options defined in the selected execution mode.
   More specifically, any additional ReFrame options will be *appended* to the command line options of the selected mode.
   As a result, if a normal option is specified both inside the execution mode and the in the command line, the command line option will take precedence.
   On the other hand, if an option that is allowed to be specified multiple times, e.g., the :option:`-S` option, is passed both inside the execution mode and in the command line, their values will be combined.
   For example, if the execution mode ``foo`` defines ``-S modules=foo``, the invocation ``--mode=foo -S num_tasks=10`` is the equivalent of ``-S modules=foo -S num_tasks=10``.

   .. versionchanged:: 4.1
      Options that can be specified multiple times are now combined between execution modes and the command line.

   .. versionchanged:: 4.7
      The :option:`--mode` must always be combined with a :ref:`command option <commands>`.
      If the mode contains a command option already, the command option that will finally take effect is implementation defined.

   .. versionchanged:: 4.8
      Command options are disallowed from execution modes.


.. option:: --reruns=N

   Rerun the whole test session ``N`` times.

   In total, the selected tests will run ``N+1`` times as the first time does not count as a rerun.

   At the end, failures from every run will be reported and, similarly, the failure statistics printed by the :option:`--failure-stats` option will include all runs.

   Although similar to :option:`--repeat`, this option behaves differently.
   This option repeats the *whole* test session multiple times.
   All the tests of the session will finish before a new run is started.
   The :option:`--repeat` option on the other hand generates clones of the selected tests and schedules them for running in a single session.
   As a result, all the test clones will run (by default) concurrently.

   .. versionadded:: 4.2


.. option:: --restore-session [REPORT1[,REPORT2,...]]

   Restore a testing session that has run previously.

   ``REPORT1`` etc. are a run report files generated by ReFrame.
   If a report is not given, ReFrame will pick the last report file found in the default location of report files (see the :option:`--report-file` option).
   If passed alone, this option will simply rerun all the test cases that have run previously based on the report file data.
   It is more useful to combine this option with any of the `test filtering <#test-filtering>`__ options, in which case only the selected test cases will be executed.
   The difference in test selection process when using this option is that the dependencies of the selected tests will not be selected for execution, as they would normally, but they will be restored.
   For example, if test ``T1`` depends on ``T2`` and ``T2`` depends on ``T3``, then running ``reframe -n T1 -r`` would cause both ``T2`` and ``T3`` to run.
   However, by doing ``reframe -n T1 --restore-session -r``, only ``T1`` would run and its immediate dependence ``T2`` will be restored.
   This is useful when you have deep test dependencies or some of the tests in the dependency chain are very time consuming.

   Multiple reports may be passed as a comma-separated list.
   ReFrame will try to restore any required test case by looking it up in each report sequentially.
   If it cannot find it, it will issue an error and exit.

   .. note::
      In order for a test case to be restored, its stage directory must be present.
      This is not a problem when rerunning a failed case, since the stage directories of its dependencies are automatically kept, but if you want to rerun a successful test case, you should make sure to have run with the :option:`--keep-stage-files` option.

   .. note::
      This option will not work with the :ref:`test generation options <test-generators>`.

   .. versionadded:: 3.4

   .. versionchanged:: 3.6.1
      Multiple report files are now accepted.


.. option:: --retries-threshold=VALUE[%]

   Skip retries (see :option:`--max-retries`) if failures exceed the given threshold.

   Threshold can be specified either as an absolute value or as a percentage using the ``%`` character, e.g., ``--retries-threshold=30%``.
   Note that in certain shells the ``%`` character may need to be escaped.

   .. versionadded:: 4.7


.. option:: -S, --setvar=[TEST.]VAR=VAL

   Set variable ``VAR`` in all tests or optionally only in test ``TEST`` to ``VAL``.

   ``TEST`` can have the form ``[TEST.][FIXT.]*``, in which case ``VAR`` will be set in fixture ``FIXT`` of ``TEST``.
   Note that this syntax is recursive on fixtures, so that a variable can be set in a fixture arbitrarily deep.
   ``TEST`` prefix refers to the test class name, *not* the test name and ``FIXT`` refers to the fixture *variable name* inside the referenced test, i.e., the test variable to which the fixture is bound.
   The fixture variable name is referred to as ``'<varname>`` when listing tests with the :option:`-l` and :option:`-L` options.

   Multiple variables can be set at the same time by passing this option multiple times.
   This option *cannot* change arbitrary test attributes, but only test variables declared with the :attr:`~reframe.core.pipeline.RegressionMixin.variable` built-in.
   If an attempt is made to change an inexistent variable or a test parameter, a warning will be issued.

   ReFrame will try to convert ``VAL`` to the type of the variable.
   If it does not succeed, a warning will be issued and the variable will not be set.
   ``VAL`` can take the special value ``@none`` to denote that the variable must be set to :obj:`None`.
   Boolean variables can be set in one of the following ways:

   - By passing ``true``, ``yes`` or ``1`` to set them to :class:`True`.
   - By passing ``false``, ``no`` or ``0`` to set them to :class:`False`.

   Passing any other value will issue an error.

   .. note::

      Boolean variables in a test must be declared of type :class:`~reframe.utility.typecheck.Bool` and *not* of the built-in :class:`bool` type, in order to adhere to the aforementioned behaviour.
      If a variable is defined as :class:`bool` there is no way you can set it to :obj:`False`, since all strings in Python evaluate to :obj:`True`.

   Sequence and mapping types can also be set from the command line by using the following syntax:

   - Sequence types: ``-S seqvar=1,2,3,4``
   - Mapping types: ``-S mapvar=a:1,b:2,c:3``

   They can also be converted using JSON syntax.
   For example, the :attr:`~reframe.core.pipeline.RegressionTest.extra_resources` complex dictionary could be set with ``-S extra_resources='{"gpu": {"num_gpus_per_node":8}}'``.

   Conversions to arbitrary objects are also supported.
   See :class:`~reframe.utility.typecheck.ConvertibleType` for more details.

   Variable assignments passed from the command line happen *before* the test is instantiated and is the exact equivalent of assigning a new value to the variable *at the end* of the test class body.
   This has a number of implications that users of this feature should be aware of:

   - In the following test, :attr:`num_tasks` will have always the value ``1`` regardless of any command-line assignment of the variable :attr:`foo`:

   .. code-block:: python

      @rfm.simple_test
      class my_test(rfm.RegressionTest):
          foo = variable(int, value=1)
          num_tasks = foo

   .. tip::

     In cases where the class body expresses logic as a function of a variable and this variable, as well as its dependent logic, need to be controlled externally, the variable's default value (i.e. the value set through the value argument) may be modified as follows through an environment variable and not through the `-S` option:

     .. code-block:: python

      import os

      @rfm.simple_test
      class my_test(rfm.RegressionTest):
          max_nodes = variable(int, value=int(os.getenv('MAX_NODES', 1)))
          # Parameterise number of nodes
          num_nodes = parameter((1 << i for i in range(0, int(max_nodes))))

   - If the variable is set in any pipeline hook, the command line assignment will have an effect until the variable assignment in the pipeline hook is reached.
     The variable will be then overwritten.
   - The `test filtering <#test-filtering>`__ happens *after* a test is instantiated, so the only way to scope a variable assignment is to prefix it with the test class name.
     However, this has some positive side effects:

     - Passing ``-S valid_systems='*'`` and ``-S valid_prog_environs='*'`` is the equivalent of passing the :option:`--skip-system-check` and :option:`--skip-prgenv-check` options.
     - Users could alter the behavior of tests based on tag values that they pass from the command line, by changing the behavior of a test in a post-init hook based on the value of the :attr:`~reframe.core.pipeline.RegressionTest.tags` attribute.
     - Users could force a test with required variables to run if they set these variables from the command line.
       For example, the following test could only be run if invoked with ``-S num_tasks=<NUM>``:

     .. code-block:: python

        @rfm.simple_test
        class my_test(rfm.RegressionTest):
            num_tasks = required

   .. versionadded:: 3.8.0

   .. versionchanged:: 3.9.3

      Proper handling of boolean variables.

   .. versionchanged:: 3.11.1

      Allow setting variables in fixtures.

   .. versionchanged:: 4.4

      Allow setting nested mapping types using JSON syntax.

   .. versionchanged:: 4.8

      Allow setting sequence types using JSON syntax.

.. option:: --skip-performance-check

   Skip performance checking phase.

   The phase is completely skipped, meaning that performance data will *not* be logged.

.. option:: --skip-sanity-check

   Skip sanity checking phase.


Options controlling job submission
----------------------------------

.. option:: -J, --job-option=OPTION

   Pass ``OPTION`` directly to the job scheduler backend.

   The syntax of ``OPTION`` is ``-J key=value``.
   If ``OPTION`` starts with ``-`` it will be passed verbatim to the backend job scheduler.
   If ``OPTION`` starts with ``#`` it will be emitted verbatim in the job script.
   Otherwise, ReFrame will pass ``--key value`` or ``-k value`` (if ``key`` is a single character) to the backend scheduler.
   Any job options specified with this command-line option will be emitted after any job options specified in the :attr:`~config.systems.partitions.access` system partition configuration parameter.

   Especially for the Slurm backends, constraint options, such as ``-J constraint=value``, ``-J C=value``, ``-J --constraint=value`` or ``-J -C=value``, are going to be combined with any constraint options specified in the :attr:`~config.systems.partitions.access` system partition configuration parameter.
   For example, if ``-C x`` is specified in the :attr:`~config.systems.partitions.access` and ``-J C=y`` is passed to the command-line, ReFrame will pass ``-C x&y`` as a constraint to the scheduler.
   Notice, however, that if constraint options are specified through multiple :option:`-J` options, only the last one will be considered.
   If you wish to completely overwrite any constraint options passed in :attr:`~config.systems.partitions.access`, you should consider passing explicitly the Slurm directive with ``-J '#SBATCH --constraint=new'``.

   .. versionchanged:: 3.0
      This option has become more flexible.

   .. versionchanged:: 3.1
      Use ``&`` to combine constraints.


Options controlling flexible node allocation
--------------------------------------------

ReFrame can automatically set the number of tasks of a test, if its :attr:`num_tasks <reframe.core.pipeline.RegressionTest.num_tasks>` attribute is set to a value less than or equal to zero.
This scheme is conveniently called *flexible node allocation* and is valid only for the Slurm backend.
When allocating nodes automatically, ReFrame will take into account all node limiting factors, such as partition :attr:`~config.systems.partitions.access` options, and any job submission control options described above.
Particularly for Slurm constraints, ReFrame will only recognize simple AND or OR constraints and any parenthesized expression of them.
The full syntax of `Slurm constraints <https://slurm.schedmd.com/sbatch.html#OPT_constraint>`__ is not currently supported.

Nodes from this pool are allocated according to different policies.
If no node can be selected, the test will be marked as a failure with an appropriate message.

.. option:: --flex-alloc-nodes=POLICY

   Set the flexible node allocation policy.

   Available values are the following:

   - Any of the values supported by the :option:`--distribute` option.
   - Any positive integer: flexible tests will be assigned as many tasks as needed in order to span over the specified number of nodes from the node pool.

   .. versionchanged:: 3.1
      It is now possible to pass an arbitrary node state as a flexible node allocation parameter.

   .. versionchanged:: 4.6
      Align the state selection with the :option:`--distribute` option.
      See the :option:`--distribute` for more details.

      Slurm OR constraints and parenthesized expressions are supported in flexible node allocation.

   .. versionchanged:: 4.7
      The test is not marked as a failure if not enough nodes are available, but it is skipped instead.
      To enforce a failure, use :option:`--flex-alloc-strict`

.. option:: --flex-alloc-strict

   Fail flexible tests if their minimum task requirement is not satisfied.
   Otherwise the tests will be skipped.

   .. versionadded:: 4.7


Options controlling ReFrame environment
---------------------------------------

ReFrame offers the ability to dynamically change its environment as well as the environment of tests.
It does so by leveraging the selected system's environment modules system.

.. option:: -M, --map-module=MAPPING

   Apply a module mapping.

   ReFrame allows manipulating test modules on-the-fly using module mappings.
   A module mapping has the form ``old_module: module1 [module2]...`` and will cause ReFrame to replace a module with another list of modules upon load time.
   For example, the mapping ``foo: foo/1.2`` will load module ``foo/1.2`` whenever module ``foo`` needs to be loaded.
   A mapping may also be self-referring, e.g., ``gnu: gnu gcc/10.1``, however cyclic dependencies in module mappings are not allowed and ReFrame will issue an error if it detects one.
   This option is especially useful for running tests using a newer version of a software or library.

   This option may be specified multiple times, in which case multiple mappings will be applied.

   This option can also be set using the :envvar:`RFM_MODULE_MAPPINGS` environment variable or the :attr:`~config.general.module_mappings` general configuration parameter.

   .. versionchanged:: 3.3
      If the mapping replaces a module collection, all new names must refer to module collections, too.

   .. seealso::
      Module collections with `Environment Modules <https://modules.readthedocs.io/en/latest/MIGRATING.html#module-collection>`__ and `Lmod <https://lmod.readthedocs.io/en/latest/010_user.html#user-collections>`__.

.. option:: -m, --module=NAME

   Load environment module ``NAME`` before acting on any tests.

   This option may be specified multiple times, in which case all specified modules will be loaded in order.
   ReFrame will *not* perform any automatic conflict resolution.

   This option can also be set using the :envvar:`RFM_USER_MODULES` environment variable or the :attr:`~config.general.user_modules` general configuration parameter.

.. option:: --module-mappings=FILE

   A file containing module mappings.

   Each line of the file contains a module mapping in the form described in the :option:`-M` option.
   This option may be combined with the :option:`-M` option, in which case module mappings specified will be applied additionally.

   This option can also be set using the :envvar:`RFM_MODULE_MAP_FILE` environment variable or the :attr:`~config.general.module_map_file` general configuration parameter.

.. option:: --module-path=PATH

   Manipulate the ``MODULEPATH`` environment variable before acting on any tests.

   If ``PATH`` starts with the ``-`` character, it will be removed from the ``MODULEPATH``, whereas if it starts with the ``+`` character, it will be added to it.
   In all other cases, ``PATH`` will completely override MODULEPATH.
   This option may be specified multiple times, in which case all the paths specified will be added or removed in order.

   .. versionadded:: 3.3

.. option:: --non-default-craype

   Test a non-default Cray Programming Environment.

   Since CDT 19.11, this option can be used in conjunction with :option:`-m`, which will load the target CDT.
   For example:

   .. code:: bash

      reframe -m cdt/20.03 --non-default-craype -r

   This option causes ReFrame to properly set the ``LD_LIBRARY_PATH`` for such cases.
   It will emit the following code after all the environment modules of a test have been loaded:

   .. code:: bash

     export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH

   This option can also be set using the :envvar:`RFM_NON_DEFAULT_CRAYPE` environment variable or the :attr:`~config.general.non_default_craype` general configuration parameter.

.. option:: --purge-env

   Unload all environment modules before acting on any tests.

   This will unload also sticky Lmod modules.

   This option can also be set using the :envvar:`RFM_PURGE_ENVIRONMENT` environment variable or the :attr:`~config.general.purge_environment` general configuration parameter.

.. option:: -u, --unload-module=NAME

   Unload environment module ``NAME`` before acting on any tests.

   This option may be specified multiple times, in which case all specified modules will be unloaded in order.

   This option can also be set using the :envvar:`RFM_UNLOAD_MODULES` environment variable or the :attr:`~config.general.unload_modules` general configuration parameter.


.. _test-generators:

Options for generating tests dynamically
----------------------------------------

These options generate *new* tests dynamically from a set of previously `selected <#test-filtering>`__ tests.
The way the tests are generated and how they interact with the test filtering options poses some limitations:

1. These tests do not have an associated test file and are *different* from their original tests although the share the same base name.
   As a result, the :option:`--restore-session` option cannot be used to restore dynamically generated tests.
2. Since these tests are generated after the test selection phase, the :option:`--ci-generate` option cannot be used to generate a child pipeline, as the child pipeline uses the :option:`-n` option to select the tests for running.


.. option:: --distribute[=NODESTATE]

   Distribute the selected tests on all the nodes in state ``NODESTATE`` in their respective valid partitions.

   ReFrame will parameterize and run the tests on the selected nodes.
   Effectively, it will dynamically create new tests that inherit from the original tests and add a new parameter named ``$nid`` which contains the list of nodes that the test must run on.
   The new tests are named with the following pattern  ``{orig_test_basename}_{partition_fullname}``.

   When determining the list of nodes to distribute the selected tests, ReFrame will take into account any job options passed through the :option:`-J` option.

   You can optionally specify the state of the nodes to consider when distributing the test through the ``NODESTATE`` argument:

   - ``all``: Tests will run on all the nodes of their respective valid partitions regardless of the node state.
   - ``avail``: Tests will run on all the nodes of their respective valid partitions that are available for running jobs.
     Note that if a node is currently allocated to another job it is still considered as "available."
     Also, for ReFrame partitions using the Slurm backends, if this option is used on a reservation with the ``MAINT`` flag set, then nodes in ``MAINTENANCE`` state will also be considered as available.
   - ``NODESTATE``: Tests will run on all the nodes of their respective valid partitions that are exclusively in state ``NODESTATE``.
     If ``NODESTATE`` is not specified, ``idle`` is assumed.
   - ``NODESTATE*``: Tests will run on all the nodes of their respective valid partitions that are at least in state ``NODESTATE``.

   The state of the nodes will be determined once, before beginning the
   execution of the tests, so it might be different at the time the tests are actually submitted.

   .. note::
      Currently, only single-node jobs can be distributed and only local or the Slurm-based backends support this feature.

   .. note::
      Distributing tests with dependencies is not supported, but you can distribute tests that use fixtures.

   .. note::
      This option is supported only for the ``local``, ``squeue``, ``slurm`` and ``ssh`` scheduler backends.

   .. versionadded:: 3.11.0

   .. versionadded:: 4.6

      The ``avail`` argument is introduced and the ability to differentiate between exclusive and non-exclusive node states.

   .. versionchanged:: 4.6

      ``--distribute=NODESTATE`` now matches nodes that are exclusively in state ``NODESTATE``, so that the default ``--distribute=idle`` will match only the Slurm nodes that are in the ``IDLE`` state exclusively.
      To achieve the previous behaviour, you should use ``--distribute=idle*``.

   .. versionchanged:: 4.9

      ``--distribute=NODESTATE`` now allows you to specify multiple valid states using the ``|`` character.

   .. versionchanged:: 4.10

      Nodes in ``MAINTENANCE`` state are considered available, if this option is run on a Slurm reservation with the ``MAINT`` flag set.

.. option:: -P, --parameterize=[TEST.]VAR=VAL0,VAL1,...

   Parameterize a test on an existing variable or parameter.

   In case of variables, the test will behave as if the variable ``VAR`` was a parameter taking the values ``VAL0,VAL1,...``.
   The values will be converted based on the type of the target variable ``VAR``.
   In case of parameters, the test will behave is the parameter had been defined with the values ``VAL0,VAL1,...``.
   The ``TEST.`` prefix will only parameterize the variable ``VAR`` of test ``TEST``.

   The :option:`-P` can be specified multiple times in order to parameterize multiple variables or redefine multiple parameters.

   .. note::

      Conversely to the :option:`-S` option that can set a variable in an arbitrarily nested fixture,
      the :option:`-P` option can only parameterize the leaf test:
      it cannot be used to parameterize a fixture of the test.

   .. note::

      The :option:`-P` option supports only tests that use fixtures.
      Tests that use raw dependencies are not supported.

   .. versionadded:: 4.3

   .. versionchanged:: 4.9

      It is now possible to use the :option:`-P` option to redefine existing test parameters.

.. option:: --param-values-delim=<delim>

   Use the given delimiter to separate the parameter values passed with :option:`--parameterize`.

   Default delimiter is ``,``.

   .. versionadded:: 4.9

.. option:: --repeat=N

   Repeat the selected tests ``N`` times.
   This option can be used in conjunction with the :option:`--distribute` option in which case the selected tests will be repeated multiple times and distributed on individual nodes of the system's partitions.

   .. note::
      Repeating tests with dependencies is not supported, but you can repeat tests that use fixtures.

   .. versionadded:: 3.12.0


Miscellaneous options
---------------------

.. option:: -C, --config-file=FILE

   Use ``FILE`` as configuration file for ReFrame.

   This option can be passed multiple times, in which case multiple configuration files will be read and loaded successively.
   The base of the configuration chain is always the :ref:`builtin configuration file <builtin-configuration>`, namely the ``${RFM_INSTALL_PREFIX}/reframe/core/settings.py``.
   At any point, the user can "break" the chain of configuration files by prefixing the configuration file name with a colon as in the following example: ``-C :/path/to/new_config.py``.
   This will ignore any previously loaded configuration file and will only load the one specified.
   Note, however, that the builtin configuration file cannot be overriden;
   It will always be loaded first in the chain.

   This option can also be set using the :envvar:`RFM_CONFIG_FILES` environment variable.

   In order to determine its final configuration, ReFrame first loads the builtin configuration file unconditionally and then starts looking for possible configuration file locations defined in the :envvar:`RFM_CONFIG_PATH` environment variable.
   For each directory defined in the :envvar:`RFM_CONFIG_PATH`, ReFrame looks for a file named ``settings.py`` or ``settings.json`` inside it and loads it.
   If both a ``settings.py`` and a ``settings.json`` files are found, the Python configuration will be preferred.
   ReFrame, finally, processes any configuration files specified in the command line or in the :envvar:`RFM_CONFIG_FILES` environment variable.

   .. versionchanged:: 4.0.0

.. option:: --failure-stats

   Print failure statistics at the end of the run.

.. option:: -h, --help

   Print a short help message and exit.

.. option:: --nocolor

   Disable output coloring.

   This option can also be set using the :envvar:`RFM_COLORIZE` environment variable or the :attr:`~config.general.colorize` general configuration parameter.

.. _--performance-report:

.. option:: --performance-report[=CMPSPEC]

   Print a report summarizing the performance of all performance tests that have run in the current session.

   For each test all of their performance variables are reported and optionally compared to past results based on the ``CMPSPEC`` specified.
   If not specified, ``CMPSPEC`` defaults to ``now:now/last:/+job_nodelist+result``, meaning that the current performance will not be compared to any past run and, additionally, the ``job_nodelist`` and the test result (``pass`` or ``fail``) will be listed.

   For the exact syntax of ``CMPSPEC``, refer to :ref:`querying-past-results`.

   .. versionchanged:: 4.7

      The format of the performance report has changed and the optional ``CMPSPEC`` argument is now added.

.. option:: -q, --quiet

   Decrease the verbosity level.

   This option can be specified multiple times.
   Every time this option is specified, the verbosity level will be decreased by one.
   This option can be combined arbitrarily with the :option:`-v` option, in which case the final verbosity level will be determined by the final combination.
   For example, specifying ``-qv`` will not change the verbosity level, since the two options cancel each other, but ``-qqv`` is equivalent to ``-q``.
   For a list of ReFrame's verbosity levels, see the description of the :option:`-v` option.

   .. versionadded:: 3.9.3


.. option:: --session-extras KV_DATA

   Annotate the current session with custom key/value metadata.

   The key/value data is specified as a comma-separated list of `key=value` pairs.
   When listing stored sessions with the :option:`--list-stored-sessions` option, any associated custom metadata will be presented.

   This option can be specified multiple times, in which case the data from all options will be combined in a single list of key/value data.

   .. versionadded:: 4.7


.. option:: --system=NAME

   Load the configuration for system ``NAME``.

   The ``NAME`` must be a valid system name in the configuration file.
   It may also have the form ``SYSNAME:PARTNAME``, in which case the configuration of system ``SYSNAME`` will be loaded, but as if it had ``PARTNAME`` as its sole partition.
   Of course, ``PARTNAME`` must be a valid partition of system ``SYSNAME``.
   If this option is not specified, ReFrame will try to pick the correct configuration entry automatically.
   It does so by trying to match the hostname of the current machine again the hostname patterns defined in the :attr:`~config.systems.hostnames` system configuration parameter.
   The system with the first match becomes the current system.

   This option can also be set using the :envvar:`RFM_SYSTEM` environment variable.

.. option:: --table-format=csv|plain|pretty

   Set the formatting of tabular output printed by the options :option:`--performance-compare`, :option:`--performance-report` and the options controlling the stored sessions.

   The acceptable values are the following:

   - ``csv``: Generate CSV output
   - ``plain``: Generate a plain table without any vertical lines allowing for easy ``grep``-ing
   - ``pretty``: (default) Generate a pretty table

   .. versionadded:: 4.7

.. option:: --table-format-delim[=DELIM]

   Delimiter to use when emitting tables in CSV format using the :option:`--table-format=csv` option.

   The default delimiter is ``,``.

   .. versionadded:: 4.9

.. option:: --upgrade-config-file=OLD[:NEW]

   Convert the old-style configuration file ``OLD``, place it into the new file ``NEW`` and exit.

   If a new file is not given, a file in the system temporary directory will be created.

.. option:: -v, --verbose

   Increase verbosity level of output.

   This option can be specified multiple times.
   Every time this option is specified, the verbosity level will be increased by one.
   There are the following message levels in ReFrame listed in increasing verbosity order:
   ``critical``, ``error``, ``warning``, ``info``, ``verbose`` and ``debug``.
   The base verbosity level of the output is defined by the :attr:`~config.logging.handlers.level` stream logging handler configuration parameter.

   This option can also be set using the :envvar:`RFM_VERBOSE` environment variable or the :attr:`~config.general.verbose` general configuration parameter.


.. _test_naming_scheme:

Test Naming Scheme
==================

.. versionadded:: 3.10.0

This section describes the test naming scheme.
This scheme has superseded the old one in ReFrame 4.0.

Each ReFrame test is assigned a unique name, which will be used internally by the framework to reference the test.
Any test-specific path component will use that name, too.
It is formed as follows for the various types of tests:

- *Regular tests*: The unique name is simply the test class name.
  This implies that you cannot load two tests with the same class name within the same run session even if these tests reside in separate directories.
- *Parameterized tests*: The unique name is formed by the test class name followed by an ``_`` and the variant number of the test.
  Each point in the parameter space of the test is assigned a unique variant number.
- *Fixtures*: The unique name is formed by the test class name followed by an ``_`` and a hash.
  The hash is constructed by combining the information of the fixture variant (if the fixture is parameterized), the fixture's scope and any fixture variables that were explicitly set.

Since unique names can be cryptic, they are not listed by the :option:`-l` option, but are listed when a detailed listing is requested by using the :option:`-L` option.

A human readable version of the test name, which is called the *display name*, is also constructed for each test.
This name encodes all the parameterization information as well as the fixture-specific information (scopes, variables).
The format of the display name is the following in BNF notation:

.. code-block:: bnf

   <display_name> ::= <test_class_name> (<params>)* (<scope> ("'"<fixtvar>)+)?
   <params> ::= "%" <parametrization> "=" <pvalue>
   <parametrization> ::= (<fname> ".")* <pname>
   <scope> ::= "~" <scope_descr>
   <scope_descr> ::= <first> ("+" <second>)*

   <test_class_name> ::= (* as in Python *)
   <fname> ::= (* string *)
   <pname> ::= (* string *)
   <pvalue> ::= (* string *)
   <first> ::= (* string *)
   <second> ::= (* string *)
   <fixtvar> ::= (* string *)

The following is an example of a fictitious complex test that is itself parameterized and depends on parameterized fixtures as well.

.. code-block:: python

   import reframe as rfm


   class MyFixture(rfm.RunOnlyRegressionTest):
       p = parameter([1, 2])


   class X(rfm.RunOnlyRegressionTest):
       foo = variable(int, value=1)


   @rfm.simple_test
   class TestA(rfm.RunOnlyRegressionTest):
       f = fixture(MyFixture, scope='test', action='join')
       x = parameter([3, 4])
       t = fixture(MyFixture, scope='test')
       l = fixture(X, scope='environment', variables={'foo': 10})
       valid_systems = ['*']
       valid_prog_environs = ['*']


Here is how this test is listed where the various components of the display name can be seen:

.. code-block:: console

   - TestA %x=4 %l.foo=10 %t.p=2 /8804be5d
       ^MyFixture %p=1 ~TestA_3 't 'f /f027ee75
       ^MyFixture %p=2 ~TestA_3 't 'f /830323a4
       ^X %foo=10 ~generic:default+builtin 'l /7dae3cc5
   - TestA %x=3 %l.foo=10 %t.p=2 /89f6f5d1
       ^MyFixture %p=1 ~TestA_2 't 'f /02368516
       ^MyFixture %p=2 ~TestA_2 't 'f /854b99b5
       ^X %foo=10 ~generic:default+builtin 'l /7dae3cc5
   - TestA %x=4 %l.foo=10 %t.p=1 /af9b2941
       ^MyFixture %p=2 ~TestA_1 't 'f /f0383f7f
       ^MyFixture %p=1 ~TestA_1 't 'f /d07f4281
       ^X %foo=10 ~generic:default+builtin 'l /7dae3cc5
   - TestA %x=3 %l.foo=10 %t.p=1 /a9e50aa3
       ^MyFixture %p=2 ~TestA_0 't 'f /b894ab05
       ^MyFixture %p=1 ~TestA_0 't 'f /ca376ca8
       ^X %foo=10 ~generic:default+builtin 'l /7dae3cc5
   Found 4 check(s)

Notice that the variable name to which every fixture is bound in its parent test is also listed as ``'<varname>``.
This is useful for setting variables down the fixture hierarchy using the :option:`-S` option.



Display names may not always be unique.
Assume the following test:

.. code-block:: python

   class MyTest(RegressionTest):
       p = parameter([1, 1, 1])

This generates three different tests with different unique names, but their display name is the same for all: ``MyTest %p=1``.
Notice that this example leads to a name conflict with the old naming scheme, since all tests would be named ``MyTest_1``.

Each test is also associated with a hash code that is derived from the test name, its parameters and their values.
As in the example listing above, the hash code of each test is printed with the :option:`-l` option and individual tests can be selected by their hash using the :option:`-n` option, e.g., ``-n /1c51609b``.
The stage and output directories, as well as the performance log file of the ``filelog`` :ref:`performance log handler <filelog-handler>` will use the hash code for the test-specific directories and files.
This might lead to conflicts for tests as the one above when executing them with the asynchronous execution policy, but ensures consistency of performance record files when parameter values are added to or deleted from a test parameter.
More specifically, the test's hash will not change if a new parameter value is added or deleted or even if the parameter values are shuffled.
Test variants on the other side are more volatile and can change with such changes.
Also users should not rely on how the variant numbers are assigned to a test, as this is an implementation detail.


.. versionchanged:: 4.0.0

   A hash code is associated with each test.


Differences from the old naming scheme
--------------------------------------

Prior to version 3.10, ReFrame used to encode the parameter values of an instance of parameterized test in its name.
It did so by taking the string representation of the value and replacing any non-alphanumeric character with an underscore.
This could lead to very large and hard to read names when a test defined multiple parameters or the parameter type was more complex.
Very large test names meant also very large path names which could also lead to problems and random failures.
Fixtures followed a similar naming pattern making them hard to debug.


Result storage
==============

.. versionadded:: 4.7

ReFrame stores the results of every session that has executed at least one test into a database.
There is only one storage backend supported at the moment and this is SQLite.
The full session information as recorded in a run report file (see :option:`--report-file`) is stored in the database.
The test cases of the session are indexed by their run job completion time for quick retrieval of all the test cases that have run in a certain period of time.

The database file is controlled by the :attr:`~config.storage.sqlite_db_file` configuration parameter and multiple ReFrame processes can access it safely simultaneously.

There are several command-line options that allow users to query the results database, such as the :option:`--list-stored-sessions`, :option:`--list-stored-testcases`, :option:`--describe-stored-sessions` etc.
Other options that access the results database are the :option:`--performance-compare` and :option:`--performance-report` which compare the performance results of the same test cases in different periods of time or from different sessions.
Check the :ref:`commands` section for the complete list and details of each option related to the results database.

Since the report file information is now kept in the results database, there is no need to keep the report files separately, although this remains the default behavior for backward compatibility.
You can disable the report generation by turning off the :attr:`~config.general.generate_file_reports` configuration parameter.
The file report of any session can be retrieved from the database with the :option:`--describe-stored-sessions` option.

.. warning::

    ReFrame uses file locking to coordinate storing of run session data in the database file.
    Enabling the database storage on filesystems that do not support locking (e.g some networked filesystems) might lead to hangs at the end of a run session.
    For this reason, you must make sure that the database file is located on a filesystem that supports locking.
    You can set the database location through the :attr:`~config.storage.sqlite_db_file` configuration setting or the :envvar:`RFM_SQLITE_DB_FILE` environment variable.


.. _querying-past-results:

Querying past results
=====================

.. versionadded:: 4.7

ReFrame provides several options for querying and inspecting past sessions and test case results.
All those options follow a common syntax that builds on top of the following elements:

1. Selection of sessions and test cases
2. Grouping of test cases and aggregations
3. Selection of test case attributes to present

Throughout the documentation, we use the ``<select>`` notation for (1), ``<aggr>`` for (2) and ``<cols>`` for (3).
For the options performing aggregations on test case performance we use the notation ``<cmpspec>`` which takes the following form:

.. _cmpspec-syntax:

.. code-block:: bnf

   <cmpspec> ::= (<select> "/")? <select> "/" <aggr> "/" <cols>

The first optional ``<select>`` is relevant only for the :option:`--performance-compare` option, where a selection of two test case sets is needed.
For the rest of the query options, including the :option:`--performance-report` option, which performs an implicit comparison, the single-select syntax is expected.
In case of performance comparisons, any attribute referring to the first selection group is referred to as left-hand-side or lhs or, simply, left, whereas every attribute referring to the second selection group is referred to as right-hand-side, rhs, or, simple, right.

In the following we present in detail the exact syntax of every of the above syntactic elements.

.. _session-selection:

Selecting sessions and test cases
----------------------------------

The syntax for selecting test cases in past results queries is the following:

.. code-block:: bnf

   <select> ::= <session_uuid> | <session_filter> | <time_period>
   <session_uuid> ::= /* any valid UUID */
   <session_filter> ::= (<time_period>)? "?" <python_expr>
   <python_expr> ::= /* any valid Python expression */

Test cases can be practically selected in three ways:

1. By an explicit session UUID, such as ``ae43e247-375f-4b05-8ab5-c7a017d4afc3``.
2. By a time period, such as ``20251201:now`` (the exact syntax of the time period is explained in :ref:`time-periods`).
3. By filter, which can either take the form of a pure Python expression or a Python expression prefixed by a time period.
   The expression is evaluated over the session information including any user-specific session extras (see also :option:`--session-extras`).
   Here are two examples:

   - ``?'tag=="123"'`` will select all stored sessions with ``tag`` set to ``123``.
   - ``20251201:now?'tag=="123"'`` will select stored sessions from December 2025 with ``tag`` set to ``123``.
     When filtering using an expression, it is a good idea to limit the scope of the query using a time period as this will reduce significantly the query times in large databases.

.. tip::

   When using session filters to select the test cases, quoting is important.
   If ``tag=="123"`` was used unquoted in the example above, the shell would remove the double quotes from ``"123"`` and the expression passed to ReFrame would be ``tag==123``.
   This is a valid expression but will always evaluate to false, since ``tag``, as every session attribute is a string.
   Single-quoting the expression avoids this and the actual comparison will be ``tag=="123"`` giving the desired outcome.


.. note::
   .. versionchanged:: 4.8

      Support for scoping the session filter queries by a time period was added.


.. _time-periods:

Time periods
^^^^^^^^^^^^

The syntax for defining time periods in past results queries is the following:

.. code-block:: bnf

   <time_period> ::= <timestamp> ":" <timestamp>
   <timestamp> ::= ("now" | <abs_timestamp>) (("+" | "-") <number> ("w" | "d" | "h" | "m"))?
   <abs_timestamp> ::= /* any timestamp of the format `%Y%m%d`, `%Y%m%dT%H%M`, `%Y%m%dT%H%M%S` */
   <number> ::= [0-9]+

A time period is defined as a starting and ending timestamps separated by colon.
A timestamp can have any of the following ``strptime``-compatible formats: ``%Y%m%d``, ``%Y%m%dT%H%M``, ``%Y%m%dT%H%M%S``, ``%Y%m%dT%H%M%S%z``.
A timestamp can also be the special value ``now`` which denotes the current local time.
Optionally, a shift argument can be appended with ``+`` or ``-`` signs, followed by the number of weeks (``w``), days (``d``), hours (``h``) or minutes (``m``).
For example, the period of the last 10 days can be specified as ``now-10d:now``.
Similarly, the period of 3 weeks starting on August 5, 2024 can be specified as ``20240805:20240805+3w``.

.. _testcase-grouping:

Groupings and aggregations
--------------------------

The aggregation specification follows the general syntax:

.. code-block:: bnf

   <aggr> ::= <aggr_list> ":" <cols>?

Where ``<aggr_list>`` is a list of aggregation specs and ``<cols>`` is a list of attributes to group the test cases.
An aggregation spec has the following general syntax:

.. code-block:: bnf

   <aggr_spec> ::= <aggr_any> | <aggr_pval>
   <aggr_any> ::= <aggr_fn> "(" <attr> ")"
   <aggr_pval> ::= <aggr_fn>

It can either a single aggregation function name or an aggregation function name followed by a test attribute in parenthesis, e.g., ``max`` or ``max(num_tasks)``.
A single function name as an aggregation is equivalent to ``fn(pval)``, i.e. the aggregation is applied to performance value, e.g., ``max`` is equivalent to ``max(pval)``.
The following aggregation functions are supported:

- ``first``: return the first element of every group
- ``last``: return the last element of every group
- ``max``: return the maximum value of every group
- ``mix``: return the minimu value of every group
- ``mean``: return the minimu value of every group
- ``std``: return the standard deviation of every group
- ``sum``: return the sum of every group
- ``median``: return the median of every group
- ``p01``: return the 1% percentile of every group
- ``p05``: return the 5% percentile of every group
- ``p95``: return the 95% percentile of every group
- ``p99``: return the 99% percentile of every group

There is also the pseudo-function ``stats``, which is essentially a shortcut for ``min,p01,p05,median,p95,p99,max,mean,std``.
It can also be applied to any other attribute than ``pval``

When performing aggregations, test cases are grouped by the following attributes by default:

- The test :attr:`~reframe.core.pipeline.RegressionTest.name`
- A unique combination of the system name, partition name and environment name, called ``sysenv``.
  ``sysenv`` is equivalent to ``<system>:<partition>+<environ>``.
- The performance variable name (see :func:`@performance_function <reframe.core.builtins.performance_function>` and :attr:`~reframe.core.pipeline.RegressionTest.perf_variables`)
- The performance variable unit

Note that if an aggregation is requested on an attribute, this attribute has to be present in the group-by list, otherwise ReFrame will complain.
For example, the following spec is problematic as ``num_tasks`` is not in the default group-by list:

.. code-block:: bash

   # `num_tasks` is not in the group-by list, reframe will complain
   'now-1d:now/mean,mean(num_tasks):/'

The correct is to add ``num_tasks`` in the group-by list as follows:

.. code-block:: bash

   'now-1d:now/mean,mean(num_tasks):/+num_tasks'


The ``<cols>`` spec has the following syntax:

.. code-block:: bnf

   <cols> ::= <extra_cols> | <explicit_cols>
   <extra_cols> ::= ("+" <attr>)+
   <explicit_cols> ::= <attr> ("," <attr>)*

Users can either add attributes to the default list by followint the syntax ``+attr1+attr2...`` or can completely override the group-by attributes by providing an explcit list, such as ``attr1,attr2,...``.

As an attribute for grouping test cases, any loggable test variable or parameter can be selected, as well as the following pseudo-attributes which are extracted or calculated on-the-fly:

- ``basename``: The test's name stripped off from any parameters.
  This is a equivalent to the test's class name.
- ``pvar``: the name of the performance variable
- ``pval``: the value of the performance variable (i.e., the obtained performance)
- ``pref``: the reference value of the performance variable
- ``plower``: the lower threshold of the performance variable as an absolute value
- ``pupper``: the upper threshold of the performance variable as an absolute value
- ``punit``: the unit of the performance variable
- ``presult``: the result (``pass`` or ``fail``) for this performance variable.
  The result is ``pass`` if the obtained performance value is within the acceptable bounds.
- ``pdiff``: the difference as a percentage between the :ref:`left <cmpspec-syntax>` and :ref:`right <cmpspec-syntax>` performance values when a performance comparison is attempted.
  More specifically, ``pdiff = (pval_lhs - pval_rhs) / pval_rhs``.
- ``psamples``: the number of test cases aggregated.
- ``sysenv``: The system/partition/environment combination as a single string of the form ``{system}:{partition}+{environ}``

.. note::

   For performance comparisons, either implicit or explicit, the aggregation applies to both the left- and right-hand-side test cases.

.. note::

   .. versionadded:: 4.8
      The ``presult`` special column was added.

   .. versionadded:: 4.9
      More aggregations are added, multiple aggregations at once are supported and the ``stats`` shortcut is introduced.


Presenting the results
----------------------

The selection of the final columns of the results table is specified by the same syntax as the ``<cols>`` subspec described above.

However, for performance comparisons, ReFrame will generate two columns for every attribute in the subspec that is not also a group-by attribute, suffixed with ``(lhs)`` and ``(rhs)``.
These suffixes can be changed using the :option:`--term-lhs` and :option:`--term-rhs` options, respectively.
These columns contain the aggregated values of the corresponding attributes.

Note that any attributes/columns that are not part of the group-by set will be aggregated by joining their unique values.

It is also possible to select only one of the lhs/rhs variants of the extra columns by adding the ``_lhs`` or ``_rhs`` suffix to the column name in the ``<cols>`` subspec, e.g., ``+result_lhs`` will only show the result of the left selection group in the comparison.

.. versionchanged:: 4.8

   Support for selecting lhs/rhs column variants in performance comparisons.

.. versionchanged:: 4.9

   Multiple aggregations at once are supported. New aggregations are added and ``A/B`` column variants are renamed to ``lhs`` and ``rhs``, respectively.


Examples
--------

Here are some examples of performance comparison specs:

- Compare the test cases of the session ``7a70b2da-1544-4ac4-baf4-0fcddd30b672`` with the mean performance of the last 10 days:

  .. code-block:: console

     7a70b2da-1544-4ac4-baf4-0fcddd30b672/now-10d:now/mean:/

- Compare the best performance of the test cases run on two specific days, group by the node list and report also the test result:

  .. code-block:: console

     20240701:20240702/20240705:20240706/max:+job_nodelist/+result

Grammar
-------

The formal grammar of the comparison syntax in BNF form is the following.
Note that parts that have a grammar defined elsewhere (e.g., Python attributes and expressions, UUIDs etc.) are omitted.

.. code-block:: bnf

   <cmpspec> ::= (<select> "/")? <select> "/" <aggr> "/" <cols>
   <aggr> ::= <aggr_list> ":" <cols>?
   <aggr_list> ::= <aggr_spec> ("," <aggr_spec>)* | "stats"
   <aggr_spec> ::= <aggr_any> | <aggr_pval>
   <aggr_any> ::= <aggr_fn> "(" <attr> ")"
   <aggr_pval> ::= <aggr_fn>
   <aggr_fn> ::= "first" | "last" | "max" | "min" | "mean" | "median" | "std" | "sum" | "p01" | "p05" | "p95" | "p99"
   <cols> ::= <extra_cols> | <explicit_cols>
   <extra_cols> ::= ("+" <attr>)+
   <explicit_cols> ::= <attr> ("," <attr>)*
   <attr> ::= /* any Python attribute */
   <select> ::= <session_uuid> | <session_filter> | <time_period>
   <session_uuid> ::= /* any valid UUID */
   <session_filter> ::= (<time_period>)? "?" <python_expr>
   <python_expr> ::= /* any valid Python expression */
   <time_period> ::= <timestamp> ":" <timestamp>
   <timestamp> ::= ("now" | <abs_timestamp>) (("+" | "-") <number> ("w" | "d" | "h" | "m"))?
   <abs_timestamp> ::= /* any timestamp of the format `%Y%m%d`, `%Y%m%dT%H%M`, `%Y%m%dT%H%M%S` */
   <number> ::= [0-9]+


Results data schema
-------------------

The report generated by ReFrame and stored in the results database follows a specific schema that can be found in `reframe/schemas/runreport.json <https://github.com/reframe-hpc/reframe/blob/develop/reframe/schemas/runreport.json>`__.
Note that the schema is permissive for test case information as we cannot know beforehand all the variables and parameters that a user test defines.
Besides, we cannot duplicate the definition of every built-in test variable in the report schema.

When parsing raw report data, users should always pay attention to the ``session_info.data_version`` property of the report and the ReFrame version that has generated it (``session_info.version``).
The first defines the general structure of the report and major version bumps mean that existing scripts will likely fail parsing the report, whereas minor version bumps will likely cause no disruption.
The second is important when parsing built-in testcase variables.
When a new variable is added or modified (edited, deleted), this is recorded in ReFrame's version scheme and not in the report's scheme.
The following list summarizes the schema changes (version numbers refer to schema data version numbers).

.. admonition:: Schema Changes

   .. admonition:: 1.1

      Should be identical to 1.0.
      Bumped due to a refactoring of the reporting code.

   .. admonition:: 1.2

      ``.restored_cases`` is added.

   .. admonition:: 1.3

      ``.runs[].num_aborted`` is added.

   .. admonition:: 2.0

      ``.runs[].testcases[].display_name`` and ``.runs[].testcases[].unique_name`` are added and required properties for ``runs`` and ``testcases`` changed.

   .. admonition:: 2.1

      ``.runs[].testcases[].fixture`` is added.

   .. admonition:: 3.0

      ``.session_info.config_file`` is renamed to ``.session_info.config_files`` and becomes a list.

   .. admonition:: 3.1

      ``.session_info.log_files`` is added.

   .. admonition:: 4.0

      - ``.runs[].testcases[].check_vars`` and ``.[].runs[].testcases[].check_params`` are removed.
        All variables and parameters are now direct testcase properties.
      - All properties that were duplicate of existing test variables are not included in the schema explicitly.
        As a result, the following changes should be taken into consideration:

        - ``description`` is replaced by ``descr``.
        - ``environment`` is replaced by ``environ``.
        - ``nodelist`` is replaced by ``job_nodelist``.
        - ``perfvars`` is replaced by ``perfvalues`` which corresponds to the :py:attr:`~reframe.core.pipeline.RegressionTest.perfvalues` test property.
          Note that the format of ``perfvalues`` is different from previous ``perfvars``.
        - New properties are added: ``build_jobid``, ``job_completion_time``, ``job_completion_time_unix`` and ``partition``.
        - The ``system`` property contains only the system name.
          The partition should be accessed through ``partition``.
        - The ``result`` values have changed and are now left free in the schema.
        - The ``uuid`` property is added.
      - The ``session_info`` has the following changes:

        - New properties are added: ``num_skipped``, ``time_end_unix``, ``time_start_unix``, ``uuid``.

      - The ``runs`` section has the following changes:

        - New property ``num_skipped``.
        - ``runid`` is renamed to ``run_index``.


   .. admonition:: 4.1

      Every reference tuple of ``.[].runs[].testcases[].perfvalues`` has now an additional element at the end, denoting the result (``pass`` or ``fail``) for the corresponding performance variable.

   .. admonition:: 4.2

      A new test ``result`` is introduced: ``fail_deps``, for when the dependencies of a test fail and the test is skipped.

      Since ReFrame 4.9, if a test's dependencies fail, the test is skipped and is put in the ``fail_deps`` state.
      Previously, it was treated as a normal failure.


Environment
===========

Several aspects of ReFrame can be controlled through environment variables.
Usually environment variables have counterparts in command line options or configuration parameters.
In such cases, command-line options take precedence over environment variables, which in turn precede configuration parameters.
Boolean environment variables can have any value of ``true``, ``yes``, ``y`` (case insensitive) or ``1`` to denote true and any value of ``false``, ``no``, ``n`` (case insensitive) or ``0`` to denote false.

.. note::

   .. versionchanged:: 3.9.2
      Values ``1`` and ``0`` are now valid for boolean environment variables.


Here is an alphabetical list of the environment variables recognized by ReFrame.
Whenever an environment variable is associated with a configuration option, its default value is omitted as it is the same.


.. envvar:: RFM_AUTODETECT_FQDN

   Use the fully qualified domain name as the hostname.
   This is a boolean variable and defaults to ``0``.


   .. table::
      :align: left

      ================================== ==================
      Associated command line option     N/A
      Associated configuration parameter N/A
      ================================== ==================


   .. versionadded:: 3.11.0

   .. versionchanged:: 4.0.0
      This variable now defaults to ``0``.

   .. deprecated:: 4.3
      Please use ``RFM_AUTODETECT_METHODS=py::fqdn`` in the future.


.. envvar:: RFM_AUTODETECT_METHOD

   Method to use for detecting the current system and pick the right configuration.
   The following values can be used:

   - ``hostname``: The ``hostname`` command will be used to detect the current system.
     This is the default value, if not specified.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     N/A
      Associated configuration parameter N/A
      ================================== ==================


   .. versionadded:: 3.11.0
   .. deprecated:: 4.3
      This has no effect.
      For setting multiple auto-detection methods, please use the :envvar:`RFM_AUTODETECT_METHODS`.

.. envvar:: RFM_AUTODETECT_METHODS

   A comma-separated list of system auto-detection methods.
   Please refer to the :attr:`autodetect_methods` configuration parameter for more information on how to set this variable.

   .. versionadded:: 4.3


.. envvar:: RFM_AUTODETECT_XTHOSTNAME

   Use ``/etc/xthostname`` file, if present, to retrieve the current system's name.
   If the file cannot be found, the hostname will be retrieved using the ``hostname`` command.
   This is a boolean variable and defaults to ``0``.

   This option meaningful for Cray systems.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     N/A
      Associated configuration parameter N/A
      ================================== ==================


   .. versionadded:: 3.11.0

   .. versionchanged:: 4.0.0
      This variable now defaults to ``0``.

   .. deprecated:: 4.3
      Please use ``RFM_AUTODETECT_METHODS='cat /etc/xthostname,hostname'`` in the future.


.. envvar:: RFM_CHECK_SEARCH_PATH

   A colon-separated list of filesystem paths where ReFrame should search for tests.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`-c`
      Associated configuration parameter :attr:`~config.general.check_search_path`
      ================================== ==================


.. envvar:: RFM_CHECK_SEARCH_RECURSIVE

   Search for test files recursively in directories found in the check search path.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`-R`
      Associated configuration parameter :attr:`~config.general.check_search_recursive`
      ================================== ==================


.. envvar:: RFM_CLEAN_STAGEDIR

   Clean stage directory of tests before populating it.

   .. versionadded:: 3.1

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`--dont-restage`
      Associated configuration parameter :attr:`~config.general.clean_stagedir`
      ================================== ==================


.. envvar:: RFM_COLORIZE

   Enable output coloring.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`--nocolor`
      Associated configuration parameter :attr:`~config.general.colorize`
      ================================== ==================


.. envvar:: RFM_COMPRESS_REPORT

   Compress the generated run report file.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`--compress-report`
      Associated configuration parameter :attr:`~config.general.compress_report`
      ================================== ==================

   .. versionadded:: 3.12.0

.. envvar:: RFM_CONFIG_FILE

   Set the configuration file for ReFrame.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`-C`
      Associated configuration parameter N/A
      ================================== ==================

   .. deprecated:: 4.0.0
      Please use the :envvar:`RFM_CONFIG_FILES` instead.


.. envvar:: RFM_CONFIG_FILES

   A colon-separated list of configuration files to load.
   Refer to the documentation of the :option:`--config-file` option for a detailed description on how ReFrame loads its configuration.


   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`-C`
      Associated configuration parameter N/A
      ================================== ==================

   .. versionadded:: 4.0.0

.. envvar:: RFM_CONFIG_PATH

   A colon-separated list of directories that contain ReFrame configuration files.
   Refer to the documentation of the :option:`--config-file` option for a detailed description on how ReFrame loads its configuration.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     N/A
      Associated configuration parameter N/A
      ================================== ==================

   .. versionadded:: 4.0.0


.. envvar:: RFM_FLEX_ALLOC_STRICT

   Fail flexible tests if their minimum task requirement is not satisfied.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`--flex-alloc-strict`
      Associated configuration parameter :attr:`~config.general.flex_alloc_strict`
      ================================== ==================

   .. versionadded:: 4.7


.. envvar:: RFM_GENERATE_FILE_REPORTS

   Store session reports also in files.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     n/a
      Associated configuration parameter :attr:`~config.general.generate_file_reports`
      ================================== ==================

   .. versionadded:: 4.7

.. envvar:: RFM_GIT_TIMEOUT

   Timeout value in seconds used when checking if a git repository exists.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     N/A
      Associated configuration parameter :attr:`~config.general.git_timeout`
      ================================== ==================


   .. versionadded:: 3.9.0


.. envvar:: RFM_GRAYLOG_ADDRESS

   The address of the Graylog server to send performance logs.
   The address is specified in ``host:port`` format.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     N/A
      Associated configuration parameter :attr:`~config.logging.handlers_perflog..graylog..address`
      ================================== ==================


   .. versionadded:: 3.1


.. envvar:: RFM_HTTPJSON_URL

   The URL of the server to send performance logs in JSON format.
   The URL is specified in ``scheme://host:port/path`` format.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     N/A
      Associated configuration parameter :attr:`~config.logging.handlers_perflog..httpjson..url`
      ================================== ==================


.. versionadded:: 3.6.1


.. envvar:: RFM_IGNORE_REQNODENOTAVAIL

   Do not treat specially jobs in pending state with the reason ``ReqNodeNotAvail`` (Slurm only).

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     N/A
      Associated configuration parameter :attr:`~config.systems.partitions.sched_options.ignore_reqnodenotavail`
      ================================== ==================


.. envvar:: RFM_INSTALL_PREFIX

   The framework's installation prefix.
   Users cannot set this variable.
   ReFrame will set it always upon startup.


.. envvar:: RFM_KEEP_STAGE_FILES

   Keep test stage directories even for tests that finish successfully.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`--keep-stage-files`
      Associated configuration parameter :attr:`~config.general.keep_stage_files`
      ================================== ==================


.. envvar:: RFM_MODULE_MAP_FILE

   A file containing module mappings.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`--module-mappings`
      Associated configuration parameter :attr:`~config.general.module_map_file`
      ================================== ==================


.. envvar:: RFM_MODULE_MAPPINGS

   A comma-separated list of module mappings.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`-M`
      Associated configuration parameter :attr:`~config.general.module_mappings`
      ================================== ==================


.. envvar:: RFM_NON_DEFAULT_CRAYPE

   Test a non-default Cray Programming Environment.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`--non-default-craype`
      Associated configuration parameter :attr:`~config.general.non_default_craype`
      ================================== ==================


.. envvar:: RFM_OUTPUT_DIR

   Directory prefix for test output files.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`-o`
      Associated configuration parameter :attr:`~config.systems.outputdir`
      ================================== ==================


.. envvar:: RFM_PERF_INFO_LEVEL

   Logging level at which the immediate performance information is logged.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     n/a
      Associated configuration parameter :attr:`~config.general.perf_info_level`
      ================================== ==================


.. envvar:: RFM_PERF_REPORT_SPEC

   The default ``CMPSPEC`` of the :option:`--performance-report` option.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`--performance-report`
      Associated configuration parameter :attr:`~config.general.perf_report_spec`
      ================================== ==================

   .. versionadded:: 4.7


.. envvar:: RFM_PERFLOG_DIR

   Directory prefix for logging performance data.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`--perflogdir`
      Associated configuration parameter :attr:`~config.logging.handlers_perflog..filelog..basedir`
      ================================== ==================


.. envvar:: RFM_PIPELINE_TIMEOUT

   Timeout in seconds for advancing the pipeline in the asynchronous execution policy.
   See :ref:`pipeline-timeout` for more guidance on how to set this.


   .. table::
      :align: left

      ================================== ==================
      Associated command line option     N/A
      Associated configuration parameter :attr:`~config.general.pipeline_timeout`
      ================================== ==================

   .. versionadded:: 3.10.0


.. _polling_envvars:

.. envvar:: RFM_POLL_RANDOMIZE_MS

   Range of randomization of the polling interval in milliseconds.

   The range is specified in the form ``l,h``.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     N/A
      Associated configuration parameter :attr:`~config.general.poll_randomize_ms`
      ================================== ==================

   .. versionadded:: 4.9

.. envvar:: RFM_POLL_RATE_DECAY

   The decay factor of the polling rate.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     N/A
      Associated configuration parameter :attr:`~config.general.poll_rate_decay`
      ================================== ==================

   .. versionadded:: 4.9


.. envvar:: RFM_POLL_RATE_MAX

   The maximum desired polling rate.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     N/A
      Associated configuration parameter :attr:`~config.general.poll_rate_max`
      ================================== ==================

   .. versionadded:: 4.9


.. envvar:: RFM_POLL_RATE_MIN

   The minimum desired polling rate.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     N/A
      Associated configuration parameter :attr:`~config.general.poll_rate_min`
      ================================== ==================

   .. versionadded:: 4.9


.. envvar:: RFM_PREFIX

   General directory prefix for ReFrame-generated directories.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`--prefix`
      Associated configuration parameter :attr:`~config.systems.prefix`
      ================================== ==================


.. envvar:: RFM_PURGE_ENVIRONMENT

   Unload all environment modules before acting on any tests.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`--purge-env`
      Associated configuration parameter :attr:`~config.general.purge_environment`
      ================================== ==================


.. envvar:: RFM_REMOTE_DETECT

   Auto-detect processor information of remote partitions as well.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     N/A
      Associated configuration parameter :attr:`~config.general.remote_detect`
      ================================== ==================

   .. versionadded:: 3.7.0


.. envvar:: RFM_REMOTE_WORKDIR

   The temporary directory prefix that will be used to create a fresh ReFrame clone, in order to auto-detect the processor information of a remote partition.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     N/A
      Associated configuration parameter :attr:`~config.general.remote_workdir`
      ================================== ==================

   .. versionadded:: 3.7.0


.. envvar:: RFM_REPORT_FILE

   The file where ReFrame will store its report.

   .. versionadded:: 3.1

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`--report-file`
      Associated configuration parameter :attr:`~config.general.report_file`
      ================================== ==================


.. envvar:: RFM_REPORT_JUNIT

   The file where ReFrame will generate a JUnit XML report.

   .. versionadded:: 3.6.0

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`--report-junit`
      Associated configuration parameter :attr:`~config.general.report_junit`
      ================================== ==================


.. envvar:: RFM_RESOLVE_MODULE_CONFLICTS

   Resolve module conflicts automatically.

   .. versionadded:: 3.6.0

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     N/A
      Associated configuration parameter :attr:`~config.general.resolve_module_conflicts`
      ================================== ==================


.. envvar:: RFM_SAVE_LOG_FILES

   Save ReFrame log files in the output directory before exiting.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`--save-log-files`
      Associated configuration parameter :attr:`~config.general.save_log_files`
      ================================== ==================


.. envvar:: RFM_SCHED_ACCESS_IN_SUBMIT

   Pass access options in the submission command (relevant for LSF, OAR, PBS and Slurm).

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     N/A
      Associated configuration parameter :attr::attr:`~config.systems.partitions.sched_options.sched_access_in_submit`
      ================================== ==================

.. versionadded:: 4.7


.. envvar:: RFM_STAGE_DIR

   Directory prefix for staging test resources.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`-s`
      Associated configuration parameter :attr:`~config.systems.stagedir`
      ================================== ==================


.. envvar:: RFM_SQLITE_CONN_TIMEOUT

   Timeout for SQLite database connections.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     N/A
      Associated configuration parameter :attr:`~config.storage.sqlite_conn_timeout`
      ================================== ==================

   .. versionadded:: 4.7


.. envvar:: RFM_SQLITE_DB_FILE

   The SQLite database file for storing test results.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     N/A
      Associated configuration parameter :attr:`~config.storage.sqlite_db_file`
      ================================== ==================

   .. versionadded:: 4.7


.. envvar:: RFM_SQLITE_DB_FILE_MODE

   The permissions of the SQLite database file in octal form.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     N/A
      Associated configuration parameter :attr:`~config.storage.sqlite_db_file_mode`
      ================================== ==================

   .. versionadded:: 4.7


.. envvar:: RFM_SYSLOG_ADDRESS

   The address of the Syslog server to send performance logs.
   The address is specified in ``host:port`` format.
   If no port is specified, the address refers to a UNIX socket.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     N/A
      Associated configuration parameter :attr:`~config.logging.handlers..syslog..address`
      ================================== ==================


.. versionadded:: 3.1

.. envvar:: RFM_SYSTEM

   Set the current system.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`--system`
      Associated configuration parameter N/A
      ================================== ==================


.. envvar:: RFM_TABLE_FORMAT

   Set the format of the tables printed by various options accessing the results storage.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`--table-format`
      Associated configuration parameter :attr:`~config.general.table_format`
      ================================== ==================

   .. versionadded:: 4.7

.. envvar:: RFM_TABLE_FORMAT_DELIM

   Delimiter for CSV tables.


   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`--table-format-delim`
      Associated configuration parameter :attr:`~config.general.table_format_delim`
      ================================== ==================

   .. versionadded:: 4.9


.. envvar:: RFM_TIMESTAMP_DIRS

   Append a timestamp to the output and stage directory prefixes.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     |--timestamp|_
      Associated configuration parameter :attr:`~config.general.timestamp_dirs`
      ================================== ==================

.. |--timestamp| replace:: :attr:`--timestamp`
.. _--timestamp: #cmdoption-timestamp



.. envvar:: RFM_TRAP_JOB_ERRORS

   Trap job errors in submitted scripts and fail tests automatically.

   .. table::
      :align: left

      ================================== ==================
      Associated configuration parameter :attr:`~config.general.trap_job_errors`
      ================================== ==================

   .. versionadded:: 3.9.0


.. envvar:: RFM_UNLOAD_MODULES

   A comma-separated list of environment modules to be unloaded before acting on any tests.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`-u`
      Associated configuration parameter :attr:`~config.general.unload_modules`
      ================================== ==================


.. envvar:: RFM_USE_LOGIN_SHELL

   Use a login shell for the generated job scripts.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     N/A
      Associated configuration parameter :attr:`~config.general.use_login_shell`
      ================================== ==================


.. envvar:: RFM_USER_MODULES

   A comma-separated list of environment modules to be loaded before acting on any tests.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`-m`
      Associated configuration parameter :attr:`~config.general.user_modules`
      ================================== ==================


.. envvar:: RFM_VERBOSE

   Set the verbosity level of output.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`-v`
      Associated configuration parameter :attr:`~config.general.verbose`
      ================================== ==================


.. _manpage-configuration:

Configuration
=============

ReFrame's configuration can be stored in one or multiple configuration files.
Two configuration file types are supported: Python and YAML.

.. note::

   .. versionchanged:: 4.8

   The JSON configuration files are deprecated.

The configuration of ReFrame defines the systems and environments to test as well as parameters controlling the framework's behavior.

To determine its final configuration, ReFrame executes the following steps:

- First, it unconditionally loads the builtin configuration which is located in ``${RFM_INSTALL_PREFIX}/reframe/core/settings.py``.
- Second, if the :envvar:`RFM_CONFIG_PATH` environment variable is defined, ReFrame will look for configuration files named either ``settings.py`` or ``settings.yaml`` or ``settings.json`` (in that order) in every location in the path and will load them.
- Finally, the :option:`--config-file` option is processed and any configuration files specified will also be loaded.

For a complete reference of the available configuration options, please refer to the :doc:`reframe.settings(8) <config_reference>` man page.


Reporting Bugs
==============

For bugs, feature request, help, please open an issue on Github: <https://github.com/reframe-hpc/reframe>


See Also
========

See full documentation online: <https://reframe-hpc.readthedocs.io/>
