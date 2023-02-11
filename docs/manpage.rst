======================
Command Line Reference
======================


Synopsis
--------

.. option:: reframe [OPTION]... ACTION


Description
-----------

ReFrame provides both a `programming interface <regression_test_api.html>`__ for writing regression tests and a command-line interface for managing and running the tests, which is detailed here.
The ``reframe`` command is part of ReFrame's frontend.
This frontend is responsible for loading and running regression tests written in ReFrame.
ReFrame executes tests by sending them down to a well defined pipeline.
The implementation of the different stages of this pipeline is part of ReFrame's core architecture, but the frontend is responsible for driving this pipeline and executing tests through it.
There are three basic phases that the frontend goes through, which are described briefly in the following.


-------------------------------
Test discovery and test loading
-------------------------------

This is the very first phase of the frontend.
ReFrame will search for tests in its *check search path* and will load them.
When ReFrame loads a test, it actually *instantiates* it, meaning that it will call its :func:`__init__` method unconditionally whether this test is meant to run on the selected system or not.
This is something that writers of regression tests should bear in mind.

.. option:: -c, --checkpath=PATH

   A filesystem path where ReFrame should search for tests.

   ``PATH`` can be a directory or a single test file.
   If it is a directory, ReFrame will search for test files inside this directory load all tests found in them.
   This option can be specified multiple times, in which case each ``PATH`` will be searched in order.

   The check search path can also be set using the :envvar:`RFM_CHECK_SEARCH_PATH` environment variable or the :attr:`~config.general.check_search_path` general configuration parameter.

.. option:: -R, --recursive

   Search for test files recursively in directories found in the check search path.

   This option can also be set using the :envvar:`RFM_CHECK_SEARCH_RECURSIVE` environment variable or the :attr:`~config.general.check_search_recursive` general configuration parameter.


--------------
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

.. option:: --failed

   Select only the failed test cases for a previous run.

   This option can only be used in combination with the :option:`--restore-session`.
   To rerun the failed cases from the last run, you can use ``reframe --restore-session --failed -r``.

   .. versionadded:: 3.4

.. option:: --gpu-only

   Select tests that can run on GPUs.

   These are all tests with :attr:`num_gpus_per_node` greater than zero.
   This option and :option:`--cpu-only` are mutually exclusive.

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

------------
Test actions
------------

ReFrame will finally act upon the selected tests.
There are currently two actions that can be performed on tests: (a) list the tests and (b) execute the tests.
An action must always be specified.


.. option:: --ci-generate=FILE

   Do not run the tests, but generate a Gitlab `child pipeline <https://docs.gitlab.com/ee/ci/parent_child_pipelines.html>`__ specification in ``FILE``.

   You can set up your Gitlab CI to use the generated file to run every test as a separate Gitlab job respecting test dependencies.
   For more information, have a look in :ref:`generate-ci-pipeline`.

   .. versionadded:: 3.4.1

.. option:: --describe

   Print a detailed description of the `selected tests <#test-filtering>`__ in JSON format and exit.

   .. note::
      The generated test description corresponds to its state after it has been initialized.
      If any of its attributes are changed or set during its execution, their updated values will not be shown by this listing.

   .. versionadded:: 3.10.0


.. option:: -L, --list-detailed[=T|C]

   List selected tests providing more details for each test.

   The unique id of each test (see also :attr:`~reframe.core.pipeline.RegressionTest.unique_name`) as well as the file where each test is defined are printed.

   This option accepts optionally a single argument denoting what type of listing is requested.
   Please refer to :option:`-l` for an explanation of this argument.

   .. versionadded:: 3.10.0
      Support for different types of listing is added.

.. option:: -l, --list[=T|C]

   List selected tests and their dependencies.

   This option accepts optionally a single argument denoting what type of listing is requested.
   There are two types of possible listings:

   - *Regular test listing* (``T``, the default): This type of listing lists the tests and their dependencies or fixtures using their :attr:`~reframe.core.pipeline.RegressionTest.display_name`. A test that is listed as a dependency of another test will not be listed separately.
   - *Concretized test case listing* (``C``): This type of listing lists the exact test cases and their dependencies as they have been concretized for the current system and environment combinations.
     This listing shows practically the exact test DAG that will be executed.

   .. versionadded:: 3.10.0
      Support for different types of listing is added.

.. option:: --list-tags

   List the unique tags of the selected tests.

   The tags are printed in alphabetical order.

   .. versionadded:: 3.6.0

.. option:: -r, --run

   Execute the selected tests.

If more than one action options are specified, the precedence order is the following:

   .. code-block:: console

      --describe > --list-detailed > --list > --list-tags > --ci-generate


----------------------------------
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

   This option is relevant only to the ``filelog`` `logging handler <config_reference.html#the-filelog-log-handler>`__.

   This option can also be set using the :envvar:`RFM_PERFLOG_DIR` environment variable or the :attr:`~config.logging.handlers_perflog..filelog..basedir` logging handler configuration parameter.

.. option:: --prefix=DIR

   General directory prefix for ReFrame-generated directories.

   The base stage and output directories (see below) will be specified relative to this prefix if not specified explicitly.

   This option can also be set using the :envvar:`RFM_PREFIX` environment variable or the :attr:`~config.systems.prefix` system configuration parameter.

.. option:: --report-file=FILE

   The file where ReFrame will store its report.

   The ``FILE`` argument may contain the special placeholder ``{sessionid}``, in which case ReFrame will generate a new report each time it is run by appending a counter to the report file.

   This option can also be set using the :envvar:`RFM_REPORT_FILE` environment variable or the :attr:`~config.general.report_file` general configuration parameter.

   .. versionadded:: 3.1

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

   Only log files generated by ``file`` `log handlers <config_reference.html#the-file-log-handler>`__ will be copied.

   This option can also be set using the :envvar:`RFM_SAVE_LOG_FILES` environment variable or the :attr:`~config.general.save_log_files` general configuration parameter.

.. option:: --timestamp [TIMEFMT]

   Append a timestamp to the output and stage directory prefixes.

   ``TIMEFMT`` can be any valid :manpage:`strftime(3)` time format.
   If not specified, ``TIMEFMT`` is set to ``%FT%T``.

   This option can also be set using the :envvar:`RFM_TIMESTAMP_DIRS` environment variable or the :attr:`~config.general.timestamp_dirs` general configuration parameter.


-------------------------------------
Options controlling ReFrame execution
-------------------------------------

.. option:: --disable-hook=HOOK

   Disable the pipeline hook named ``HOOK`` from all the tests that will run.

   This feature is useful when you have implemented test workarounds as pipeline hooks, in which case you can quickly disable them from the command line.
   This option may be specified multiple times in order to disable multiple hooks at the same time.

   .. versionadded:: 3.2

.. option:: --distribute[=NODESTATE]

   Distribute the selected tests on all the nodes in state ``NODESTATE`` in their respective valid partitions.

   ReFrame will parameterize and run the tests on the selected nodes.
   Effectively, it will dynamically create new tests that inherit from the original tests and add a new parameter named ``$nid`` which contains the list of nodes that the test must run on.
   The new tests are named with the following pattern  ``{orig_test_basename}_{partition_fullname}``.

   When determining the list of nodes to distribute the selected tests, ReFrame will take into account any job options passed through the :option:`-J` option.

   You can optionally specify the state of the nodes to consider when distributing the test through the ``NODESTATE`` argument:

   - ``all``: Tests will run on all the nodes of their respective valid partitions regardless of the nodes' state.
   - ``idle``: Tests will run on all *idle* nodes of their respective valid partitions.
   - ``NODESTATE``: Tests will run on all the nodes in state ``NODESTATE`` of their respective valid partitions.
     If ``NODESTATE`` is not specified, ``idle`` will be assumed.

   The state of the nodes will be determined once, before beginning the
   execution of the tests, so it might be different at the time the tests are actually submitted.

   .. note::
      Currently, only single-node jobs can be distributed and only local or the Slurm-based backends support this feature.

   .. note::
      Distributing tests with dependencies is not supported, but you can distribute tests that use fixtures.


   .. versionadded:: 3.11.0


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
   For example, if the execution mode ``foo`` defines ``-S modules=foo``, the invocation ``-S mode=foo -S num_tasks=10`` is the equivalent of ``-S modules=foo -S num_tasks=10``.

   .. versionchanged:: 4.1
      Options that can be specified multiple times are now combined between execution modes and the command line.

.. option:: --repeat=N

   Repeat the selected tests ``N`` times.
   This option can be used in conjunction with the :option:`--distribute` option in which case the selected tests will be repeated multiple times and distributed on individual nodes of the system's partitions.

   .. note::
      Repeating tests with dependencies is not supported, but you can repeat tests that use fixtures.

   .. versionadded:: 3.12.0

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

   .. versionadded:: 3.4

   .. versionchanged:: 3.6.1
      Multiple report files are now accepted.

.. option:: -S, --setvar=[TEST.]VAR=VAL

   Set variable ``VAR`` in all tests or optionally only in test ``TEST`` to ``VAL``.

   ``TEST`` can have the form ``[TEST.][FIXT.]*``, in which case ``VAR`` will be set in fixture ``FIXT`` of ``TEST``.
   Note that this syntax is recursive on fixtures, so that a variable can be set in a fixture arbitrarily deep.
   ``TEST`` prefix refers to the test class name, *not* the test name, but ``FIXT`` refers to the fixture name *inside* the referenced test.

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


.. option:: --skip-performance-check

   Skip performance checking phase.

   The phase is completely skipped, meaning that performance data will *not* be logged.

.. option:: --skip-sanity-check

   Skip sanity checking phase.


----------------------------------
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

------------------------
Flexible node allocation
------------------------

ReFrame can automatically set the number of tasks of a test, if its :attr:`num_tasks <reframe.core.pipeline.RegressionTest.num_tasks>` attribute is set to a value less than or equal to zero.
This scheme is conveniently called *flexible node allocation* and is valid only for the Slurm backend.
When allocating nodes automatically, ReFrame will take into account all node limiting factors, such as partition :attr:`~config.systems.partitions.access` options, and any job submission control options described above.
Nodes from this pool are allocated according to different policies.
If no node can be selected, the test will be marked as a failure with an appropriate message.

.. option:: --flex-alloc-nodes=POLICY

   Set the flexible node allocation policy.

   Available values are the following:

   - ``all``: Flexible tests will be assigned as many tasks as needed in order to span over *all* the nodes of the node pool.
   - ``STATE``: Flexible tests will be assigned as many tasks as needed in order to span over the nodes that are currently in state ``STATE``.
     Querying of the node state and submission of the test job are two separate steps not executed atomically.
     It is therefore possible that the number of tasks assigned does not correspond to the actual nodes in the given state.

     If this option is not specified, the default allocation policy for flexible tests is 'idle'.
   - Any positive integer: Flexible tests will be assigned as many tasks as needed in order to span over the specified number of nodes from the node pool.

   .. versionchanged:: 3.1
      It is now possible to pass an arbitrary node state as a flexible node allocation parameter.


---------------------------------------
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


---------------------
Miscellaneous options
---------------------

.. option:: -C, --config-file=FILE

   Use ``FILE`` as configuration file for ReFrame.

   This option can be passed multiple times, in which case multiple configuration files will be read and loaded successively.
   The base of the configuration chain is always the builtin configuration file, namely the ``${RFM_INSTALL_PREFIX}/reframe/core/settings.py``.
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

.. _--detect-host-topology:

.. option:: --detect-host-topology[=FILE]

   Detect the local host processor topology, store it to ``FILE`` and exit.

   If no ``FILE`` is specified, the standard output will be used.

   .. versionadded:: 3.7.0

.. option:: --failure-stats

   Print failure statistics at the end of the run.

.. option:: -h, --help

   Print a short help message and exit.

.. option:: --nocolor

   Disable output coloring.

   This option can also be set using the :envvar:`RFM_COLORIZE` environment variable or the :attr:`~config.general.colorize` general configuration parameter.

.. option:: --performance-report

   Print a performance report for all the performance tests that have been run.

   The report shows the performance values retrieved for the different performance variables defined in the tests.

.. option:: -q, --quiet

   Decrease the verbosity level.

   This option can be specified multiple times.
   Every time this option is specified, the verbosity level will be decreased by one.
   This option can be combined arbitrarily with the :option:`-v` option, in which case the final verbosity level will be determined by the final combination.
   For example, specifying ``-qv`` will not change the verbosity level, since the two options cancel each other, but ``-qqv`` is equivalent to ``-q``.
   For a list of ReFrame's verbosity levels, see the description of the :option:`-v` option.

   .. versionadded:: 3.9.3


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

.. option:: --system=NAME

   Load the configuration for system ``NAME``.

   The ``NAME`` must be a valid system name in the configuration file.
   It may also have the form ``SYSNAME:PARTNAME``, in which case the configuration of system ``SYSNAME`` will be loaded, but as if it had ``PARTNAME`` as its sole partition.
   Of course, ``PARTNAME`` must be a valid partition of system ``SYSNAME``.
   If this option is not specified, ReFrame will try to pick the correct configuration entry automatically.
   It does so by trying to match the hostname of the current machine again the hostname patterns defined in the :attr:`~config.systems.hostnames` system configuration parameter.
   The system with the first match becomes the current system.

   This option can also be set using the :envvar:`RFM_SYSTEM` environment variable.

.. option:: --upgrade-config-file=OLD[:NEW]

   Convert the old-style configuration file ``OLD``, place it into the new file ``NEW`` and exit.

   If a new file is not given, a file in the system temporary directory will be created.

.. option:: -V, --version

   Print version and exit.

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
------------------

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

   <display_name> ::= <test_class_name> (<params>)* (<scope>)?
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

   - TestA %x=4 %l.foo=10 %t.p=2 /1c51609b
       ^Myfixture %p=1 ~TestA_3 /f027ee75
       ^MyFixture %p=2 ~TestA_3 /830323a4
       ^X %foo=10 ~generic:default+builtin /7dae3cc5
   - TestA %x=3 %l.foo=10 %t.p=2 /707b752c
       ^MyFixture %p=1 ~TestA_2 /02368516
       ^MyFixture %p=2 ~TestA_2 /854b99b5
       ^X %foo=10 ~generic:default+builtin /7dae3cc5
   - TestA %x=4 %l.foo=10 %t.p=1 /c65657d5
       ^MyFixture %p=2 ~TestA_1 /f0383f7f
       ^MyFixture %p=1 ~TestA_1 /d07f4281
       ^X %foo=10 ~generic:default+builtin /7dae3cc5
   - TestA %x=3 %l.foo=10 %t.p=1 /1b9f44df
       ^MyFixture %p=2 ~TestA_0 /b894ab05
       ^MyFixture %p=1 ~TestA_0 /ca376ca8
       ^X %foo=10 ~generic:default+builtin /7dae3cc5
   Found 4 check(s)

Display names may not always be unique.
Assume the following test:

.. code-block:: python

   class MyTest(RegressionTest):
       p = parameter([1, 1, 1])

This generates three different tests with different unique names, but their display name is the same for all: ``MyTest %p=1``.
Notice that this example leads to a name conflict with the old naming scheme, since all tests would be named ``MyTest_1``.

Each test is also associated with a hash code that is derived from the test name, its parameters and their values.
As in the example listing above, the hash code of each test is printed with the :option:`-l` option and individual tests can be selected by their hash using the :option:`-n` option, e.g., ``-n /1c51609b``.
The stage and output directories, as well as the performance log file of the ``filelog`` `performance log handler <config_reference.html#the-filelog-log-handler>`__ will use the hash code for the test-specific directories and files.
This might lead to conflicts for tests as the one above when executing them with the asynchronous execution policy, but ensures consistency of performance record files when parameter values are added to or deleted from a test parameter.
More specifically, the test's hash will not change if a new parameter value is added or deleted or even if the parameter values are shuffled.
Test variants on the other side are more volatile and can change with such changes.
Also users should not rely on how the variant numbers are assigned to a test, as this is an implementation detail.


.. versionchanged:: 4.0.0

   A hash code is associated with each test.


--------------------------------------
Differences from the old naming scheme
--------------------------------------

Prior to version 3.10, ReFrame used to encode the parameter values of an instance of parameterized test in its name.
It did so by taking the string representation of the value and replacing any non-alphanumeric character with an underscore.
This could lead to very large and hard to read names when a test defined multiple parameters or the parameter type was more complex.
Very large test names meant also very large path names which could also lead to problems and random failures.
Fixtures followed a similar naming pattern making them hard to debug.


Environment
-----------

Several aspects of ReFrame can be controlled through environment variables.
Usually environment variables have counterparts in command line options or configuration parameters.
In such cases, command-line options take precedence over environment variables, which in turn precede configuration parameters.
Boolean environment variables can have any value of ``true``, ``yes``, ``y`` (case insensitive) or ``1`` to denote true and any value of ``false``, ``no``, ``n`` (case insensitive) or ``0`` to denote false.

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
      Associated configuration parameter :attr:`~config.logging.handlers..graylog..address`
      ================================== ==================


   .. versionadded:: 3.1


.. envvar:: RFM_HTTPJSON_URL

   The URL of the server to send performance logs in JSON format.
   The URL is specified in ``scheme://host:port/path`` format.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     N/A
      Associated configuration parameter :attr:`~config.logging.handlers..httpjson..url`
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


.. envvar:: RFM_PERFLOG_DIR

   Directory prefix for logging performance data.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`--perflogdir`
      Associated configuration parameter :attr:`~config.logging.handlers..filelog..basedir`
      ================================== ==================


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


.. envvar:: RFM_STAGE_DIR

   Directory prefix for staging test resources.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`-s`
      Associated configuration parameter :attr:`~config.systems.stagedir`
      ================================== ==================


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



Configuration File
------------------

The configuration file of ReFrame defines the systems and environments to test as well as parameters controlling its behavior.
Upon start up ReFrame checks for configuration files in the following locations in that order:

1. ``$HOME/.reframe/settings.{py,json}``
2. ``$RFM_INSTALL_PREFIX/settings.{py,json}``
3. ``/etc/reframe.d/settings.{py,json}``

ReFrame accepts configuration files either in Python or JSON syntax.
If both are found in the same location, the Python file will be preferred.

The ``RFM_INSTALL_PREFIX`` environment variable refers to the installation directory of ReFrame.
Users have no control over this variable.
It is always set by the framework upon startup.

If no configuration file can be found in any of the predefined locations, ReFrame will fall back to a generic configuration that allows it to run on any system.
This configuration file is located in |reframe/core/settings.py|_.
Users may *not* modify this file.

For a complete reference of the configuration, please refer to |reframe.settings(8)|_ man page.

.. |reframe/core/settings.py| replace:: ``reframe/core/settings.py``
.. _reframe/core/settings.py: https://github.com/reframe-hpc/reframe/blob/master/reframe/core/settings.py
.. |reframe.settings(8)| replace:: ``reframe.settings(8)``
.. _reframe.settings(8): config_reference.html


Reporting Bugs
--------------

For bugs, feature request, help, please open an issue on Github: <https://github.com/reframe-hpc/reframe>


See Also
--------

See full documentation online: <https://reframe-hpc.readthedocs.io/>
