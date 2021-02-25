==============================
ReFrame Command Line Reference
==============================


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

   The check search path can also be set using the :envvar:`RFM_CHECK_SEARCH_PATH` environment variable or the :js:attr:`check_search_path` general configuration parameter.

.. option:: -R, --recursive

   Search for test files recursively in directories found in the check search path.

   This option can also be set using the :envvar:`RFM_CHECK_SEARCH_RECURSIVE` environment variable or the :js:attr:`check_search_recursive` general configuration parameter.

.. option:: --ignore-check-conflicts

   Ignore tests with conflicting names when loading.
   ReFrame requires test names to be unique.
   Test names are used as components of the stage and output directory prefixes of tests, as well as for referencing target test dependencies.
   This option should generally be avoided unless there is a specific reason.

   This option can also be set using the :envvar:`RFM_IGNORE_CHECK_CONFLICTS` environment variable or the :js:attr:`ignore_check_conflicts` general configuration parameter.


--------------
Test filtering
--------------

After all tests in the search path have been loaded, they are first filtered by the selected system.
Any test that is not valid for the current system, it will be filtered out.
The current system is either auto-selected or explicitly specified with the :option:`--system` option.
Tests can be filtered by different attributes and there are specific command line options for achieving this.
A common characteristic of all test filtering options is that if a test is selected, then all its dependencies will be selected, too, regardless if they match the filtering criteria or not.
This happens recursively so that if test ``T1`` depends on ``T2`` and ``T2`` depends on ``T3``, then selecting ``T1`` would also select ``T2`` and ``T3``.

.. option:: -t, --tag=TAG

   Filter tests by tag.
   ``TAG`` is interpreted as a `Python Regular Expression <https://docs.python.org/3/library/re.html>`__; all tests that have at least a matching tag will be selected.
   ``TAG`` being a regular expression has the implication that ``-t 'foo'`` will select also tests that define ``'foobar'`` as a tag.
   To restrict the selection to tests defining only ``'foo'``, you should use ``-t 'foo$'``.

   This option may be specified multiple times, in which case only tests defining or matching *all* tags will be selected.

.. option:: -n, --name=NAME

   Filter tests by name.
   ``NAME`` is interpreted as a `Python Regular Expression <https://docs.python.org/3/library/re.html>`__;
   any test whose name matches ``NAME`` will be selected.

   This option may be specified multiple times, in which case tests with *any* of the specified names will be selected:
   ``-n NAME1 -n NAME2`` is therefore equivalent to ``-n 'NAME1|NAME2'``.

.. option:: -x, --exclude=NAME

   Exclude tests by name.
   ``NAME`` is interpreted as a `Python Regular Expression <https://docs.python.org/3/library/re.html>`__;
   any test whose name matches ``NAME`` will be excluded.

   This option may be specified multiple times, in which case tests with *any* of the specified names will be excluded:
   ``-x NAME1 -x NAME2`` is therefore equivalent to ``-x 'NAME1|NAME2'``.

.. option:: -p, --prgenv=NAME

   Filter tests by programming environment.
   ``NAME`` is interpreted as a `Python Regular Expression <https://docs.python.org/3/library/re.html>`__;
   any test for which at least one valid programming environment is matching ``NAME`` will be selected.

   This option may be specified multiple times, in which case only tests matching all of the specified programming environments will be selected.

.. option:: --gpu-only

   Select tests that can run on GPUs.
   These are all tests with :attr:`num_gpus_per_node` greater than zero.
   This option and :option:`--cpu-only` are mutually exclusive.

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


.. option:: --skip-system-check

   Do not filter tests against the selected system.


.. option:: --skip-prgenv-check

   Do not filter tests against programming environments.
   Even if the :option:`-p` option is not specified, ReFrame will filter tests based on the programming environments defined for the currently selected system.
   This option disables that filter completely.


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

.. option:: -l, --list

   List selected tests.
   A single line per test is printed.


.. option:: -L, --list-detailed

   List selected tests providing detailed information per test.


.. option:: -r, --run

   Execute the selected tests.


If more than one action options are specified, :option:`-l` precedes :option:`-L`, which in turn precedes :option:`-r`.


----------------------------------
Options controlling ReFrame output
----------------------------------

.. option:: --prefix=DIR

   General directory prefix for ReFrame-generated directories.
   The base stage and output directories (see below) will be specified relative to this prefix if not specified explicitly.

   This option can also be set using the :envvar:`RFM_PREFIX` environment variable or the :js:attr:`prefix` system configuration parameter.

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

   This option can also be set using the :envvar:`RFM_OUTPUT_DIR` environment variable or the :js:attr:`outputdir` system configuration parameter.


.. option:: -s, --stage=DIR

   Directory prefix for staging test resources.
   ReFrame does not execute tests from their original source directory.
   Instead it creates a test-specific stage directory and copies all test resources there.
   It then changes to that directory and executes the test.
   This test-specific directory is of the form ``{stage_prefix}/{system}/{partition}/{environment}/{test_name}``,
   where ``stage_prefix`` is set by this option.
   If a test finishes successfully, its stage directory will be removed.

   This option can also be set using the :envvar:`RFM_STAGE_DIR` environment variable or the :js:attr:`stagedir` system configuration parameter.

.. option:: --timestamp [TIMEFMT]

   Append a timestamp to the output and stage directory prefixes.
   ``TIMEFMT`` can be any valid :manpage:`strftime(3)` time format.
   If not specified, ``TIMEFMT`` is set to ``%FT%T``.

   This option can also be set using the :envvar:`RFM_TIMESTAMP_DIRS` environment variable or the :js:attr:`timestamp_dirs` general configuration parameter.


.. option:: --perflogdir=DIR

   Directory prefix for logging performance data.
   This option is relevant only to the ``filelog`` `logging handler <config_reference.html#the-filelog-log-handler>`__.

   This option can also be set using the :envvar:`RFM_PERFLOG_DIR` environment variable or the :js:attr:`basedir` logging handler configuration parameter.


.. option:: --keep-stage-files

   Keep test stage directories even for tests that finish successfully.

   This option can also be set using the :envvar:`RFM_KEEP_STAGE_FILES` environment variable or the :js:attr:`keep_stage_files` general configuration parameter.

.. option:: --dont-restage

   Do not restage a test if its stage directory exists.
   Normally, if the stage directory of a test exists, ReFrame will remove it and recreate it.
   This option disables this behavior.

   This option can also be set using the :envvar:`RFM_CLEAN_STAGEDIR` environment variable or the :js:attr:`clean_stagedir` general configuration parameter.

   .. versionadded:: 3.1

.. option:: --save-log-files

   Save ReFrame log files in the output directory before exiting.
   Only log files generated by ``file`` `log handlers <config_reference.html#the-file-log-handler>`__ will be copied.


   This option can also be set using the :envvar:`RFM_SAVE_LOG_FILES` environment variable or the :js:attr:`save_log_files` general configuration parameter.


.. option:: --report-file=FILE

   The file where ReFrame will store its report.
   The ``FILE`` argument may contain the special placeholder ``{sessionid}``, in which case ReFrame will generate a new report each time it is run by appending a counter to the report file.

   This option can also be set using the :envvar:`RFM_REPORT_FILE` environment variable or the :js:attr:`report_file` general configuration parameter.

   .. versionadded:: 3.1


-------------------------------------
Options controlling ReFrame execution
-------------------------------------

.. option:: --force-local

   Force local execution of tests.
   Execute tests as if all partitions of the currently selected system had a ``local`` scheduler.

.. option:: --skip-sanity-check

   Skip sanity checking phase.


.. option:: --skip-performance-check

   Skip performance checking phase.
   The phase is completely skipped, meaning that performance data will *not* be logged.

.. option:: --strict

   Enforce strict performance checking, even if a performance test is marked as not performance critical by having set its :attr:`strict_check` attribute to :class:`False`.


.. option:: --exec-policy=POLICY

   The execution policy to be used for running tests.
   There are two policies defined:

   - ``serial``: Tests will be executed sequentially.
   - ``async``: Tests will be executed asynchronously.
     This is the default policy.

     The ``async`` execution policy executes the run phase of tests asynchronously by submitting their associated jobs in a non-blocking way.
     ReFrame's runtime monitors the progress of each test and will resume the pipeline execution of an asynchronously spawned test as soon as its run phase has finished.
     Note that the rest of the pipeline stages are still executed sequentially in this policy.

     Concurrency can be controlled by setting the :js:attr:`max_jobs` system partition configuration parameter.
     As soon as the concurrency limit is reached, ReFrame will first poll the status of all its pending tests to check if any execution slots have been freed up.
     If there are tests that have finished their run phase, ReFrame will keep pushing tests for execution until the concurrency limit is reached again.
     If no execution slots are available, ReFrame will throttle job submission.


.. option:: --mode=MODE

   ReFrame execution mode to use.
   An execution mode is simply a predefined invocation of ReFrame that is set with the :js:attr:`modes` configuration parameter.
   If an option is specified both in an execution mode and in the command-line, then command-line takes precedence.

.. option:: --max-retries=NUM

   The maximum number of times a failing test can be retried.
   The test stage and output directories will receive a ``_retry<N>`` suffix every time the test is retried.


.. option:: --maxfail=NUM

   The maximum number of failing test cases before the execution is aborted.
   After ``NUM`` failed test cases the rest of the test cases will be aborted.
   The counter of the failed test cases is reset to 0 in every retry.


.. option:: --disable-hook=HOOK

   Disable the pipeline hook named ``HOOK`` from all the tests that will run.
   This feature is useful when you have implemented test workarounds as pipeline hooks, in which case you can quickly disable them from the command line.
   This option may be specified multiple times in order to disable multiple hooks at the same time.

   .. versionadded:: 3.2


.. option:: --restore-session [REPORT]

   Restore a testing session that has run previously.
   ``REPORT`` is a run report file generated by ReFrame.
   If ``REPORT`` is not given, ReFrame will pick the last report file found in the default location of report files (see the :option:`--report-file` option).
   If passed alone, this option will simply rerun all the test cases that have run previously based on the report file data.
   It is more useful to combine this option with any of the `test filtering <#test-filtering>`__ options, in which case only the selected test cases will be executed.
   The difference in test selection process when using this option is that the dependencies of the selected tests will not be selected for execution, as they would normally, but they will be restored.
   For example, if test ``T1`` depends on ``T2`` and ``T2`` depends on ``T3``, then running ``reframe -n T1 -r`` would cause both ``T2`` and ``T3`` to run.
   However, by doing ``reframe -n T1 --restore-session -r``, only ``T1`` would run and its immediate dependence ``T2`` will be restored.
   This is useful when you have deep test dependencies or some of the tests in the dependency chain are very time consuming.

   .. note::
      In order for a test case to be restored, its stage directory must be present.
      This is not a problem when rerunning a failed case, since the stage directories of its dependencies are automatically kept, but if you want to rerun a successful test case, you should make sure to have run with the :option:`--keep-stage-files` option.

   .. versionadded:: 3.4


----------------------------------
Options controlling job submission
----------------------------------

.. option:: -J, --job-option=OPTION

   Pass ``OPTION`` directly to the job scheduler backend.
   The syntax of ``OPTION`` is ``-J key=value``.
   If ``OPTION`` starts with ``-`` it will be passed verbatim to the backend job scheduler.
   If ``OPTION`` starts with ``#`` it will be emitted verbatim in the job script.
   Otherwise, ReFrame will pass ``--key value`` or ``-k value`` (if ``key`` is a single character) to the backend scheduler.
   Any job options specified with this command-line option will be emitted after any job options specified in the :js:attr:`access` system partition configuration parameter.

   Especially for the Slurm backends, constraint options, such as ``-J constraint=value``, ``-J C=value``, ``-J --constraint=value`` or ``-J -C=value``, are going to be combined with any constraint options specified in the :js:attr:`access` system partition configuration parameter.
   For example, if ``-C x`` is specified in the :js:attr:`access` and ``-J C=y`` is passed to the command-line, ReFrame will pass ``-C x&y`` as a constraint to the scheduler.
   Notice, however, that if constraint options are specified through multiple :option:`-J` options, only the last one will be considered.
   If you wish to completely overwrite any constraint options passed in :js:attr:`access`, you should consider passing explicitly the Slurm directive with ``-J '#SBATCH --constraint=new'``.

   .. versionchanged:: 3.0
      This option has become more flexible.

   .. versionchanged:: 3.1
      Use ``&`` to combine constraints.

------------------------
Flexible node allocation
------------------------

ReFrame can automatically set the number of tasks of a test, if its :attr:`num_tasks <reframe.core.pipeline.RegressionTest.num_tasks>` attribute is set to a value less than or equal to zero.
This scheme is conveniently called *flexible node allocation* and is valid only for the Slurm backend.
When allocating nodes automatically, ReFrame will take into account all node limiting factors, such as partition :js:attr:`access` options, and any job submission control options described above.
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

.. option:: -m, --module=NAME

   Load environment module ``NAME`` before acting on any tests.
   This option may be specified multiple times, in which case all specified modules will be loaded in order.
   ReFrame will *not* perform any automatic conflict resolution.

   This option can also be set using the :envvar:`RFM_USER_MODULES` environment variable or the :js:attr:`user_modules` general configuration parameter.


.. option:: -u, --unload-module=NAME

   Unload environment module ``NAME`` before acting on any tests.
   This option may be specified multiple times, in which case all specified modules will be unloaded in order.

   This option can also be set using the :envvar:`RFM_UNLOAD_MODULES` environment variable or the :js:attr:`unload_modules` general configuration parameter.


.. option:: --module-path=PATH

   Manipulate the ``MODULEPATH`` environment variable before acting on any tests.
   If ``PATH`` starts with the ``-`` character, it will be removed from the ``MODULEPATH``, whereas if it starts with the ``+`` character, it will be added to it.
   In all other cases, ``PATH`` will completely override MODULEPATH.
   This option may be specified multiple times, in which case all the paths specified will be added or removed in order.

   .. versionadded:: 3.3


.. option:: --purge-env

   Unload all environment modules before acting on any tests.
   This will unload also sticky Lmod modules.

   This option can also be set using the :envvar:`RFM_PURGE_ENVIRONMENT` environment variable or the :js:attr:`purge_environment` general configuration parameter.


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

   This option can also be set using the :envvar:`RFM_NON_DEFAULT_CRAYPE` environment variable or the :js:attr:`non_default_craype` general configuration parameter.

.. option:: -M, --map-module=MAPPING

   Apply a module mapping.
   ReFrame allows manipulating test modules on-the-fly using module mappings.
   A module mapping has the form ``old_module: module1 [module2]...`` and will cause ReFrame to replace a module with another list of modules upon load time.
   For example, the mapping ``foo: foo/1.2`` will load module ``foo/1.2`` whenever module ``foo`` needs to be loaded.
   A mapping may also be self-referring, e.g., ``gnu: gnu gcc/10.1``, however cyclic dependencies in module mappings are not allowed and ReFrame will issue an error if it detects one.
   This option is especially useful for running tests using a newer version of a software or library.

   This option may be specified multiple times, in which case multiple mappings will be applied.

   This option can also be set using the :envvar:`RFM_MODULE_MAPPINGS` environment variable or the :js:attr:`module_mappings` general configuration parameter.

   .. versionchanged:: 3.3
      If the mapping replaces a module collection, all new names must refer to module collections, too.

   .. seealso::
      Module collections with `Environment Modules <https://modules.readthedocs.io/en/latest/MIGRATING.html#module-collection>`__ and `Lmod <https://lmod.readthedocs.io/en/latest/010_user.html#user-collections>`__.


.. option:: --module-mappings=FILE

   A file containing module mappings.
   Each line of the file contains a module mapping in the form described in the :option:`-M` option.
   This option may be combined with the :option:`-M` option, in which case module mappings specified will be applied additionally.

   This option can also be set using the :envvar:`RFM_MODULE_MAP_FILE` environment variable or the :js:attr:`module_map_file` general configuration parameter.


---------------------
Miscellaneous options
---------------------

.. option:: -C --config-file=FILE

   Use ``FILE`` as configuration file for ReFrame.

   This option can also be set using the :envvar:`RFM_CONFIG_FILE` environment variable.

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
   It does so by trying to match the hostname of the current machine again the hostname patterns defined in the :js:attr:`hostnames` system configuration parameter.
   The system with the first match becomes the current system.
   For Cray systems, ReFrame will first look for the *unqualified machine name* in ``/etc/xthostname`` before trying retrieving the hostname of the current machine.

   This option can also be set using the :envvar:`RFM_SYSTEM` environment variable.

.. option:: --failure-stats

   Print failure statistics at the end of the run.


.. option:: --performance-report

   Print a performance report for all the performance tests that have been run.
   The report shows the performance values retrieved for the different performance variables defined in the tests.


.. option:: --nocolor

   Disable output coloring.

   This option can also be set using the :envvar:`RFM_COLORIZE` environment variable or the :js:attr:`colorize` general configuration parameter.

.. option:: --upgrade-config-file=OLD[:NEW]

   Convert the old-style configuration file ``OLD``, place it into the new file ``NEW`` and exit.
   If a new file is not given, a file in the system temporary directory will be created.

.. option:: -v, --verbose

   Increase verbosity level of output.
   This option can be specified multiple times.
   Every time this option is specified, the verbosity level will be increased by one.
   There are the following message levels in ReFrame listed in increasing verbosity order:
   ``critical``, ``error``, ``warning``, ``info``, ``verbose`` and ``debug``.
   The base verbosity level of the output is defined by the :js:attr:`level` `stream logging handler <config_reference.html#common-logging-handler-properties>`__ configuration parameter.

   This option can also be set using the :envvar:`RFM_VERBOSE` environment variable or the :js:attr:`verbose` general configuration parameter.


.. option:: -V, --version

   Print version and exit.


.. option:: -h, --help

   Print a short help message and exit.


Environment
-----------

Several aspects of ReFrame can be controlled through environment variables.
Usually environment variables have counterparts in command line options or configuration parameters.
In such cases, command-line options take precedence over environment variables, which in turn precede configuration parameters.
Boolean environment variables can have any value of ``true``, ``yes`` or ``y`` (case insensitive) to denote true and any value of ``false``, ``no`` or ``n`` (case insensitive) to denote false.

Here is an alphabetical list of the environment variables recognized by ReFrame:


.. envvar:: RFM_CHECK_SEARCH_PATH

   A colon-separated list of filesystem paths where ReFrame should search for tests.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`-c`
      Associated configuration parameter :js:attr:`check_search_path` general configuration parameter
      ================================== ==================


.. envvar:: RFM_CHECK_SEARCH_RECURSIVE

   Search for test files recursively in directories found in the check search path.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`-R`
      Associated configuration parameter :js:attr:`check_search_recursive` general configuration parameter
      ================================== ==================


.. envvar:: RFM_CLEAN_STAGEDIR

   Clean stage directory of tests before populating it.

   .. versionadded:: 3.1

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`--dont-restage`
      Associated configuration parameter :js:attr:`clean_stagedir` general configuration parameter
      ================================== ==================


.. envvar:: RFM_COLORIZE

   Enable output coloring.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`--nocolor`
      Associated configuration parameter :js:attr:`colorize` general configuration parameter
      ================================== ==================


.. envvar:: RFM_CONFIG_FILE

   Set the configuration file for ReFrame.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`-C`
      Associated configuration parameter N/A
      ================================== ==================


.. envvar:: RFM_GRAYLOG_ADDRESS

   The address of the Graylog server to send performance logs.
   The address is specified in ``host:port`` format.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     N/A
      Associated configuration parameter :js:attr:`address` graylog log handler configuration parameter
      ================================== ==================


.. versionadded:: 3.1


.. envvar:: RFM_GRAYLOG_SERVER

   .. deprecated:: 3.1
      Please :envvar:`RFM_GRAYLOG_ADDRESS` instead.


.. envvar:: RFM_IGNORE_CHECK_CONFLICTS

   Ignore tests with conflicting names when loading.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`--ignore-check-conflicts`
      Associated configuration parameter :js:attr:`ignore_check_conflicts` general configuration parameter
      ================================== ==================


.. envvar:: RFM_TRAP_JOB_ERRORS

   Ignore job exit code

   .. table::
      :align: left

      ================================== ==================
      Associated configuration parameter :js:attr:`trap_job_errors` general configuration parameter
      ================================== ==================


.. envvar:: RFM_IGNORE_REQNODENOTAVAIL

   Do not treat specially jobs in pending state with the reason ``ReqNodeNotAvail`` (Slurm only).

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     N/A
      Associated configuration parameter :js:attr:`ignore_reqnodenotavail` scheduler configuration parameter
      ================================== ==================


.. envvar:: RFM_KEEP_STAGE_FILES

   Keep test stage directories even for tests that finish successfully.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`--keep-stage-files`
      Associated configuration parameter :js:attr:`keep_stage_files` general configuration parameter
      ================================== ==================


.. envvar:: RFM_MODULE_MAP_FILE

   A file containing module mappings.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`--module-mappings`
      Associated configuration parameter :js:attr:`module_map_file` general configuration parameter
      ================================== ==================


.. envvar:: RFM_MODULE_MAPPINGS

   A comma-separated list of module mappings.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`-M`
      Associated configuration parameter :js:attr:`module_mappings` general configuration parameter
      ================================== ==================


.. envvar:: RFM_NON_DEFAULT_CRAYPE

   Test a non-default Cray Programming Environment.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`--non-default-craype`
      Associated configuration parameter :js:attr:`non_default_craype` general configuration parameter
      ================================== ==================


.. envvar:: RFM_OUTPUT_DIR

   Directory prefix for test output files.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`-o`
      Associated configuration parameter :js:attr:`outputdir` system configuration parameter
      ================================== ==================


.. envvar:: RFM_PERFLOG_DIR

   Directory prefix for logging performance data.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`--perflogdir`
      Associated configuration parameter :js:attr:`basedir` logging handler configuration parameter
      ================================== ==================


.. envvar:: RFM_PREFIX

   General directory prefix for ReFrame-generated directories.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`--prefix`
      Associated configuration parameter :js:attr:`prefix` system configuration parameter
      ================================== ==================


.. envvar:: RFM_PURGE_ENVIRONMENT

   Unload all environment modules before acting on any tests.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`--purge-env`
      Associated configuration parameter :js:attr:`purge_environment` general configuration parameter
      ================================== ==================


.. envvar:: RFM_REPORT_FILE

   The file where ReFrame will store its report.

   .. versionadded:: 3.1

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`--report-file`
      Associated configuration parameter :js:attr:`report_file` general configuration parameter
      ================================== ==================


.. envvar:: RFM_SAVE_LOG_FILES

   Save ReFrame log files in the output directory before exiting.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`--save-log-files`
      Associated configuration parameter :js:attr:`save_log_files` general configuration parameter
      ================================== ==================


.. envvar:: RFM_STAGE_DIR

   Directory prefix for staging test resources.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`-s`
      Associated configuration parameter :js:attr:`stagedir` system configuration parameter
      ================================== ==================


.. envvar:: RFM_SYSLOG_ADDRESS

   The address of the Syslog server to send performance logs.
   The address is specified in ``host:port`` format.
   If no port is specified, the address refers to a UNIX socket.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     N/A
      Associated configuration parameter :js:attr:`address` syslog log handler configuration parameter
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
      Associated configuration parameter :js:attr:`timestamp_dirs` general configuration parameter.
      ================================== ==================

.. |--timestamp| replace:: :attr:`--timestamp`
.. _--timestamp: #cmdoption-timestamp



.. envvar:: RFM_UNLOAD_MODULES

   A comma-separated list of environment modules to be unloaded before acting on any tests.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`-u`
      Associated configuration parameter :js:attr:`unload_modules` general configuration parameter
      ================================== ==================


.. envvar:: RFM_USE_LOGIN_SHELL

   Use a login shell for the generated job scripts.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     N/A
      Associated configuration parameter :js:attr:`use_login_shell` general configuration parameter
      ================================== ==================


.. envvar:: RFM_USER_MODULES

   A comma-separated list of environment modules to be loaded before acting on any tests.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`-m`
      Associated configuration parameter :js:attr:`user_modules` general configuration parameter
      ================================== ==================


.. envvar:: RFM_VERBOSE

   Increase verbosity level of output.

   .. table::
      :align: left

      ================================== ==================
      Associated command line option     :option:`-v`
      Associated configuration parameter :js:attr:`verbose` general configuration parameter
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
.. _reframe/core/settings.py: https://github.com/eth-cscs/reframe/blob/master/reframe/core/settings.py
.. |reframe.settings(8)| replace:: ``reframe.settings(8)``
.. _reframe.settings(8): config_reference.html


Reporting Bugs
--------------

For bugs, feature request, help, please open an issue on Github: <https://github.com/eth-cscs/reframe>


See Also
--------

See full documentation online: <https://reframe-hpc.readthedocs.io/>
