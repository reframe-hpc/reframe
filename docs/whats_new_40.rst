What's New in ReFrame 4.0
=========================

ReFrame 4.0 introduces some important new features and removes all features, configuration options and interfaces that were deprecated in the 3.x versions.
It also introduces a couple of new deprecations.

ReFrame 4.0 maintains backward compatibility as much as possible.
Existing 3.x configurations and 3.x tests are expected to run out-of-the-box, despite any warnings issued.
Framework's behavior with respect to performance logging has also changed, but configuration options are offered so that users can switch to the old behavior.

This page summarizes the key changes in ReFrame 4.0 and what users should pay attention to.


New Features and Enchancements
------------------------------

Chaining Configuration Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There is no need anymore to keep a huge configuration file with all your system and environment definitions and it is no more required to carry on the generic system configuration as well as any part of the builtin configuration.
ReFrame 4.0 allows you to split your configuration in multiple files.
This allows you to create minimal configuration files that contain only the necessary parts.
For example, if you want to define a general configuration parameter, you don't need to copy the builtin configuration file and add it, but you simply add it in a single ``general`` section.
This can also be very useful if you maintain a ReFrame installation user by others, as you can update your settings (systems, environments and other options) without and any of your users' custom configuration will automatically inherit your settings if it is properly chained.
To assist with system-wide installation the ``RFM_CONFIG_PATH`` environment variable is introduced that allows you to specify a path where ReFrame will look for configuration files to load.

Now that systems and environments definitions can be distributed over multiple configuration files, it can become easy to accidentally redefine a system or environment without a notice.
For this reason, ReFrame warns you if a system or an environment are redefined in the same scope.
Since in the past all configuration files where extended copies of the builtin, you will get warnings that the ``generic`` system and the ``builtin`` environment are redefined, as ReFrame finds them in the builtin configuration, which is always loaded.
You can safely ignore these warnings and use the definitions in your configuration file.
If you want to eliminate them, though, you should remove the conflicting definitions from your configuration file.

Although ReFrame will not warn you for redefining other configuration sections, you are also advised to tidy up your configuration file and remove any parts that were copied unchanged from the builtin configuration.

For more information on how ReFrame 4.0 builds and loads its configuration, please refer to the documentation of the :option:`--config-file` option, as well as the :ref:`building-the-final-config` section.


Performance Reporting and Logging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ReFrame 4.0 improves on how performance values are logged and reported.
This is a breaking change, but you can easily revert to the old behavior.

ReFrame now logs performance *after* the test has finished and not during the performance stage.
You can now log the result of the test by including ``%(check_result)s`` in your log handler format string.
However, now, by default, ReFrame will log all the performance variables in a *single* record;
in the past, a new record was logged for each performance variable.
Also, the ``%(check_perf_*)s`` format placeholders are valid only in the :attr:`~config.logging.handlers_perflog.format_perfvars` configuration parameter and will be used to format the performance values if the ``%(check_perfvalues)s`` placeholder is present in the handler's :attr:`~config.logging.handlers_perflog.format` parameter.
This change in behavior will likely break your log processing, especially if you are using the ``graylog`` or ``httpjson`` handlers or any handler that sends the full record to a log server.
You can revert to the old behavior by setting the :attr:`~config.logging.perflog_compat` configuration parameter.
This will send a separate record for each performance variable that will include all the individual ``%(check_perf_*)s`` attributes.
For more information, check the documentation of the :attr:`~config.logging.handlers_perflog.format_perfvars` configuration parameter.

The behavior of the ``filelog`` is also substantially improved.
The log file is printed by default in CSV format and a header is always printed at the beginning of each log file.
If the log format changes or the performance variables logged by the test change, a new log file will be created with an adapted header.
This way, every log file is consistent with the data in contains.
For more information, please refer to the ``filelog`` handler :ref:`documentation <filelog-handler>`.

When you run a performance test, ReFrame will now print immediately after the test has finished a short summary of its performance.
You can suppress this output by setting the log level at which this information is printed to ``verbose`` by setting the :attr:`~config.general.perf_info_level` general configuration parameter.

Finally, the performance report printed at the end of the run using the :option:`--performance-report` is revised providing more information in more compact form.


New Test Naming Scheme
^^^^^^^^^^^^^^^^^^^^^^

ReFrame 4.0 introduces makes default the new test naming scheme introduced in 3.10.0 and drops support of the old naming scheme.
The new naming scheme does not affect normal tests, but it changes how parameterized tests and fixtures are named.
Each test is now also associated with a unique hash code.
For parameterized tests and fixtures this hash code is appended to the test's or fixture's base name when creating any test-specific directories and files, such as the test stage and output directories).
The :option:`-n` option can match a test either by its display name (the default), or by its unique internal name or by its unique hash code.
Check the documentation of the :option:`-n` for more information.
For the details of the new naming scheme, please refer to the :ref:`test_naming_scheme` section.

Note that any tests that used the old naming scheme to depend on parameterized tests will break with this change.
Check the tutorial :ref:`param_deps` on how to create dependencies on parameterized tests in a portable way.


Custom parallel launchers
^^^^^^^^^^^^^^^^^^^^^^^^^

By relaxing the configuration schema, users can now define custom parallel launchers inside their Python configuration file.
Check the tutorial :ref:`custom_launchers` to find out how this can be achieved.


Unique run reports
^^^^^^^^^^^^^^^^^^

ReFrame now generates a unique report for each run inside the ``$HOME/.reframe/reports`` directory.
If you want to revert to the old behavior, where a single file was generated and was overwritten in every run, you should set the :attr:`~config.general.report_file` configuration option or the :envvar:`RFM_REPORT_FILE` environment variable.


New Backends
^^^^^^^^^^^^

eFrame 4.0 adds support for the `Apptainer <https://apptainer.org/>`__ container platform and the `Flux framework <http://flux-framework.org/>`__.


Dropped Features and Deprecations
---------------------------------

ReFrame 4.0 drops support for all the deprecated features and behaviors of ReFrame 3.x versions.
More specifically, the following deprecated features are dropped:

- The :attr:`@parameterized_test` decorator is dropped in favor of the :attr:`~reframe.core.builtins.parameter` builtin.
- The :attr:`~reframe.core.pipeline.RegressionTest.name` of the test is now read-only.
- The decorators :attr:`@final <reframe.core.builtins.final>`, :attr:`@require_deps <reframe.core.builtins.require_deps>`, :attr:`@run_after <reframe.core.builtins.run_after>` and :attr:`@run_before <reframe.core.builtins.run_before>` are no more accesible via the :mod:`reframe` module.
  They are directly available in the :class:`~reframe.core.pipeline.RegressionTest` namespace without the need of importing anything.
- The :attr:`@reframe.utility.sanity.sanity_function` decorator is dropped in favor of the :attr:`@deferrable <reframe.core.builtins.deferrable>` builtin.
- The :attr:`commands` attribute of the :class:`~reframe.core.containers.ContainerPlatform` is dropped in favor of the :attr:`~reframe.core.containers.ContainerPlatform.command` attribute.
- The :attr:`launcher` attribute of the :class:`~reframe.core.systems.System` is dropped in favor of the :attr:`~reframe.core.systems.System.launcher_type` attribute.
- The :attr:`@required_version` decorator is dropped in favor of the :attr:`~reframe.core.builtins.require_version` builtin.
  Also, automatically converting version strings that do not comply with the semantic versioning scheme is no more supported.
- The :data:`DEPEND_EXACT`, :data:`DEPEND_BY_ENV` and :data:`DEPEND_FULLY` integer constants that were passed as the ``how`` argument of the :meth:`~reframe.core.pipeline.RegressionTest.depends_on` method are no more supported and a callable should be used instead.
  The ``subdeps`` argument is also dropped.
- The low-level :func:`poll` and :func:`wait` :class:`RegressionTest` methods are dropped in favor of the :func:`~reframe.core.pipeline.RegressionTest.run_complete` and :func:`~reframe.core.pipeline.RegressionTest.run_wait`, respectively.
- The ``schedulers`` configuration section is dropped in favor of the partition-specific :attr:`~config.systems.partitions.sched_options`.
  Users should move any options set in the old section to the corresponding partition options.
- The :option:`--ignore-check-conflicts` command line option and the corresponding :envvar:`RFM_IGNORE_CHECK_CONFLICTS` environment variable are dropped.
- The :envvar:`RFM_GRAYLOG_SERVER` environment variable is dropped in favor of the :envvar:`RFM_GRAYLOG_ADDRESS`.


New Deprecations
^^^^^^^^^^^^^^^^

- All occurrences of the ``variables`` name are deprecated in favor of ``env_vars``.
  This includes the :attr:`~reframe.core.pipeline.RegressionTest.variables` test attribute and the homonym systems, partitions and environments configuration parameters as well as the :attr:`~reframe.core.environments.Environment.variables` of the :attr:`~reframe.core.environments.Environment` base class.
- Although :attr:`~reframe.core.pipeline.RegressionTest.perf_patterns` attribute is not deprecated, users are recommended to migrate to using the new :attr:`@performance_function <reframe.core.builtins.performance_function>` builtin.
  Please refer to :ref:`perftest-basics` tutorial for a starting point.
