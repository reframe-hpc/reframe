=======================
Configuration Reference
=======================

.. versionadded:: 3.0


ReFrame's behavior can be configured through its configuration file (see `Configuring ReFrame for Your Site <configure.html>`__), environment variables and command-line options.
An option can be specified via multiple paths (e.g., a configuration file parameter and an environment variable), in which case command-line options precede environment variables, which in turn precede configuration file options.
This section provides a complete reference guide of the configuration options of ReFrame that can be set in its configuration file or specified using environment variables.

ReFrame's configuration is in JSON syntax.
The full schema describing it can be found in `schemas/config.json <https://github.com/eth-cscs/reframe/blob/master/schemas/config.json>`__ file.
Any configuration file given to ReFrame is validated against this schema.

The syntax we use in the following to describe the different configuration object attributes is a valid query string for the `jq <https://stedolan.github.io/jq/>`__ JSON command-line processor.
Along the configuration option, the corresponding environment variable and command-line options are listed, if any.


Top-level Configuration
-----------------------

The top-level configuration object is essentially the full configuration of ReFrame.
It consists of the following properties:

.. py:attribute:: .systems

   :required: Yes

   A list of `system configuration objects <#system-configuration>`__.


.. py:attribute:: .environments

   :required: Yes

   A list of `environment configuration objects <#environment-configuration>`__.


.. py:attribute:: .logging

   :required: Yes

   A list of `logging configuration objects <#logging-configuration>`__.


.. py:attribute:: .schedulers

   :required: No

   A list of `scheduler configuration objects <#scheduler-configuration>`__.


.. py:attribute:: .modes

   :required: No

   A list of `execution mode configuration objects <#execution-mode-configuration>`__.

.. py:attribute:: .general

   :required: No

   A list of `general configuration objects <#general-configuration>`__.


System Configuration
--------------------

.. js:attribute:: .systems[].name

   :required: Yes

   The name of this system.
   Only alphanumeric characters, dashes (``-``) and underscores (``_``) are allowed.

.. js:attribute:: .systems[].descr

   :required: No
   :default: ``""``

   The description of this system.

.. js:attribute:: .systems[].hostnames

   :required: Yes

   A list of hostname regular expression patterns in Python `syntax <https://docs.python.org/3.8/library/re.html>`__, which will be used by the framework in order to automatically select a system configuration.
   For the auto-selection process, see `here <configure.html#picking-a-system-configuration>`__.

.. js:attribute:: .systems[].modules_system

   :required: No
   :default: ``"nomod"``

   The modules system that should be used for loading environment modules on this system.
   Available values are the following:

   - ``tmod``: The classic Tcl implementation of the `environment modules <https://sourceforge.net/projects/modules/files/Modules/modules-3.2.10/>`__ (version 3.2).
   - ``tmod31``: The classic Tcl implementation of the `environment modules <https://sourceforge.net/projects/modules/files/Modules/modules-3.2.10/>`__ (version 3.1).
     A separate backend is required for Tmod 3.1, because Python bindings are different from Tmod 3.2.
   - ``tmod32``: A synonym of ``tmod``.
   - ``tmod4``: The `new environment modules <http://modules.sourceforge.net/>`__ implementation (versions older than 4.1 are not supported).
   - ``lmod``: The `Lua implementation <https://lmod.readthedocs.io/en/latest/>`__ of the environment modules.
   - ``nomod``: This is to denote that no modules system is used by this system.

.. js:attribute:: .systems[].modules

   :required: No
   :default: ``[]``

   Environment modules to be loaded always when running on this system.
   These modules modify the ReFrame environment.
   This is useful in cases where a particular module is needed, for example, to submit jobs on a specific system.

.. js:attribute:: .systems[].variables

   :required: No
   :default: ``[]``

   A list of environment variables to be set always when running on this system.
   Each environment variable is specified as a two-element list containing the variable name and its value.
   You may reference other environment variables when defining an environment variable here.
   ReFrame will expand its value.
   Variables are set after the environment modules are loaded.

.. js:attribute:: .systems[].prefix

.. envvar:: RFM_PREFIX

.. option:: --prefix

   :required: No
   :default: ``"."``

   Directory prefix for a ReFrame run on this system.
   Any directories or files produced by ReFrame will use this prefix, if not specified otherwise.

.. js:attribute:: .systems[].stagedir

.. envvar:: RFM_STAGE_DIR

.. option:: -s DIR | --stage DIR

   :required: No
   :default: ``"${RFM_PREFIX}/stage"``

   Stage directory prefix for this system.
   This is the directory prefix, where ReFrame will create the stage directories for each individual test case.


.. js:attribute:: .systems[].outputdir

.. envvar:: RFM_OUTPUT_DIR

.. option:: -o DIR | --output DIR

   :required: No
   :default: ``"${RFM_PREFIX}/output"``

   Output directory prefix for this system.
   This is the directory prefix, where ReFrame will save information about the successful tests.


.. js:attribute:: .systems[].resourcesdir

   :required: No
   :default: ``"."``

   Directory prefix where external test resources (e.g., large input files) are stored.
   You may reference this prefix from within a regression test by accessing the corresponding `attribute <reference.html#reframe.core.systems.System.resourcesdir>`__ of the current system.


.. js:attribute:: .systems[].partitions

   :required: Yes

   A list of `system partition configuration objects <#system-partition-configuration>`__.
   This list must have at least one element.


------------------------------
System Partition Configuration
------------------------------

.. js:attribute:: .systems[].partitions[].name

   :required: Yes

   The name of this partition.
   Only alphanumeric characters, dashes (``-``) and underscores (``_``) are allowed.

.. js:attribute:: .systems[].partitions[].descr

   :required: No
   :default: ``""``

   The description of this partition.

.. js:attribute:: .systems[].partitions[].scheduler

   :required: Yes

   The job scheduler that will be used to launch jobs on this partition.
   Supported schedulers are the following:

   - ``local``: Jobs will be launched locally without using any job scheduler.
   - ``pbs``: Jobs will be launched using the `PBS Pro <https://en.wikipedia.org/wiki/Portable_Batch_System>`__ scheduler.
   - ``torque``: Jobs will be launched using the `Torque <https://en.wikipedia.org/wiki/TORQUE>`__ scheduler.
   - ``slurm``: Jobs will be launched using the `Slurm <https://www.schedmd.com/>`__ scheduler.
     This backend requires job accounting to be enabled in the target system.
     If not, you should consider using the ``squeue`` backend below.
   - ``squeue``: Jobs will be launched using the `Slurm <https://www.schedmd.com/>`__ scheduler.
     This backend does not rely on job accounting to retrieve job statuses, but ReFrame does its best to query the job state as reliably as possible.

.. js:attribute:: .systems[].partitions[].launcher

   :required: Yes

   The parallel job launcher that will be used in this partition to launch parallel programs.
   Available values are the following:

   - ``alps``: Parallel programs will be launched using the `Cray ALPS <https://pubs.cray.com/content/S-2393/CLE%205.2.UP03/cle-xc-system-administration-guide-s-2393-5203-xc/the-aprun-client>`__ ``aprun`` command.
   - ``ibrun``: Parallel programs will be launched using the ``ibrun`` command.
     This is a custom parallel program launcher used at `TACC <https://portal.tacc.utexas.edu/user-guides/stampede2>`__.
   - ``local``: No parallel program launcher will be used.
     The program will be launched locally.
   - ``mpirun``: Parallel programs will be launched using the ``mpirun`` command.
   - ``mpiexec``: Parallel programs will be launched using the ``mpiexec`` command.
   - ``srun``: Parallel programs will be launched using `Slurm <https://slurm.schedmd.com/srun.html>`__'s ``srun`` command.
   - ``srunalloc``: Parallel programs will be launched using `Slurm <https://slurm.schedmd.com/srun.html>`__'s ``srun`` command, but job allocation options will also be emitted.
     This can be useful when combined with the ``local`` job scheduler.
   - ``ssh``: Parallel programs will be launched using SSH.
     This launcher uses the partition’s `access <config_reference.rst#.systems[].partitions[].access>`__ property in order to determine the remote host and any additional options to be passed to the SSH client.
     The ssh command will be launched in "batch mode," meaning that password-less access to the remote host must be configured.
     Here is an example configuration for the ssh launcher:

     .. code:: python

				{
				    'name': 'foo'
				    'scheduler': 'local',
				    'launcher': 'ssh'
				    'access': ['-l admin', 'remote.host'],
				    'environs': ['builtin'],
				}

.. js:attribute:: .systems[].partitions[].access

   :required: No
   :default: ``[]``

   A list of job scheduler options that will be passed to the generated job script for gaining access to that logical partition.


.. js:attribute:: .systems[].partitions[].environs

   :required: No
   :default: ``[]``

  A list of environment names that ReFrame will use to run regression tests on this partition.
  Each environment must be defined in the `environments <config_reference.html#.environments>`__ section of the configuration and the definition of the environment must be valid for this partition.


.. js:attribute:: .systems[].partitions[].container_platforms

   :required: No
   :default: ``[]``

   A list for `container platform configuration objects <#container-platform-configuration>`__.
   This will allow launching regression tests that use `containers <advanced.html#testing-containerized-applications>`__ on this partition.


.. js:attribute:: .systems[].partitions[].modules

   :required: No
   :default: ``[]``

   A list of environment modules to be loaded before running a regression test on this partition.


.. js:attribute:: .systems[].partitions[].variables

   :required: No
   :default: ``[]``

   A list of environment variables to be set before running a regression test on this partition.
   Each environment variable is specified as a two-element list containing the variable name and its value.
   You may reference other environment variables when defining an environment variable here.
   ReFrame will expand its value.
   Variables are set after the environment modules are loaded.


.. js:attribute:: .systems[].partitions[].max_jobs

   :required: No
   :default: ``1``

   The maximum number of concurrent regression tests that may be active (i.e., not completed) on this partition.
   This option is relevant only when ReFrame executes with the `asynchronous execution policy <running.html#asynchronous-execution-of-regression-checks>`__.


.. js:attribute:: .systems[].partitions[].resources

   :required: No
   :default: ``[]``

   A list of job scheduler `resource specification <#config_reference.html#custom-job-scheduler-resources>`__ objects.



Container Platform Configuration
================================

ReFrame can launch containerized applications, but you need to configure properly a system partition in order to do that by defining a container platform configuration.

.. js:attribute:: .systems[].partitions[].container_platforms[].type

   :required: Yes

   The type of the container platform.
   Available values are the following:

   - ``Docker``: The `Docker <https://www.docker.com/>`__ container runtime.
   - ``Sarus``: The `Sarus <https://sarus.readthedocs.io/>`__ container runtime.
   - ``Shifter``: The `Shifter <https://github.com/NERSC/shifter>`__ container runtime.
   - ``Singularity``: The `Singularity <https://sylabs.io/>`__ container runtime.


.. js:attribute:: .systems[].partitions[].container_platforms[].modules

   :required: No
   :default: ``[]``

  List of environment modules to be loaded when running containerized tests using this container platform.


.. js:attribute:: .systems[].partitions[].container_platforms[].variables

   :required: No
   :default: ``[]``

   List of environment variables to be set when running containerized tests using this container platform.
   Each environment variable is specified as a two-element list containing the variable name and its value.
   You may reference other environment variables when defining an environment variable here.
   ReFrame will expand its value.
   Variables are set after the environment modules are loaded.


Custom Job Scheduler Resources
==============================

ReFrame allows you to define custom scheduler resources for each partition that you can then transparently access through the :attr:`extra_resources` attribute of a regression test.

.. js:attribute:: .systems[].partitions[].resources[].name

   :required: Yes

  The name of this resources.
  This name will be used to request this resource in a regression test's :attr:`extra_resources`.


.. js:attribute:: .systems[].partitions[].resources[].options

   :required: No
   :default: ``[]``

   A list of options to be passed to this partition’s job scheduler.
   The option strings can contain placeholders of the form ``{placeholder_name}``.
   These placeholders may be replaced with concrete values by a regression test through the :attr:`extra_resources` attribute.

   For example, one could define a ``gpu`` resource for a multi-GPU system that uses Slurm as follows:

   .. code:: python

      'resources': [
          {
              'name': 'gpu',
              'options': ['--gres=gpu:{num_gpus_per_node}']
          }
      ]


   A regression test then may request this resource as follows:

   .. code:: python

      self.extra_resources = {'gpu': {'num_gpus_per_node': '8'}}


   And the generated job script will have the following line in its preamble:

   .. code:: bash

      #SBATCH --gres=gpu:8


   A resource specification may also start with ``#PREFIX``, in which case ``#PREFIX`` will replace the standard job script prefix of the backend scheduler of this partition.
   This is useful in cases of job schedulers like Slurm, that allow alternative prefixes for certain features.
   An example is the `DataWarp <https://www.cray.com/datawarp>`__ functionality of Slurm which is supported by the ``#DW`` prefix.
   One could then define DataWarp related resources as follows:

   .. code:: python

      'resources': [
          {
              'name': 'datawarp',
              'options': [
                  '#DW jobdw capacity={capacity} access_mode={mode} type=scratch',
                  '#DW stage_out source={out_src} destination={out_dst} type={stage_filetype}'
              ]
          }
      ]


   A regression test that wants to make use of that resource, it can set its :attr:`extra_resources` as follows:

   .. code:: python

     self.extra_resources = {
         'datawarp': {
             'capacity': '100GB',
             'mode': 'striped',
             'out_src': '$DW_JOB_STRIPED/name',
             'out_dst': '/my/file',
             'stage_filetype': 'file'
         }
     }

 .. note::

    For the ``pbs`` and ``torque`` backends, options accepted in the `access <#.systems[].partitions[].access>`__ and `resources <#.systems[].partitions[].resources>`__ attributes may either refer to actual ``qsub`` options or may be just resources specifications to be passed to the ``-l`` option.
    The backend assumes a ``qsub`` option, if the options passed in these attributes start with a ``-``.


Environment Configuration
-------------------------

Environments defined in this section will be used for running regression tests.
They are associated with `system partitions <#system-partition-configuration>`__.


.. js:attribute:: .environments[].name

   :required: Yes

   The name of this environment.


.. js:attribute:: .environments[].modules

   :required: No
   :default: ``[]``

   A list of environment modules to be loaded when this environment is loaded.


.. js:attribute:: .environments[].variables

   :required: No
   :default: ``[]``

   A list of environment variables to be set when loading this environment.
   Each environment variable is specified as a two-element list containing the variable name and its value.
   You may reference other environment variables when defining an environment variable here.
   ReFrame will expand its value.
   Variables are set after the environment modules are loaded.


.. js:attribute:: .environments[].cc

   :required: No
   :default: ``"cc"``

   The C compiler to be used with this environment.


.. js:attribute:: .environments[].cxx

   :required: No
   :default: ``"CC"``

   The C++ compiler to be used with this environment.


.. js:attribute:: .environments[].ftn

   :required: No
   :default: ``"ftn"``

   The Fortran compiler to be used with this environment.


.. js:attribute:: .environments[].cppflags

   :required: No
   :default: ``[]``

   A list of C preprocessor flags to be used with this environment by default.


.. js:attribute:: .environments[].cflags

   :required: No
   :default: ``[]``

   A list of C flags to be used with this environment by default.


.. js:attribute:: .environments[].cxxflags

   :required: No
   :default: ``[]``

   A list of C++ flags to be used with this environment by default.


.. js:attribute:: .environments[].fflags

   :required: No
   :default: ``[]``

   A list of Fortran flags to be used with this environment by default.


.. js:attribute:: .environments[].ldflags

   :required: No
   :default: ``[]``

   A list of linker flags to be used with this environment by default.


.. js:attribute:: .environments[].target_systems

   :required: No
   :default: ``["*"]``

   A list of systems or system/partitions combinations that this environment definition is valid for.
   A ``*`` entry denotes any system.
   In case of multiple definitions of an environment, the most specific to the current system partition will be used.
   For example, if the current system/partition combination is ``daint:mc``, the second definition of the ``PrgEnv-gnu`` environment will be used:

   .. code::  python

      'environments': [
          {
              'name': 'PrgEnv-gnu',
              'modules': ['PrgEnv-gnu']
          },
          {
              'name': 'PrgEnv-gnu',
              'modules': ['PrgEnv-gnu', 'openmpi'],
              'cc':  'mpicc',
              'cxx': 'mpicxx',
              'ftn': 'mpif90',
              'target_systems': ['daint:mc']
          }
      ]

   However, if the current system was ``daint:gpu``, the first definition would be selected, despite the fact that the second definition is relevant for another partition of the same system.
   To better understand this, ReFrame resolves definitions in a hierarchical way.
   It first looks for definitions for the current partition, then for the containing system and, finally, for global definitions (the ``*`` pseudo-system).


Logging Configuration
---------------------

Logging in ReFrame is handled by logger objects which further delegate message to *logging handlers* which are eventually responsible for emitting or sending the log records to their destinations.
You may define different logger objects per system but *not* per partition.


.. js:attribute:: .logging[].level

   :required: No
   :default: ``"debug"``

   The level associated with this logger object.
   There are the following levels in decreasing severity order:

   - ``critical``: Catastrophic errors; the framework cannot proceed with its execution.
   - ``error``: Normal errors; the framework may or may not proceed with its execution.
   - ``warning``: Warning messages.
   - ``info``: Informational messages.
   - ``verbose``: More informational messages.
   - ``debug``: Debug messages.

   If a message is logged by the framework, its severity level will be checked by the logger and if it is higher from the logger's level, it will be passed down to its handlers.


.. js:attribute:: .logging[].handlers

   :required: Yes

   A list of `logging handlers <#logging-handlers>`__ responsible for handling normal framework output.


.. js:attribute:: .logging[].handlers_perflog

   :required: Yes

   A list of logging handlers responsible for handling `performance data <#performance-logging-handlers>`__ from tests.


.. js:attribute:: .logging[].target_systems

   :required: No
   :default: ``["*"]``

   A list of systems or system/partitions combinations that this logging configuration is valid for.
   For a detailed description of this property, you may refer `here <#.environments[].target_systems>`__.



---------------------------------
Common logging handler properties
---------------------------------

All logging handlers share the following set of common attributes:


.. js:attribute:: .logging[].handlers[].type

.. js:attribute:: .logging[].handlers_perflog[].type

   :required: Yes

   The type of handler.
   There are the following types available:

   - ``file``: This handler sends log records to file.
     See `here <#the-file-handler>`__ for more details.
   - ``filelog``: This handler sends performance log records to files.
     See `here <#the-filelog-handler>`__ for more details.
   - ``graylog``: This handler sends performance log records to Graylog.
     See `here <#the-graylog-handler>`__ for more details.
   - ``stream``: This handler sends log records to a file stream.
     See `here <#the-stream-handler>`__ for more details.
   - ``syslog``: This handler sends log records to a Syslog facility.
     See `here <#the-syslog-handler>`__ for more details.


.. js:attribute:: .logging[].handlers[].level

.. js:attribute:: .logging[].handlers_perflog[].level

   :required: No
   :default: ``"info"``

   The `log level <#.logging[].level>`__ associated with this handler.



.. js:attribute:: .logging[].handlers[].format

.. js:attribute:: .logging[].handlers_perflog[].format

   :required: No
   :default: ``"%(message)s"``

   Log record format string.
   ReFrame accepts all log record attributes from Python's `logging <https://docs.python.org/3.8/library/logging.html#logrecord-attributes>`__ mechanism and adds the following:

   - ``%(check_environ)s``: The name of the `environment <#environment-configuration>`__ that the current test is being executing for.
   - ``%(check_info)s``: General information of the currently executing check.
     By default this field has the form ``%(check_name)s on %(check_system)s:%(check_partition)s using %(check_environ)s``.
     It can be configured on a per test basis by overriding the :func:`info <reframe.core.pipeline.RegressionTest.info>` method of a specific regression test.
   - ``%(check_jobid)s``: The job or process id of the job or process associated with the currently executing regression test.
     If a job or process is not yet created, ``-1`` will be printed.
   - ``%(check_job_completion_time)s``: The completion time of the job spawned by this regression test.
     This timestamp will be formatted according to `datefmt <#.logging[].handlers[].datefmt>`__ handler property.
     The accuracy of this timestamp depends on the backend scheduler.
     The ``slurm`` scheduler `backend <#.systems[].partitions[].scheduler>`__ relies on job accounting and returns the actual termination time of the job.
     The rest of the backends report as completion time the moment when the framework realizes that the spawned job has finished.
     In this case, the accuracy depends on the execution policy used.
     If tests are executed with the serial execution policy, this is close to the real completion time, but if the asynchronous execution policy is used, it can differ significantly.
     If the job completion time cannot be retrieved, ``None`` will be printed.
   - ``%(check_job_completion_time_unix)s``: The completion time of the job spawned by this regression test expressed as UNIX time.
     This is a raw time field and will not be formatted according to ``datefmt``.
     If specific formatting is desired, the ``check_job_completion_time`` should be used instead.
   - ``%(check_name)s``: The name of the regression test on behalf of which ReFrame is currently executing.
     If ReFrame is not executing in the context of a regression test, ``reframe`` will be printed instead.
   - ``%(check_num_tasks)s``: The number of tasks assigned to the regression test.
   - ``%(check_outputdir)s``: The output directory associated with the currently executing test.
   - ``%(check_partition)s``: The system partition where this test is currently executing.
   - ``%(check_stagedir)s``: The stage directory associated with the currently executing test.
   - ``%(check_system)s``: The system where this test is currently executing.
   - ``%(check_tags)s``: The tags associated with this test.
   - ``%(check_perf_lower_thres)s``: The lower threshold of the performance difference from the reference value expressed as a fractional value.
     See the `reference <tutorial.html#writing-a-performance-test>`__ attribute of regression tests for more details.
   - ``%(check_perf_ref)s``: The reference performance value of a certain performance variable.
   - ``%(check_perf_unit)s``: The unit of measurement for the measured performance variable.
   - ``%(check_perf_upper_thres)s``: The lower threshold of the performance difference from the reference value expressed as a fractional value.
     See the `reference <tutorial.html#writing-a-performance-test>`__ attribute of regression tests for more details.
   - ``%(check_perf_value)s``: The performance value obtained for a certain performance variable.
   - ``%(check_perf_var)s``: The name of the `performance variable <tutorial.html#writing-a-performance-test>`__ being logged.
   - ``%(osuser)s``: The name of the OS user running ReFrame.
   - ``%(osgroup)s``: The name of the OS group running ReFrame.
   - ``%(version)s``: The ReFrame version.


.. js:attribute:: .logging[].handlers[].datefmt

.. js:attribute:: .logging[].handlers_perflog[].datefmt

   :required: No
   :default: ``"%FT%T"``

   Time format to be used for printing timestamps fields.
   There are two timestamp fields available: ``%(asctime)s`` and ``%(check_job_completion_time)s``.
   In addition to the format directives supported by the standard library's `time.strftime() <https://docs.python.org/3.8/library/time.html#time.strftime>`__ function, ReFrame allows you to use the ``%:z`` directive -- a GNU ``date`` extension --  that will print the time zone difference in a RFC3339 compliant way, i.e., ``+/-HH:MM`` instead of ``+/-HHMM``.


------------------------
The ``file`` log handler
------------------------

This log handler handles output to normal files.
The additional properties for the ``file`` handler are the following:


.. js:attribute:: .logging[].handlers[].name

.. js:attribute:: .logging[].handlers_perflog[].name

   :required: Yes

   The name of the file where this handler will write log records.


.. js:attribute:: .logging[].handlers[].append

.. js:attribute:: .logging[].handlers_perflog[].append

   :required: No
   :default: ``false``

   Controls whether this handler should append to its file or not.


.. js:attribute:: .logging[].handlers[].timestamp

.. js:attribute:: .logging[].handlers_perflog[].timestamp

   :required: No
   :default: ``false``

   Append a timestamp to this handler's log file.
   This property may also accept a date format as described in the `datefmt <#.logging[].handlers[].datefmt>`__ property.
   If the handler's `name <#.logging[].handlers[].name>`__ property is set to ``filename.log`` and this property is set to ``true`` or to a specific timestamp format, the resulting log file will be ``filename_<timestamp>.log``.


---------------------------
The ``filelog`` log handler
---------------------------

This handler is meant primarily for performance logging and logs the performance of a regression test in one or more files.
The additional properties for the ``filelog`` handler are the following:


.. js:attribute:: .logging[].handlers[].basedir

.. js:attribute:: .logging[].handlers_perflog[].basedir

.. envvar:: RFM_PERFLOG_DIR

.. option:: --perflogdir

   :required: No
   :default: ``"./perflogs"``

   The base directory of performance data log files.


.. js:attribute:: .logging[].handlers[].prefix

.. js:attribute:: .logging[].handlers_perflog[].prefix

   :required: Yes

   This is a directory prefix (usually dynamic), appended to the `basedir <#.logging[].handlers_perflog[].basedir>`__, where the performance logs of a test will be stored.
   This attribute accepts any of the check-specific `formatting placeholders <#.logging[].handlers_perflog[].format>`__.
   This allows to create dynamic paths based on the current system, partition and/or programming environment a test executes with.
   For example, a value of ``%(check_system)s/%(check_partition)s`` would generate the following structure of performance log files:


   .. code-block:: none

     {basedir}/
        system1/
            partition1/
                test_name.log
            partition2/
                test_name.log
            ...
        system2/
        ...


.. js:attribute:: .logging[].handlers[].append

.. js:attribute:: .logging[].handlers_perflog[].append

   :required: No
   :default: ``true``

   Open each log file in append mode.


---------------------------
The ``graylog`` log handler
---------------------------

This handler sends log records to a `Graylog <https://www.graylog.org/>`__ server.
The additional properties for the ``graylog`` handler are the following:

.. js:attribute:: .logging[].handlers[].address

.. js:attribute:: .logging[].handlers_perflog[].address

.. envvar:: RFM_GRAYLOG_SERVER

   :required: Yes

   The address of the Graylog server defined as ``host:port``.


.. js:attribute:: .logging[].handlers[].extras

.. js:attribute:: .logging[].handlers_perflog[].extras

   :required: No
   :default: ``{}``

   A set of optional key/value pairs to be passed with each log record to the server.
   These may depend on the server configuration.


This log handler uses internally `pygelf <https://pypi.org/project/pygelf/>`__.
If ``pygelf`` is not available, this log handler will be ignored.
`GELF <http://docs.graylog.org/en/latest/pages/gelf.html>`__ is a format specification for log messages that are sent over the network.
The ``graylog`` handler sends log messages in JSON format using an HTTP POST request to the specified address.
More details on this log format may be found `here <http://docs.graylog.org/en/latest/pages/gelf.html#gelf-payload-specification>`__.
An example configuration of this handler for performance logging is shown here:

.. code:: python

   {
       'type': 'graylog',
       'address': 'graylog-server:12345',
       'level': 'info',
       'format': '%(message)s',
       'extras': {
           'facility': 'reframe',
           'data-version': '1.0'
       }
   }


Although the ``format`` is defined for this handler, it is not only the log message that will be transmitted the Graylog server.
This handler transmits the whole log record, meaning that all the information will be available and indexable at the remote end.


--------------------------
The ``stream`` log handler
--------------------------

This handler sends log records to a file stream.
The additional properties for the ``stream`` handler are the following:


.. js:attribute:: .logging[].handlers[].name

.. js:attribute:: .logging[].handlers_perflog[].name

   :required: No
   :default: ``"stdout"``

   The name of the file stream to send records to.
   There are only two available streams:

   - ``stdout``: the standard output.
   - ``stderr``: the standard error.


--------------------------
The ``syslog`` log handler
--------------------------

This handler sends log records to UNIX syslog.
The additional properties for the ``syslog`` handler are the following:


.. js:attribute:: .logging[].handlers[].socktype

.. js:attribute:: .logging[].handlers_perflog[].socktype

   :required: No
   :default: ``"udp"``

   The socket type where this handler will send log records to.
   There are two socket types:

   - ``udp``: A UDP datagram socket.
   - ``tcp``: A TCP stream socket.


.. js:attribute:: .logging[].handlers[].facility

.. js:attribute:: .logging[].handlers_perflog[].facility

   :required: No
   :default: ``"user"``

   The Syslog facility where this handler will send log records to.
   The list of supported facilities can be found `here <https://docs.python.org/3.8/library/logging.handlers.html#logging.handlers.SysLogHandler.encodePriority>`__.


.. js:attribute:: .logging[].handlers[].address

.. js:attribute:: .logging[].handlers_perflog[].address

   :required: Yes

   The socket address where this handler will connect to.
   This can either be of the form ``<host>:<port>`` or simply a path that refers to a Unix domain socket.



Scheduler Configuration
-----------------------

A scheduler configuration object contains configuration options specific to the scheduler's behavior.


------------------------
Common scheduler options
------------------------


.. js:attribute:: .schedulers[].name

   :required: Yes

   The name of the scheduler that these options refer to.
   It can be any of the supported job scheduler `backends <#.systems[].partitions[].scheduler>`__.


.. js:attribute:: .schedulers[].job_submit_timeout

   :required: No
   :default: 60

   Timeout in seconds for the job submission command.
   If timeout is reached, the regression test issuing that command will be marked as a failure.


.. js:attribute:: .schedulers[].target_systems

   :required: No
   :default: ``["*"]``

   A list of systems or system/partitions combinations that this scheduler configuration is valid for.
   For a detailed description of this property, you may refer `here <#.environments[].target_systems>`__.



Execution Mode Configuration
----------------------------

ReFrame allows you to define groups of command line options that are collectively called *execution modes*.
An execution mode can then be selected from the command line with the ``-mode`` option.
The options of an execution mode will be passed to ReFrame as if they were specified in the command line.


.. js:attribute:: .modes[].name

   :required: Yes

   The name of this execution mode.
   This can be used with the ``-mode`` command line option to invoke this mode.


.. js:attribute:: .modes[].options

   :required: No
   :default: ``[]``

   The command-line options associated with this execution mode.


.. js:attribute:: .schedulers[].target_systems

   :required: No
   :default: ``["*"]``

   A list of systems or system/partitions combinations that this execution mode is valid for.
   For a detailed description of this property, you may refer `here <#.environments[].target_systems>`__.



General Configuration
---------------------

.. js:attribute:: .general[].check_search_path

.. envvar:: RFM_CHECK_SEARCH_PATH

.. option:: -c PATH | --checkpath PATH

   :required: No
   :default: ``["${RFM_INSTALL_PREFIX}/checks/"]``

   A list of paths (files or directories) where ReFrame will look for regression test files.
   If the search path is set through the environment variable, it should be a colon separated list.
   If specified from command line, the search path is constructed by specifying multiple times the command line option.


.. js:attribute:: .general[].check_search_recursive

.. envvar:: RFM_CHECK_SEARCH_RECURSIVE

.. option:: -R | --recursive

   :required: No
   :default: ``true``

   Search directories in the `search path <#.general[].check_search_path>`__ recursively.



.. js:attribute:: .general[].colorize

.. envvar:: RFM_COLORIZE

.. option:: --nocolor

   :required: No
   :default: ``true``

   Use colors in output.
   The command-line option sets the configuration option to ``false``.


.. js:attribute:: .general[].ignore_check_conflicts

.. envvar:: RFM_IGNORE_CHECK_CONFLICTS

.. option:: --ignore-check-conflicts

   :required: No
   :default: ``false``

   Ignore test name conflicts when loading tests.


.. js:attribute:: .general[].keep_stage_files

.. envvar:: RFM_KEEP_STAGE_FILES

.. option:: --keep-stage-files

   :required: No
   :default: ``false``

   Keep stage files of tests even if they succeed.


.. js:attribute:: .general[].module_map_file

.. envvar:: RFM_MODULE_MAP_FILE

.. option:: --module-mappings

   :required: No
   :default: ``""``

   File containing `module mappings <running.html#manipulating-modules>`__.


.. js:attribute:: .general[].module_mappings

.. envvar:: RFM_MODULE_MAPPINGS

.. option:: -M MAPPING | --map-module MAPPING

   :required: No
   :default: ``[]``

   A list of `module mappings <running.html#manipulating-modules>`__.
   If specified through the environment variable, the mappings must be separated by commas.
   If specified from command line, multiple module mappings are defined by passing the command line option multiple times.


.. js:attribute:: .general[].non_default_craype

.. envvar:: RFM_NON_DEFAULT_CRAYPE

.. option:: --non-default-craype

   :required: No
   :default: ``false``

   Test a non-default Cray Programming Environment.
   This will emit some special instructions in the generated build and job scripts.
   For more details, you may refer `here <running.html#testing-non-default-cray-programming-environments>`__.


.. js:attribute:: .general[].purge_environment

.. envvar:: RFM_PURGE_ENVIRONMENT

.. option:: --purge-env

   :required: No
   :default: ``false``

   Purge any loaded environment modules before running any tests.


.. js:attribute:: .general[].save_log_files

.. envvar:: RFM_SAVE_LOG_FILES

.. option:: --save-log-files

   :required: No
   :default: ``false``

   Save any log files generated by ReFrame to its output directory


.. js:attribute:: .general[].target_systems

   :required: No
   :default: ``["*"]``

   A list of systems or system/partitions combinations that these general options are valid for.
   For a detailed description of this property, you may refer `here <#.environments[].target_systems>`__.


.. js:attribute:: .general[].timestamp_dirs

.. envvar:: RFM_TIMESTAMP_DIRS

.. option:: --timestamp [TIMEFMT]

   :required: No
   :default: ``""``

   Append a timestamp to ReFrame directory prefixes.
   Valid formats are those accepted by the `time.strftime() <https://docs.python.org/3.8/library/time.html#time.strftime>`__ function.
   If specified from the command line without any argument, ``"%FT%T"`` will be used as a time format.


.. js:attribute:: .general[].unload_modules

.. envvar:: RFM_UNLOAD_MODULES

.. option:: -u MOD | --unload-module MOD

   :required: No
   :default: ``[]``

   A list of environment modules to unload before executing any test.
   If specified using an the environment variable, a space separated list of modules is expected.
   If specified from the command line, multiple modules can be passed by passing the command line option multiple times.


.. js:attribute:: .general[].user_modules

.. envvar:: RFM_USER_MODULES

.. option:: -m MOD | --module MOD

   :required: No
   :default: ``[]``

   A list of environment modules to be loaded before executing any test.
   If specified using an the environment variable, a space separated list of modules is expected.
   If specified from the command line, multiple modules can be passed by passing the command line option multiple times.


.. js:attribute:: .general[].verbose

.. envvar:: RFM_VERBOSE

.. option:: -v | --verbose

   :required: No
   :default: 0

   Increase the verbosity level of the output.
   The higher the number, the more verbose the output will be.
   If specified from the command line, the command line option must be specified multiple times to increase the verbosity level more than once.


Additional Environment Variables
--------------------------------

Here is a list of environment variables that do not have a configuration option counterpart.


.. envvar:: RFM_CONFIG_FILE

.. option:: -C FILE | --config-file FILE

   The path to ReFrame's configuration file.


.. envvar:: RFM_SYSTEM

.. option:: --system NAME

   The name of the system, whose configuration will be loaded.
