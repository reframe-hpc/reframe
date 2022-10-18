=================================
Configuring ReFrame for Your Site
=================================

ReFrame comes pre-configured with a minimal generic configuration that will allow you to run ReFrame on any system.
This will allow you to run simple local tests using the default compiler of the system.
Of course, ReFrame is much more powerful than that.
This section will guide you through configuring ReFrame for your site.

If you started using ReFrame from version 3.0, you can keep on reading this section, otherwise you are advised to have a look first at the :doc:`migration_2_to_3` page.


ReFrame's configuration file can be either a JSON file or a Python file storing the site configuration in a JSON-formatted string.
The latter format is useful in cases that you want to generate configuration parameters on-the-fly, since ReFrame will import that Python file and the load the resulting configuration.
In the following we will use a Python-based configuration file also for historical reasons, since it was the only way to configure ReFrame in versions prior to 3.0.


Locating the Configuration File
-------------------------------

ReFrame looks for a configuration file in the following locations in that order:

1. ``${HOME}/.reframe/settings.{py,json}``
2. ``${RFM_INSTALL_PREFIX}/settings.{py,json}``
3. ``/etc/reframe.d/settings.{py,json}``

If both ``settings.py`` and ``settings.json`` are found, the Python file is preferred.
The ``RFM_INSTALL_PREFIX`` variable refers to the installation directory of ReFrame or the top-level source directory if you are running ReFrame from source.
Users have no control over this variable.
It is always set by the framework upon startup.

If no configuration file is found in any of the predefined locations, ReFrame will fall back to a generic configuration that allows it to run on any system.
You can find this generic configuration file `here <https://github.com/reframe-hpc/reframe/blob/master/reframe/core/settings.py>`__.
Users may *not* modify this file.

There are two ways to provide a custom configuration file to ReFrame:

1. Pass it through the ``-C`` or ``--config-file`` option.
2. Specify it using the ``RFM_CONFIG_FILE`` environment variable.

Command line options take always precedence over their respective environment variables.


Anatomy of the Configuration File
---------------------------------

The whole configuration of ReFrame is a single JSON object whose properties are responsible for configuring the basic aspects of the framework.
We'll refer to these top-level properties as *sections*.
These sections contain other objects which further define in detail the framework's behavior.
If you are using a Python file to configure ReFrame, this big JSON configuration object is stored in a special variable called ``site_configuration``.

We will explore the basic configuration of ReFrame by looking into the configuration file of the tutorials, which permits ReFrame to run both on the Piz Daint supercomputer and a local computer.
For the complete listing and description of all configuration options, you should refer to the :doc:`config_reference`.

.. literalinclude:: ../tutorials/config/settings.py
   :start-after: # rfmdocstart: site-configuration
   :end-before: # rfmdocend: site-configuration

There are three required sections that each configuration file must provide: ``systems``, ``environments`` and ``logging``.
We will first cover these and then move on to the optional ones.


---------------------
Systems Configuration
---------------------

ReFrame allows you to configure multiple systems in the same configuration file.
Each system is a different object inside the ``systems`` section.
In our example we define three systems, a Mac laptop, Piz Daint and a generic fallback system:

.. literalinclude:: ../tutorials/config/settings.py
   :start-after: # rfmdocstart: systems
   :end-before: # rfmdocend: systems

Each system is associated with a set of properties, which in this case are the following:

* ``name``: The name of the system.
  This should be an alphanumeric string (dashes ``-`` are allowed) and it will be used to refer to this system in other contexts.
* ``descr``: A detailed description of the system.
* ``hostnames``: This is a list of hostname patterns following the `Python Regular Expression Syntax <https://docs.python.org/3/library/re.html#regular-expression-syntax>`__, which will be used by ReFrame when it tries to automatically select a configuration entry for the current system.
* ``modules_system``: This refers to the modules management backend which should be used for loading environment modules on this system.
  Multiple backends are supported, as well as the special ``nomod`` backend which implements the different modules system operations as no-ops.
  For the complete list of the supported modules systems, see `here <config_reference.html#.systems[].modules_system>`__.
* ``partitions``: The list of partitions that are defined for this system.
  Each partition is defined as a separate object.
  We devote the rest of this section in system partitions, since they are an essential part of ReFrame's configuration.

A system partition in ReFrame is not bound to a real scheduler partition.
It is a virtual partition or separation of the system.
In the example shown here, we define three partitions that none of them corresponds to a scheduler partition.
The ``login`` partition refers to the login nodes of the system, whereas the ``gpu`` and ``mc`` partitions refer to two different set of nodes in the same cluster that are effectively separated using Slurm constraints.
Let's pick the ``gpu`` partition and look into it in more detail:

.. literalinclude:: ../tutorials/config/settings.py
   :start-after: # rfmdocstart: gpu-partition
   :end-before: # rfmdocend: gpu-partition

The basic properties of a partition are the following:

* ``name``: The name of the partition.
  This should be an alphanumeric string (dashes ``-`` are allowed) and it will be used to refer to this partition in other contexts.
* ``descr``: A detailed description of the system partition.
* ``scheduler``: The workload manager (job scheduler) used in this partition for launching parallel jobs.
  In this particular example, the `Slurm <https://slurm.schedmd.com/>`__ scheduler is used.
  For a complete list of the supported job schedulers, see `here <config_reference.html#.systems[].partitions[].scheduler>`__.
* ``launcher``: The parallel job launcher used in this partition.
  In this case, the ``srun`` command will be used.
  For a complete list of the supported parallel job launchers, see `here <config_reference.html#.systems[].partitions[].launcher>`__.
* ``access``: A list of scheduler options that will be passed to the generated job script for gaining access to that logical partition.
  Notice how in this case, the nodes are selected through a constraint and not an actual scheduler partition.
* ``environs``: The list of environments that ReFrame will use to run regression tests on this partition.
  These are just symbolic names that refer to environments defined in the ``environments`` section described below.
* ``container_platforms``: A set of supported container platforms in this partition.
  Each container platform is an object with a name and list of environment modules to load, in order to enable this platform.
  For a complete list of the supported container platforms, see `here <config_reference.html#.systems[].partitions[].container_platforms[].type>`__.
* ``max_jobs``: The maximum number of concurrent regression tests that may be active (i.e., not completed) on this partition.
  This option is relevant only when ReFrame executes with the `asynchronous execution policy <pipeline.html#execution-policies>`__.
* ``resources``: This is a set of optional additional scheduler resources that the tests can access transparently.
  For more information, please have a look `here <config_reference.html#custom-job-scheduler-resources>`__.


--------------------------
Environments Configuration
--------------------------

We have seen already environments to be referred to by the ``environs`` property of a partition.
An environment in ReFrame is simply a collection of environment modules, environment variables and compiler and compiler flags definitions.
None of these attributes is required.
An environment can simply by empty, in which case it refers to the actual environment that ReFrame runs in.
In fact, this is what the generic fallback configuration of ReFrame does.

Environments in ReFrame are configured under the ``environments`` section of the documentation.
For each environment referenced inside a partition, a definition of it must be present in this section.
In our example, we define environments for all the basic compilers as well as a default built-in one, which is used with the generic system configuration.
In certain contexts, it is useful to see a ReFrame environment as a wrapper of a programming toolchain (MPI + compiler combination):

.. literalinclude:: ../tutorials/config/settings.py
   :start-after: # rfmdocstart: environments
   :end-before: # rfmdocend: environments

Each environment is associated with a name.
This name will be used to reference this environment in different contexts, as for example in the ``environs`` property of the system partitions.
A programming environment in ReFrame is essentially a collection of environment modules, environment variables and compiler definitions.

An important feature in ReFrame's configuration, is that you can define section objects differently for different systems or system partitions by using the ``target_systems`` property.
Notice, for example, how the ``gnu`` environment is defined differently for the system ``daint`` compared to the generic definition.
The ``target_systems`` property is a list of systems or system/partition combinations where this definition of the environment is in effect.
This means that ``gnu`` will be defined this way only for regression tests running on ``daint``.
For all the other systems, it will be defined using the first definition.


---------------------
Logging configuration
---------------------

ReFrame has a powerful logging mechanism that gives fine grained control over what information is being logged, where it is being logged and how this information is formatted.
Additionally, it allows for logging performance data from performance tests into different channels.
Let's see how logging is defined in our example configuration, which also represents a typical one for logging:

.. literalinclude:: ../tutorials/config/settings.py
   :start-after: # rfmdocstart: logging
   :end-before: # rfmdocend: logging

Logging is configured under the ``logging`` section of the configuration, which is a list of logger objects.
Unless you want to configure logging differently for different systems, a single logger object is enough.
Each logger object is associated with a `logging level <config_reference.html#.logging[].level>`__ stored in the ``level`` property and has a set of logging handlers that are actually responsible for handling the actual logging records.
ReFrame's output is performed through the logging mechanism, meaning that if you don't specify any logging handler, you will not get any output from ReFrame!
The ``handlers`` property of the logger object holds the actual handlers.
Notice that you can use multiple handlers at the same time, which enables you to feed ReFrame's output to different sinks and at different verbosity levels.
All handler objects share a set of common properties.
These are the following:

* ``type``: This is the type of the handler, which determines its functionality.
  Depending on the handler type, handler-specific properties may be allowed or required.
  For a complete list of available log handler types, see `here <config_reference.html#.logging[].handlers[].type>`__.
* ``level``: The cut-off level for messages reaching this handler.
  Any message with a lower level number will be filtered out.
* ``format``: A format string for formatting the emitted log record.
  ReFrame uses the format specifiers from `Python Logging <https://docs.python.org/3/library/logging.html?highlight=logging#logrecord-attributes>`__, but also defines its owns specifiers.
* ``datefmt``: A time format string for formatting timestamps.
  There are two log record fields that are considered timestamps: (a) ``asctime`` and (b) ``check_job_completion_time``.
  ReFrame follows the time formatting syntax of Python's `time.strftime() <https://docs.python.org/3/library/time.html#time.strftime>`__ with a small tweak allowing full RFC3339 compliance when formatting time zone differences.

We will not go into the details of the individual handlers here.
In this particular example we use three handlers of two distinct types:

1. A file handler to print debug messages in the ``reframe.log`` file using a more extensive message format that contains a timestamp, the level name etc.
2. A stream handler to  print any informational messages (and warnings and errors) from ReFrame to the standard output.
   This handles essentially the actual output of ReFrame.
3. A file handler to print the framework's output in the ``reframe.out`` file.

It might initially seem confusing the fact that there are two ``level`` properties: one at the logger level and one at the handler level.
Logging in ReFrame works hierarchically.
When a message is logged, a log record is created, which contains metadata about the message being logged (log level, timestamp, ReFrame runtime information etc.).
This log record first goes into ReFrame's internal logger, where the record's level is checked against the logger's level (here  ``debug``).
If the log record's level exceeds the log level threshold from the logger, it is forwarded to the logger's handlers.
Then each handler filters the log record differently and takes care of formatting the log record's message appropriately.
You can view logger's log level as a general cut off.
For example, if we have set it to ``warning``, no debug or informational messages would ever be printed.

Finally, there is a special set of handlers for handling performance log messages.
Performance log messages are generated *only* for `performance tests <tutorial_basics.html#writing-a-performance-test>`__, i.e., tests defining the :attr:`~reframe.core.pipeline.RegressionTest.perf_variables` or the :attr:`~reframe.core.pipeline.RegressionTest.perf_patterns` attributes.
The performance log handlers are stored in the ``handlers_perflog`` property.
The ``filelog`` handler used in this example will create a file per test and per system/partition combination (``./<system>/<partition>/<testname>.log``) and will append to it the obtained performance data every time a performance test is run.
Notice how the message to be logged is structured in the ``format`` and ``format_perfvars`` properties, such that it can be easily parsed from post processing tools.
Apart from file logging, ReFrame offers more advanced performance logging capabilities through Syslog, Graylog and HTTP.

For a complete reference of logging configuration parameters, please refer to the :doc:`config_reference`.


-----------------------------
General configuration options
-----------------------------

General configuration options of the framework go under the ``general`` section of the configuration file.
This section is optional and, in fact, we do not define it for our tutorial configuration file.
However, there are several options that can go into this section, but the reader is referred to the :doc:`config_reference` for the complete list.


---------------------------
Other configuration options
---------------------------

There are finally two more optional configuration sections that are not discussed here:

1. The ``schedulers`` section holds configuration variables specific to the different scheduler backends and
2. the ``modes`` section defines different execution modes for the framework.
   Execution modes are discussed in the :doc:`pipeline` page.



Picking a System Configuration
------------------------------

As discussed previously, ReFrame's configuration file can store the configurations for multiple systems.
When launched, ReFrame will pick the first matching configuration and load it.

ReFrame uses an auto-detection mechanism to get information about the host it is running on and uses that information to pick the right system configuration.
Currently, only one auto-detection method is supported that retrieves the hostname.
Based on this, ReFrame goes through all the systems in its configuration and tries to match the hostname against any of the patterns defined in each system's ``hostnames`` property.
The detection process stops at the first match found, and that system's configuration is selected.

The auto-detection process can be controlled through the :envvar:`RFM_AUTODETECT_METHOD`, :envvar:`RFM_AUTODETECT_FQDN` and :envvar:`RFM_AUTODETECT_XTHOSTNAME` environment variables.


As soon as a system configuration is selected, all configuration objects that have a ``target_systems`` property are resolved against the selected system, and any configuration object that is not applicable is dropped.
So, internally, ReFrame keeps an *instantiation* of the site configuration for the selected system only.
To better understand this, let's assume that we have the following ``environments`` defined:

.. code:: python

    'environments': [
        {
            'name': 'cray',
            'modules': ['cray']
        },
        {
            'name': 'gnu',
            'modules': ['gnu']
        },
        {
            'name': 'gnu',
            'modules': ['gnu', 'openmpi'],
            'cc':  'mpicc',
            'cxx': 'mpicxx',
            'ftn': 'mpif90',
            'target_systems': ['foo']
        }
    ],


If the selected system is ``foo``, then ReFrame will use the second definition of ``gnu`` which is specific to the ``foo`` system.

You can override completely the system auto-selection process by specifying a system or system/partition combination with the ``--system`` option, e.g., ``--system=daint`` or ``--system=daint:gpu``.


Querying Configuration Options
------------------------------

ReFrame offers the powerful ``--show-config`` command-line option that allows you to query any configuration parameter of the framework and see how it is set for the selected system.
Using no arguments or passing ``all`` to this option, the whole configuration for the currently selected system will be printed in JSON format, which you can then pipe to a JSON command line editor, such as `jq <https://stedolan.github.io/jq/>`__, and either get a colored output or even generate a completely new ReFrame configuration!

Passing specific configuration keys in this option, you can query specific parts of the configuration.
Let's see some concrete examples:

* Query the current system's partitions:

  .. code-block:: console

     ./bin/reframe -C tutorials/config/settings.py --system=daint --show-config=systems/0/partitions

  .. code:: javascript

     [
       {
         "name": "login",
         "descr": "Login nodes",
         "scheduler": "local",
         "launcher": "local",
         "environs": [
           "gnu",
           "intel",
           "nvidia",
           "cray"
         ],
         "max_jobs": 10
       },
       {
         "name": "gpu",
         "descr": "Hybrid nodes",
         "scheduler": "slurm",
         "launcher": "srun",
         "access": [
           "-C gpu",
           "-A csstaff"
         ],
         "environs": [
           "gnu",
           "intel",
           "nvidia",
           "cray"
         ],
         "max_jobs": 100
       },
       {
         "name": "mc",
         "descr": "Multicore nodes",
         "scheduler": "slurm",
         "launcher": "srun",
         "access": [
           "-C mc",
           "-A csstaff"
         ],
         "environs": [
           "gnu",
           "intel",
           "nvidia",
           "cray"
         ],
         "max_jobs": 100
       }
     ]

  Check how the output changes if we explicitly set system to ``daint:login``:

  .. code-block:: console

     ./bin/reframe -C tutorials/config/settings.py --system=daint:login --show-config=systems/0/partitions


  .. code:: javascript

     [
       {
         "name": "login",
         "descr": "Login nodes",
         "scheduler": "local",
         "launcher": "local",
         "environs": [
           "gnu",
           "intel",
           "nvidia",
           "cray"
         ],
         "max_jobs": 10
       }
     ]


  ReFrame will internally represent system ``daint`` as having a single partition only.
  Notice also how you can use indexes to objects elements inside a list.

* Query an environment configuration:

  .. code-block:: console

     ./bin/reframe -C tutorials/config/settings.py --system=daint --show-config=environments/@gnu

  .. code:: javascript

     {
       "name": "gnu",
       "modules": [
         "PrgEnv-gnu"
       ],
       "cc": "cc",
       "cxx": "CC",
       "ftn": "ftn",
       "target_systems": [
         "daint"
       ]
     }

  If an object has a ``name`` property you can address it by name using the ``@name`` syntax, instead of its index.

* Query an environment's compiler:

  .. code-block:: console

     ./bin/reframe -C tutorials/config/settings.py --system=daint --show-config=environments/@gnu/cxx

  .. code:: javascript

     "CC"

  If you explicitly query a configuration value which is not defined in the configuration file, ReFrame will print its default value.


.. _proc-autodetection:

Auto-detecting processor information
------------------------------------

.. versionadded:: 3.7.0

.. |devices| replace:: :attr:`devices`
.. _devices: config_reference.html#.systems[].partitions[].devices
.. |processor| replace:: :attr:`processor`
.. _processor: config_reference.html#.systems[].partitions[].processor
.. |detect_remote_system_topology| replace:: :attr:`remote_detect`
.. _detect_remote_system_topology: config_reference.html#.general[].remote_detect

ReFrame is able to detect the processor topology of both local and remote partitions automatically.
The processor and device information are made available to the tests through the corresponding attributes of the :attr:`~reframe.core.pipeline.RegressionTest.current_partition` allowing a test to modify its behavior accordingly.
Currently, ReFrame supports auto-detection of the local or remote processor information only.
It does not support auto-detection of devices, in which cases users should explicitly specify this information using the |devices|_ configuration option.
The processor information auto-detection works as follows:

#. If the |processor|_ configuration option is defined, then no auto-detection is attempted.

#. If the |processor|_ configuration option is not defined, ReFrame will look for a processor configuration metadata file in ``~/.reframe/topology/{system}-{part}/processor.json``.
   If the file is found, the topology information is loaded from there.
   These files are generated automatically by ReFrame from previous runs.

#. If the corresponding metadata files are not found, the processor information will be auto-detected.
   If the system partition is local (i.e., ``local`` scheduler + ``local`` launcher), the processor information is auto-detected unconditionally and stored in the corresponding metadata file for this partition.
   If the partition is remote, ReFrame will not try to auto-detect it unless the :envvar:`RFM_REMOTE_DETECT` or the |detect_remote_system_topology|_ configuration option is set.
   In that case, the steps to auto-detect the remote processor information are the following:

     a. ReFrame creates a fresh clone of itself in a temporary directory created under ``.`` by default.
        This temporary directory prefix can be changed by setting the :envvar:`RFM_REMOTE_WORKDIR` environment variable.
     b. ReFrame changes to that directory and launches a job that will first bootstrap the fresh clone and then run that clone with ``{launcher} ./bin/reframe --detect-host-topology=topo.json``.
        The :option:`--detect-host-topology` option causes ReFrame to detect the topology of the current host,
        which in this case would be the remote compute nodes.

   In case of errors during auto-detection, ReFrame will simply issue a warning and continue.
