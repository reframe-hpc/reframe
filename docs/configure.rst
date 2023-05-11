=================================
Configuring ReFrame for Your Site
=================================

ReFrame comes pre-configured with a minimal generic configuration that will allow you to run ReFrame on any system.
This will allow you to run simple local tests using the default compiler of the system.
Of course, ReFrame is much more powerful than that.
This section will guide you through configuring ReFrame for your site.

ReFrame's configuration can be either in JSON or in Python format and can be split into multiple files.
The Python format is useful in cases that you want to generate configuration parameters on-the-fly, since ReFrame will import that Python file and the load the resulting configuration.
In the following we will use a single Python-based configuration file also for historical reasons, since it was the only way to configure ReFrame in versions prior to 3.0.


.. versionchanged:: 4.0.0
   The configuration can now be split into multiple files.


Loading the configuration
-------------------------

ReFrame builds its final configuration gradually by combining multiple configuration files.
Each one can have different parts of the configuration, for example different systems, different environments, different general options or different logging handlers.
This technique allows users to avoid having a single huge configuration file.

The first configuration file loaded in this chain is always the generic builtin configuration located under ``${RFM_INSTALL_PREFIX}/reframe/core/settings.py``.
This contains everything that ReFrame needs to run on a generic system, as well as basic settings for logging, so subsequent configuration files may skip defining some configuration sections altogether, if they are not relevant.

ReFrame continues on looking for configuration files in the directories defined in :envvar:`RFM_CONFIG_PATH`.
For each directory, will look within it for a ``settings.py`` or ``settings.json`` file (in that order), and if it finds one, it will load it.

Finally, ReFrame processes the :option:`--config-file` option or the :envvar:`RFM_CONFIG_FILES` environment variable to load any specific configuration files passed from the command line.


Anatomy of the Configuration File
---------------------------------

The whole configuration of ReFrame is a single JSON object whose properties are responsible for configuring the basic aspects of the framework.
We'll refer to these top-level properties as *sections*.
These sections contain other objects which further define in detail the framework's behavior.
If you are using a Python file to configure ReFrame, this big JSON configuration object is stored in a special variable called ``site_configuration``.

We will explore the basic configuration of ReFrame by looking into the configuration file of the tutorials, which permits ReFrame to run on the Piz Daint supercomputer and a local computer.
For the complete listing and description of all configuration options, you should refer to the :doc:`config_reference`.

.. literalinclude:: ../tutorials/config/daint.py
   :start-at: site_configuration

There are three required sections that the final ReFrame configuration must have: ``systems``, ``environments`` and ``logging``, but in most cases you will define only the first two, as ReFrame's builtin configuration already defines a reasonable logging configuration. We will first cover these sections and then move on to the optional ones.

.. tip::

   These configuration sections may not all be defined in the same configuration file, but can reside in any configuration file that is being loaded.
   This is the case of the example configuration shown above, where the ``logging`` section is "missing" as it's defined in ReFrame's builtin configuration.

---------------------
Systems Configuration
---------------------

ReFrame allows you to configure multiple systems in the same configuration file.
Each system is a different object inside the ``systems`` section.
In our example we define only Piz Daint:

.. literalinclude:: ../tutorials/config/daint.py
   :start-at: 'systems'
   :end-before: 'environments'

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

.. literalinclude:: ../tutorials/config/daint.py
   :start-at: 'name': 'gpu'
   :end-at: 'max_jobs'

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
* ``max_jobs``: The maximum number of concurrent regression tests that may be active (i.e., not completed) on this partition.
  This option is relevant only when ReFrame executes with the `asynchronous execution policy <pipeline.html#execution-policies>`__.

  For more partition configuration options, have a look `here <config_reference.html#system-partition-configuration>`__.


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

.. literalinclude:: ../tutorials/config/daint.py
   :start-at: 'environments'
   :end-at: # end of environments

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
Let's see how logging is defined in the builtin configuration:

.. literalinclude:: ../reframe/core/settings.py
   :start-at: 'logging'
   :end-at: # end of logging

Logging is configured under the ``logging`` section of the configuration, which is a list of logger objects.
Unless you want to configure logging differently for different systems, a single logger object is enough.
Each logger object is associated with a `logging level <config_reference.html#.logging[].level>`__ stored in the ``level`` property and has a set of logging handlers that are actually responsible for handling the actual logging records.
ReFrame's output is performed through its logging mechanism and that's why there is the special ``handlers$`` property.
The handler defined in this property, in the builtin configuration shown here, defines how exactly the output of ReFrame will be printed.
You will not have to override this in your configuration files, unless you really need to change how ReFrame's output look like.

As a user you might need to override the ``handlers`` property to define different sinks for ReFrame logs and/or output using different verbosity levels.
Note that you can use multiple handlers at the same time.
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

There is finally one additional optional configuration section that is not discussed here:

The ``modes`` section defines different execution modes for the framework.
Execution modes are discussed in the :doc:`pipeline` page.


.. _building-the-final-config:

Building the Final Configuration
--------------------------------

.. versionadded:: 4.0.0

As mentioned above ReFrame can build its final configuration incrementally from a series of user-specified configuration files starting from the basic builtin configuration.
We discussed briefly at the beginning of this page how ReFrame locates and loads these configuration files and the documentation of the :option:`-C` option provides more detailed information.
But how are these configuration files actually combined?
This is what we will discuss in this section.

Configuration objects in the top-level configuration sections can be split in two categories: *named* and *unnamed*.
Named objects are the systems, the environments and the modes and the rest are unnamed.
The named object have a ``name`` property.
When ReFrame builds its final configuration, named objects from newer configuration files are either appended or prepended in their respective sections, but unnamed objects are merged based on their ``target_systems``.
More specifically, new systems are *prepended* in the list of the already defined, whereas environments and modes are *appended*.
The reason for that is that systems are tried from the beginning of the list until a match is found.
See :ref:`pick-system-config` for more information on how ReFrame picks the right system.
If a system is redefined, ReFrame will warn about it, but it will still use the new definition.
This is done for backward compatibility with the old configuration mechanism, where users had to redefine also the builtin systems and environments in their configuration.
Similarly, if an environment or a mode is redefined, ReFrame will issue a warning, but only if the redefinition is at the same scope as the conflicting one.
Again this is done for backward compatibility.

Given the Piz Daint configuration shown in this section and the ReFrame's builtin configuration, ReFrame will build internally the following configuration:

.. code-block:: python

   site_configuration = {
       'systems': [
           {
               # from the Daint config
               'name': 'daint',
               ...
           },
           {
               # from the builtin config
               'name': 'generic',
               ...
           }
       ],
       'environments': [
           {
               # from the builtin config
               'name': 'builtin'
                ...
           },
           {
               # from the Daint config
               'name': 'gnu',
               ...
           }
       ],
       'logging': [
           # from the builtin config
       ]
   }

You might wonder why would I need to define multiple objects in sections such as ``logging`` or ``general``.
As mentioned above, ReFrame merges them if they refer to the same target systems, but if they don't they can serve as scopes for the configuration parameters they define.
Imagine the following ``general`` section:

.. code-block:: python

   'general': [
       {
           'git_timeout': 5
       },
       {
           'git_timeout': 10,
           'target_systems': ['daint']
       },
       {
           'git_timeout': 20,
           'target_systems': ['tresa']
       }
   ]

This means that the default value for ``git_timeout`` is 5 seconds for any system, but it is 10 for ``daint`` and 20 for ``tresa``.
The nice thing is that you can spread that in multiple configuration files and ReFrame will combine them internally in a single one with the various configuration options indexed by their scope.


.. _pick-system-config:

Picking the Right System Configuration
--------------------------------------

As discussed previously, ReFrame's configuration file can store the configurations for multiple systems.
When launched, ReFrame will pick the first matching configuration and load it.

ReFrame uses an auto-detection mechanism to get information about the host it is running on and uses that information to pick the right system configuration.
The default auto-detection method is using the ``hostname`` command, but you can define more methods using either the :attr:`~config.autodetect_methods` configuration parameter or the :envvar:`RFM_AUTODETECT_METHODS` environment variable.
After having retrieved the hostname, ReFrame goes through all the systems in its configuration and tries to match it against the :attr:`~config.systems.hostnames` patterns defined for every system.
The first system whose :attr:`~config.systems.hostnames` match will become the current system and its configuration will be loaded.

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
