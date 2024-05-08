***********************
Configuration Reference
***********************

ReFrame's behavior can be configured through its configuration file, environment variables and command-line options.
An option can be specified via multiple paths (e.g., a configuration file parameter and an environment variable), in which case command-line options precede environment variables, which in turn precede configuration file options.
This section provides a complete reference guide of the configuration options of ReFrame that can be set in its configuration file or specified using environment variables.

ReFrame's configuration is in JSON syntax.
The full schema describing it can be found in |schemas/config.json|_ file.
The final configuration for ReFrame is validated against this schema.

The syntax we use to describe the different configuration objects follows the convention: ``OBJECT[.OBJECT]*.PROPERTY``.
Even if a configuration object contains a list of other objects, this is not reflected in the above syntax, as all objects in a certain list are homogeneous.
For example, by ``systems.partitions.name`` we designate the ``name`` property of any partition object inside the ``partitions`` property of any system object inside the top level ``systems`` object.
If we were to use indices, that would be rewritten as ``systems[i].partitions[j].name`` where ``i`` indexes the systems and ``j`` indexes the partitions of the i-th system.
For cases, where the objects in a list are not homogeneous, e.g., the logging handlers, we surround the object type with ``..``.
For example, the ``logging.handlers_perflog..filelog..name`` syntax designates the ``name`` attribute of the ``filelog`` logging handler.

.. |schemas/config.json| replace:: ``reframe/schemas/config.json``
.. _schemas/config.json: https://github.com/reframe-hpc/reframe/blob/master/reframe/schemas/config.json


Top-level Configuration
=======================

The top-level configuration object is essentially the full configuration of ReFrame.
It consists of the following properties, which we also call conventionally *configuration sections*:

.. py:data:: systems

   :required: Yes

   A list of `system configuration objects <#system-configuration>`__.


.. py:data:: environments

   :required: Yes

   A list of `environment configuration objects <#environment-configuration>`__.


.. py:data:: logging

   :required: Yes

   A list of `logging configuration objects <#logging-configuration>`__.


.. py:data:: modes

   :required: No

   A list of `execution mode configuration objects <#execution-mode-configuration>`__.

.. py:data:: general

   :required: No

   A list of `general configuration objects <#general-configuration>`__.


.. py:data:: autodetect_methods

   :required: No
   :default: ``["py::socket.gethostname"]``

   A list of system auto-detection methods for identifying the current system.

   The list can contain two types of methods:

   1. Python methods: These are prefixed with ``py::`` and should point to a Python callable taking zero arguments and returning a string.
      If the specified Python callable is not prefixed with a module, it will be looked up in the loaded configuration files starting from the last file.
      If the requested symbol cannot be found, a warning will be issued and the method will be ignored.
   2. Shell commands: Any string not prefixed with ``py::`` will be treated as a shell command and will be executed *during auto-detection* to retrieve the hostname.
      The standard output of the command will be used.

   If the :option:`--system` option is not passed, ReFrame will try to autodetect the current system trying the methods in this list successively, until one of them succeeds.
   The resulting name will be matched against the :attr:`~config.systems.hostnames` patterns of each system and the system that matches first will be used as the current one.

   The auto-detection methods can also be controlled through the :envvar:`RFM_AUTODETECT_METHODS` environment variable.

   .. versionadded:: 4.3


.. warning::
   .. versionchanged:: 4.0.0
      The :data:`schedulers` section is removed.
      Scheduler options should be set per partition using the :attr:`~config.systems.partitions.sched_options` attribute.



System Configuration
====================

.. currentmodule:: config

.. py:attribute:: systems.name

   :required: Yes

   The name of this system.
   Only alphanumeric characters, dashes (``-``) and underscores (``_``) are allowed.

.. py:attribute:: systems.descr

   :required: No
   :default: ``""``

   The description of this system.

.. py:attribute:: systems.hostnames

   :required: Yes

   A list of hostname regular expression patterns in Python `syntax <https://docs.python.org/3.8/library/re.html>`__, which will be used by the framework in order to automatically select a system configuration.
   For the auto-selection process, see `here <configure.html#picking-a-system-configuration>`__.

.. py:attribute:: systems.max_local_jobs

   The maximum number of forced local build or run jobs allowed.

   Forced local jobs run within the execution context of ReFrame.

   :required: No
   :default: ``8``

   .. versionadded:: 3.10.0


.. py:attribute:: systems.modules_system

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
   - ``spack``: `Spack <https://spack.readthedocs.io/en/latest/>`__'s built-in mechanism for managing modules.
   - ``nomod``: This is to denote that no modules system is used by this system.

   Normally,  upon loading the configuration of the system ReFrame checks that a sane installation exists for the modules system requested and will issue an error if it fails to find one.
   The modules system sanity check is skipped when the :attr:`~config.general.resolve_module_conflicts` is set to :obj:`False`.
   This is useful in cases where the current system does not have a modules system but the remote partitions have one and you would like ReFrame to generate the module commands.

  .. versionadded:: 3.4
      The ``spack`` backend is added.

  .. versionchanged:: 4.5.0
     The modules system sanity check is skipped when the :attr:`config.general.resolve_module_conflicts` is not set.


.. py:attribute:: systems.modules

   :required: No
   :default: ``[]``

   A list of `environment module objects <#module-objects>`__ to be loaded always when running on this system.
   These modules modify the ReFrame environment.
   This is useful in cases where a particular module is needed, for example, to submit jobs on a specific system.

.. py:attribute:: systems.env_vars

   :required: No
   :default: ``[]``

   A list of environment variables to be set always when running on this system.
   These variables modify the ReFrame environment.
   Each environment variable is specified as a two-element list containing the variable name and its value.
   You may reference other environment variables when defining an environment variable here.
   ReFrame will expand its value.
   Variables are set after the environment modules are loaded.

   .. versionadded:: 4.0.0

.. py:attribute:: systems.variables

   .. deprecated:: 4.0.0
      Please use :attr:`~config.systems.env_vars` instead.
      If specified in conjunction with :attr:`~config.systems.env_vars`, it will be ignored.

.. py:attribute:: systems.prefix

   :required: No
   :default: ``"."``

   Directory prefix for a ReFrame run on this system.
   Any directories or files produced by ReFrame will use this prefix, if not specified otherwise.

.. py:attribute:: systems.stagedir

   :required: No
   :default: ``"${RFM_PREFIX}/stage"``

   Stage directory prefix for this system.
   This is the directory prefix, where ReFrame will create the stage directories for each individual test case.


.. py:attribute:: systems.outputdir

   :required: No
   :default: ``"${RFM_PREFIX}/output"``

   Output directory prefix for this system.
   This is the directory prefix, where ReFrame will save information about the successful tests.


.. py:attribute:: systems.resourcesdir

   :required: No
   :default: ``"."``

   Directory prefix where external test resources (e.g., large input files) are stored.
   You may reference this prefix from within a regression test by accessing the :attr:`~reframe.core.systems.System.resourcesdir` attribute of the current system.


.. py:attribute:: systems.partitions

   :required: Yes

   A list of `system partition configuration objects <#system-partition-configuration>`__.
   This list must have at least one element.


.. py:attribute:: systems.sched_options

   :required: No
   :default: ``{}``

   Scheduler options for the local scheduler that is associated with the ReFrame's execution context.
   To understand the difference between the different execution contexts, please refer to ":ref:`execution-contexts`"
   For the available scheduler options, see the :attr:`~config.systems.partitions.sched_options` in the partition configuration below.

   .. versionadded:: 4.1

   .. warning::
      This option is broken in 4.0.


System Partition Configuration
==============================

.. py:attribute:: systems.partitions.name

   :required: Yes

   The name of this partition.
   Only alphanumeric characters, dashes (``-``) and underscores (``_``) are allowed.

.. py:attribute:: systems.partitions.descr

   :required: No
   :default: ``""``

   The description of this partition.

.. py:attribute:: systems.partitions.scheduler

   :required: Yes

   The job scheduler that will be used to launch jobs on this partition.
   Supported schedulers are the following:

   - ``flux``: Jobs will be launched using the `Flux Framework <https://flux-framework.org/>`_ scheduler.
   - ``local``: Jobs will be launched locally without using any job scheduler.
   - ``lsf``: Jobs will be launched using the `LSF <https://www.ibm.com/docs/en/spectrum-lsf/10.1.0?topic=lsf-session-scheduler>`__ scheduler.
   - ``oar``: Jobs will be launched using the `OAR <https://oar.imag.fr/>`__ scheduler.
   - ``pbs``: Jobs will be launched using the `PBS Pro <https://en.wikipedia.org/wiki/Portable_Batch_System>`__ scheduler.
   - ``sge``: Jobs will be launched using the `Sun Grid Engine <https://arc.liv.ac.uk/SGE/htmlman/manuals.html>`__ scheduler.
   - ``slurm``: Jobs will be launched using the `Slurm <https://www.schedmd.com/>`__ scheduler.
     This backend requires job accounting to be enabled in the target system.
     If not, you should consider using the ``squeue`` backend below.
   - ``squeue``: Jobs will be launched using the `Slurm <https://www.schedmd.com/>`__ scheduler.
     This backend does not rely on job accounting to retrieve job statuses, but ReFrame does its best to query the job state as reliably as possible.
   - ``ssh``: Jobs will be launched on a remote host using SSH.

     The remote host will be selected from the list of hosts specified in :attr:`~systems.partitions.sched_options.ssh_hosts`.
     The scheduler keeps track of the hosts that it has submitted jobs to, and it will select the next available one in a round-robin fashion.
     For connecting to a remote host, the options specified in :attr:`~systems.partitions.access` will be used.

     When a job is submitted with this scheduler, its stage directory will be copied over to a unique temporary directory on the remote host, then the job will be executed and, finally, any produced artifacts will be copied back.

     The contents of the stage directory are copied to the remote host either using ``rsync``, if available, or ``scp`` as a second choice.
     The same :attr:`~systems.partitions.access` options will be used in those operations as well.
     Please note, that the connection options of ``ssh`` and ``scp`` differ and ReFrame will not attempt to translate any options between the two utilities in case ``scp`` is selected for copying to the remote host.
     In this case, it is preferable to set up the host connection options in ``~/.ssh/config`` and leave :attr:`~systems.partition.access` blank.

     Job-scheduler command line options can be used to interact with the ``ssh`` backend.
     More specifically, if the :option:`--distribute` option is used, a test will be generated for each host listed in :attr:`~systems.partitions.sched_options.ssh_hosts`.
     You can also pin a test to a specific host if you pass the ``#host`` directive to the :option:`-J` option, e.g., ``-J '#host=myhost'``.

   - ``torque``: Jobs will be launched using the `Torque <https://en.wikipedia.org/wiki/TORQUE>`__ scheduler.

   .. versionadded:: 3.7.2
      Support for the SGE scheduler is added.

   .. versionadded:: 3.8.2
      Support for the OAR scheduler is added.

   .. versionadded:: 3.11.0
      Support for the LSF scheduler is added.

   .. versionadded:: 4.4
      The ``ssh`` scheduler is added.

   .. note::

      The way that multiple node jobs are submitted using the SGE scheduler can be very site-specific.
      For this reason, the ``sge`` scheduler backend does not try to interpret any related arguments, e.g., ``num_tasks``, ``num_tasks_per_node`` etc.
      Users must specify how these resources are to be requested by setting the :attr:`~config.systems.partitions.resources` partition configuration parameter and then request them from inside a test using the :py:attr:`~reframe.core.pipeline.RegressionTest.extra_resources` test attribute.
      Here is an example configuration for a system partition named ``foo`` that defines different ways for submitting MPI-only, OpenMP-only and MPI+OpenMP jobs:

      .. code-block:: python

         {
             'name': 'foo',
             'scheduler': 'sge',
             'resources': [
                 {
                     'name': 'smp',
                     'options': ['-pe smp {num_slots}']
                 },
                 {
                     'name': 'mpi',
                     'options': ['-pe mpi {num_slots}']
                 },
                 {
                     'name': 'mpismp',
                     'options': ['-pe mpismp {num_slots}']
                 }
             ]
         }

      Each test then can request the different type of slots as follows:

      .. code-block:: python

         self.extra_resouces = {
             'smp': {'num_slots': self.num_cpus_per_task},
             'mpi': {'num_slots': self.num_tasks},
             'mpismp': {'num_slots': self.num_tasks*self.num_cpus_per_task}
         }

      Notice that defining :attr:`~reframe.core.pipeline.RegressionTest.extra_resources` allows the test to be portable to other systems that have different schedulers;
      the :attr:`~reframe.core.pipeline.RegressionTest.extra_resources` will be simply ignored in this case and the scheduler backend will interpret the different test fields in the appropriate way.


.. py:attribute:: systems.partitions.sched_options

   :required: No
   :default: ``{}``

   Scheduler-specific options for this partition.
   See below for the available options.

   .. versionadded:: 4.1

   .. warning::
      This option is broken in 4.0.

.. py:attribute:: systems.partitions.sched_options.ssh_hosts

   :required: No
   :default: ``[]``

   List of hosts in a partition that uses the ``ssh`` scheduler.


.. py:attribute:: systems.partitions.sched_options.ignore_reqnodenotavail

   :required: No
   :default: ``false``

   Ignore the ``ReqNodeNotAvail`` Slurm state.

   If a job associated to a test is in pending state with the Slurm reason ``ReqNodeNotAvail`` and a list of unavailable nodes is also specified, ReFrame will check the status of the nodes and, if all of them are indeed down, it will cancel the job.
   Sometimes, however, when Slurm's backfill algorithm takes too long to compute, Slurm will set the pending reason to ``ReqNodeNotAvail`` and mark all system nodes as unavailable, causing ReFrame to kill the job.
   In such cases, you may set this parameter to ``true`` to avoid this.

   This option is relevant for the Slurm backends only.

.. py:attribute:: systems.partitions.sched_options.job_submit_timeout

   :required: No
   :default: ``60``

   Timeout in seconds for the job submission command.

   If timeout is reached, the test issuing that command will be marked as a failure.


.. py:attribute:: systems.partitions.sched_options.resubmit_on_errors

   :required: No
   :default: ``[]``

   If any of the listed errors occur, try to resubmit the job after some seconds.

   As an example, you could have ReFrame trying to resubmit a job in case that the maximum submission limit per user is reached by setting this field to ``["QOSMaxSubmitJobPerUserLimit"]``.
   You can ignore multiple errors at the same time if you add more error strings in the list.

   This option is relevant for the Slurm backends only.

   .. versionadded:: 3.4.1

   .. warning::
      Job submission is a synchronous operation in ReFrame.
      If this option is set, ReFrame's execution will block until the error conditions specified in this list are resolved.
      No other test would be able to proceed.


.. py:attribute:: systems.partitions.sched_options.use_nodes_option

   :required: No
   :default: ``false``

   Always emit the ``--nodes`` Slurm option in the preamble of the job script.

   This option is relevant for the Slurm backends only.


.. py:attribute:: systems.partitions.launcher

   :required: Yes

   The parallel job launcher that will be used in this partition to launch parallel programs.
   Available values are the following:

   - ``alps``: Parallel programs will be launched using the `Cray ALPS <https://pubs.cray.com/content/S-2393/CLE%205.2.UP03/cle-xc-system-administration-guide-s-2393-5203-xc/the-aprun-client>`__ ``aprun`` command.
   - ``clush``: Parallel programs will be launched using the `ClusterShell <http://clustershell.readthedocs.org/>`__ ``clush`` command. This launcher uses the partition's :attr:`~config.systems.partitions.access` property in order to determine the options to be passed to ``clush``.
   - ``ibrun``: Parallel programs will be launched using the ``ibrun`` command.
     This is a custom parallel program launcher used at `TACC <https://portal.tacc.utexas.edu/user-guides/stampede2>`__.
   - ``local``: No parallel program launcher will be used.
     The program will be launched locally.
   - ``lrun``: Parallel programs will be launched using `LC Launcher  <https://hpc.llnl.gov/training/tutorials/using-lcs-sierra-system#lrun>`__'s ``lrun`` command.
   - ``lrun-gpu``: Parallel programs will be launched using `LC Launcher <https://hpc.llnl.gov/training/tutorials/using-lcs-sierra-system#lrun>`__'s ``lrun -M "-gpu"`` command that enables the CUDA-aware Spectrum MPI.
   - ``mpirun``: Parallel programs will be launched using the ``mpirun`` command.
   - ``mpiexec``: Parallel programs will be launched using the ``mpiexec`` command.
   - ``pdsh``: Parallel programs will be launched using the ``pdsh`` command. This launcher uses the partition's :attr:`~config.systems.partitions.access` property in order to determine the options to be passed to ``pdsh``.
   - ``srun``: Parallel programs will be launched using `Slurm <https://slurm.schedmd.com/srun.html>`__'s ``srun`` command.
   - ``srunalloc``: Parallel programs will be launched using `Slurm <https://slurm.schedmd.com/srun.html>`__'s ``srun`` command, but job allocation options will also be emitted.
     This can be useful when combined with the ``local`` job scheduler.
   - ``ssh``: Parallel programs will be launched using SSH.
     This launcher uses the partition's :attr:`~config.systems.partitions.access` property in order to determine the remote host and any additional options to be passed to the SSH client.
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

   - ``upcrun``: Parallel programs will be launched using the `UPC <https://upc.lbl.gov/>`__ ``upcrun`` command.
   - ``upcxx-run``: Parallel programs will be launched using the `UPC++ <https://bitbucket.org/berkeleylab/upcxx/wiki/Home>`__ ``upcxx-run`` command.

   .. tip::

      .. versionadded:: 4.0.0

        ReFrame also allows you to register your own custom launchers simply by defining them in the configuration.
        You can follow a small tutorial `here <tutorial_advanced.html#adding-a-custom-launcher-to-a-partition>`__.


.. py:attribute:: systems.partitions.access

   :required: No
   :default: ``[]``

   A list of job scheduler options that will be passed to the generated job script for gaining access to that logical partition.

.. note::
   For the ``pbs`` and ``torque`` backends, options accepted in the :attr:`~config.systems.partitions.access` and :attr:`~config.systems.partitions.resources` parameters may either refer to actual ``qsub`` options or may just be resources specifications to be passed to the ``-l`` option.
   The backend assumes a ``qsub`` option, if the options passed in these attributes start with a ``-``.

.. note::
   If constraints are specified in :attr:`~config.systems.partition.access` for the Slurm backends,
   these will be AND'ed with any additional constraints passed either through the test job :attr:`~reframe.core.schedulers.Job.options` or the :option:`-J` command-line option.
   In other words, any constraint passed in :attr:`~config.systems.partition.access` will always be present in the generated job script.


.. py:attribute:: systems.partitions.environs

   :required: No
   :default: ``[]``

  A list of environment names that ReFrame will use to run regression tests on this partition.
  Each environment must be defined in the :data:`environments` section of the configuration and the definition of the environment must be valid for this partition.


.. py:attribute:: systems.partitions.container_platforms

   :required: No
   :default: ``[]``

   A list for `container platform configuration objects <#container-platform-configuration>`__.
   This will allow launching regression tests that use containers on this partition.


.. py:attribute:: systems.partitions.modules

   :required: No
   :default: ``[]``

  A list of `environment module objects <#module-objects>`__ to be loaded before running a regression test on this partition.


.. py:attribute:: systems.partitions.time_limit

   :required: No
   :default: ``null``

   The time limit for the jobs submitted on this partition.
   When the value is ``null``, no time limit is applied.


.. py:attribute:: systems.partitions.env_vars

   :required: No
   :default: ``[]``

   A list of environment variables to be set before running a regression test on this partition.
   Each environment variable is specified as a two-element list containing the variable name and its value.
   You may reference other environment variables when defining an environment variable here.
   ReFrame will expand its value.
   Variables are set after the environment modules are loaded.

   .. versionadded:: 4.0.0

.. py:attribute:: systems.partitions.variables

   .. deprecated:: 4.0.0
      Please use :attr:`~config.systems.partitions.env_vars` instead.
      If specified in conjunction with :attr:`~config.systems.partitions.env_vars`, it will be ignored.

.. py:attribute:: systems.partitions.max_jobs

   :required: No
   :default: ``8``

   The maximum number of concurrent regression tests that may be active (i.e., not completed) on this partition.
   This option is relevant only when ReFrame executes with the `asynchronous execution policy <pipeline.html#execution-policies>`__.


.. py:attribute:: systems.partitions.prepare_cmds

   :required: No
   :default: ``[]``

   List of shell commands to be emitted before any environment loading commands are emitted.

   .. versionadded:: 3.5.0


.. py:attribute:: systems.partitions.resources

   :required: No
   :default: ``[]``

   A list of job scheduler `resource specification <config_reference.html#custom-job-scheduler-resources>`__ objects.


.. py:attribute:: systems.partitions.processor

   :required: No
   :default: ``{}``

   Processor information for this partition stored in a `processor info object <#processor-info>`__.
   If not set, ReFrame will try to determine this information as follows:

   #. If the processor configuration metadata file in ``~/.reframe/topology/{system}-{part}/processor.json`` exists, the topology information is loaded from there.
      These files are generated automatically by ReFrame from previous runs.

   #. If the corresponding metadata files are not found, the processor information will be auto-detected.
      If the system partition is local (i.e., ``local`` scheduler + ``local`` launcher), the processor information is auto-detected unconditionally and stored in the corresponding metadata file for this partition.
      If the partition is remote, ReFrame will not try to auto-detect it unless the :envvar:`RFM_REMOTE_DETECT` or the :attr:`general.remote_detect` configuration option is set.
      In that case, the steps to auto-detect the remote processor information are the following:

        a. ReFrame creates a fresh clone of itself in a temporary directory created under ``.`` by default.
           This temporary directory prefix can be changed by setting the :envvar:`RFM_REMOTE_WORKDIR` environment variable.
        b. ReFrame changes to that directory and launches a job that will first bootstrap the fresh clone and then run that clone with ``{launcher} ./bin/reframe --detect-host-topology=topo.json``.
           The :option:`--detect-host-topology` option causes ReFrame to detect the topology of the current host,
           which in this case would be one of the remote compute nodes.

      In case of errors during auto-detection, ReFrame will simply issue a warning and continue.


   .. versionadded:: 3.5.0

   .. versionchanged:: 3.7.0
      ReFrame is now able to detect the processor information automatically.


.. py:attribute:: systems.partitions.devices

   :required: No
   :default: ``[]``

   A list with `device info objects <#device-info>`__ for this partition.

   .. versionadded:: 3.5.0


.. py:attribute:: systems.partitions.features

   :required: No
   :default: ``[]``

   User defined features of the partition.

   These are accessible through the :attr:`~reframe.core.systems.SystemPartition.features` attribute of the :attr:`~reframe.core.pipeline.RegressionTest.current_partition` and can also be selected through the extended syntax of :attr:`~reframe.core.pipeline.RegressionTest.valid_systems`.
   The values of this list must be alphanumeric strings starting with a non-digit character and may also contain a ``-``.

   .. versionadded:: 3.11.0


.. py:attribute:: systems.partitions.extras

   :required: No
   :default: ``{}``

   User defined attributes of the partition.

   These are accessible through the :attr:`~reframe.core.systems.SystemPartition.extras` attribute of the :attr:`~reframe.core.pipeline.RegressionTest.current_partition` and can also be selected through the extended syntax of :attr:`~reframe.core.pipeline.RegressionTest.valid_systems`.
   The attributes of this object must be alphanumeric strings starting with a non-digit character and their values can be of any type.

   By default, the values of the :attr:`~config.systems.partitions.scheduler` and :attr:`~config.systems.partitions.launcher` of the partition are added to the partition's extras, if not already present.

   .. versionadded:: 3.5.0

   .. versionchanged:: 4.6.0

      The default ``scheduler`` and ``launcher`` extras are added.

.. _container-platform-configuration:


Container Platform Configuration
================================

ReFrame can launch containerized applications, but you need to configure properly a system partition in order to do that by defining a container platform configuration.

.. py:attribute:: systems.partitions.container_platforms.type

   :required: Yes

   The type of the container platform.
   Available values are the following:

   - ``Apptainer``: The `Apptainer <https://apptainer.org/>`__ container runtime.
   - ``Docker``: The `Docker <https://www.docker.com/>`__ container runtime.
   - ``Sarus``: The `Sarus <https://sarus.readthedocs.io/>`__ container runtime.
   - ``Shifter``: The `Shifter <https://github.com/NERSC/shifter>`__ container runtime.
   - ``Singularity``: The `Singularity <https://sylabs.io/>`__ container runtime.


.. py:attribute:: systems.partitions.container_platforms.default

   :required: No

   If set to ``true``, this is the default container platform of this partition.
   If not specified, the default container platform is assumed to be the first in the list of :attr:`~config.systems.partitions.container_platforms`.

   .. versionadded:: 3.12.0


.. py:attribute:: systems.partitions.container_platforms.modules

   :required: No
   :default: ``[]``

   A list of `environment module objects <#module-objects>`__ to be loaded when running containerized tests using this container platform.


.. py:attribute:: systems.partitions.container_platforms.env_vars

   :required: No
   :default: ``[]``

   List of environment variables to be set when running containerized tests using this container platform.
   Each environment variable is specified as a two-element list containing the variable name and its value.
   You may reference other environment variables when defining an environment variable here.
   ReFrame will expand its value.
   Variables are set after the environment modules are loaded.

   .. versionadded:: 4.0.0

.. py:attribute:: systems.partitions.container_platforms.variables

   .. deprecated:: 4.0.0
      Please use :attr:`~systems.partitions.container_platforms.env_vars` instead.
      If specified in conjunction with :attr:`~systems.partitions.container_platforms.env_vars`, it will be ignored.


Custom Job Scheduler Resources
==============================

ReFrame allows you to define custom scheduler resources for each partition that can then be transparently accessed through the :attr:`~reframe.core.pipeline.RegressionTest.extra_resources` attribute of a test or from an environment.

.. py:attribute:: systems.partitions.resources.name

   :required: Yes

  The name of this resources.
  This name will be used to request this resource in a regression test's :attr:`~reframe.core.pipeline.RegressionTest.extra_resources`.


.. py:attribute:: systems.partitions.resources.options

   :required: No
   :default: ``[]``

   A list of options to be passed to this partition’s job scheduler.
   The option strings can contain placeholders of the form ``{placeholder_name}``.
   These placeholders may be replaced with concrete values by a regression test through the :attr:`~reframe.core.pipeline.RegressionTest.extra_resources` attribute.

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
   An example is the `DataWarp <https://www.nersc.gov/assets/Uploads/dw-overview-overby.pdf>`__ functionality of Slurm which is supported by the ``#DW`` prefix.
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


   A regression test that needs to make use of that resource, it can set its :attr:`~reframe.core.pipeline.RegressionTest.extra_resources` as follows:

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
    For the ``pbs`` and ``torque`` backends, options accepted in the :attr:`~config.systems.partitions.access` and :attr:`~config.systems.partitions.resources` parameters may either refer to actual ``qsub`` options or may just be resources specifications to be passed to the ``-l`` option.
    The backend assumes a ``qsub`` option, if the options passed in these attributes start with a ``-``.


Environment Configuration
=========================

Environments defined in this section will be used for running regression tests.
They are associated with `system partitions <#system-partition-configuration>`__.


.. py:attribute:: environments.name

   :required: Yes

   The name of this environment.


.. py:attribute:: environments.modules

   :required: No
   :default: ``[]``

   A list of `environment module objects <#module-objects>`__ to be loaded when this environment is loaded.


.. py:attribute:: environments.env_vars

   :required: No
   :default: ``[]``

   A list of environment variables to be set when loading this environment.
   Each environment variable is specified as a two-element list containing the variable name and its value.
   You may reference other environment variables when defining an environment variable here.
   ReFrame will expand its value.
   Variables are set after the environment modules are loaded.

   .. versionadded:: 4.0.0

.. py:attribute:: environments.variables

   .. deprecated:: 4.0.0
      Please use :attr:`~environments.env_vars` instead.
      If specified in conjunction with :attr:`~environments.env_vars`, it will be ignored.


.. py:attribute:: environments.features

   :required: No
   :default: ``[]``

   User defined features of the environment.
   These are accessible through the :attr:`~reframe.core.environments.Environment.features` attribute of the :attr:`~reframe.core.pipeline.RegressionTest.current_environ` and can also be selected through the extended syntax of :attr:`~reframe.core.pipeline.RegressionTest.valid_prog_environs`.
   The values of this list must be alphanumeric strings starting with a non-digit character and may also contain a ``-``.

   .. versionadded:: 3.11.0


.. py:attribute:: environments.extras

   :required: No
   :default: ``{}``

   User defined attributes of the environment.
   These are accessible through the :attr:`~reframe.coreenvironments.Environment.extras` attribute of the :attr:`~reframe.core.pipeline.RegressionTest.current_environ` and can also be selected through the extended syntax of :attr:`~reframe.core.pipeline.RegressionTest.valid_prog_environs`.
   The attributes of this object must be alphanumeric strings starting with a non-digit character and their values can be of any type.

   .. versionadded:: 3.9.1


.. py:attribute:: environments.prepare_cmds

   :required: No
   :default: ``[]``

   List of shell commands to be emitted before any commands that load the environment.

   .. versionadded:: 4.3.0


.. py:attribute:: environments.cc

   :required: No
   :default: ``"cc"``

   The C compiler to be used with this environment.


.. py:attribute:: environments.cxx

   :required: No
   :default: ``"CC"``

   The C++ compiler to be used with this environment.


.. py:attribute:: environments.ftn

   :required: No
   :default: ``"ftn"``

   The Fortran compiler to be used with this environment.


.. py:attribute:: environments.cppflags

   :required: No
   :default: ``[]``

   A list of C preprocessor flags to be used with this environment by default.


.. py:attribute:: environments.cflags

   :required: No
   :default: ``[]``

   A list of C flags to be used with this environment by default.


.. py:attribute:: environments.cxxflags

   :required: No
   :default: ``[]``

   A list of C++ flags to be used with this environment by default.


.. py:attribute:: environments.fflags

   :required: No
   :default: ``[]``

   A list of Fortran flags to be used with this environment by default.


.. py:attribute:: environments.ldflags

   :required: No
   :default: ``[]``

   A list of linker flags to be used with this environment by default.


.. py:attribute:: environments.nvcc

   :required: No
   :default: ``"nvcc"``

   The NVIDIA CUDA compiler to be used with this environment.

   .. versionadded:: 4.6


.. py:attribute:: environments.target_systems

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


.. py:attribute:: environments.resources

   :required: No
   :default: ``{}``

   Scheduler resources associated with this environments.

   This is the equivalent of a test's :attr:`~reframe.core.pipeline.RegressionTest.extra_resources`.

   .. versionadded:: 4.6


.. _logging-config-reference:

Logging Configuration
=====================

Logging in ReFrame is handled by logger objects which further delegate message to *logging handlers* which are eventually responsible for emitting or sending the log records to their destinations.
You may define different logger objects per system but *not* per partition.


.. py:attribute:: logging.level

   :required: No
   :default: ``"undefined"``

   The level associated with this logger object.
   There are the following levels in decreasing severity order:

   - ``critical``: Catastrophic errors; the framework cannot proceed with its execution.
   - ``error``: Normal errors; the framework may or may not proceed with its execution.
   - ``warning``: Warning messages.
   - ``info``: Informational messages.
   - ``verbose``: More informational messages.
   - ``debug``: Debug messages.
   - ``debug2``: Further debug messages.
   - ``undefined``: This is the lowest level; does not filter any message.

   If a message is logged by the framework, its severity level will be checked by the logger and if it is higher from the logger's level, it will be passed down to its handlers.


   .. versionadded:: 3.3
      The ``debug2`` and ``undefined`` levels are added.

   .. versionchanged:: 3.3
      The default level is now ``undefined``.


.. py:attribute:: logging.handlers

   :required: Yes

   A list of logging handlers responsible for handling normal framework output.


.. py:attribute:: logging.handlers_perflog

   :required: Yes

   A list of logging handlers responsible for handling performance data from tests.


.. py:attribute:: logging.perflog_compat

   :required: No
   :default: ``false``

   Emit a separate log record for each performance variable.
   Set this option to ``true`` if you want to keep compatibility with the performance logging prior to ReFrame 4.0.

.. py:attribute:: logging.target_systems

   :required: No
   :default: ``["*"]``

   A list of systems or system/partitions combinations that this logging configuration is valid for.
   For a detailed description of this property, have a look at the :attr:`~environments.target_systems` definition for environments.



Common logging handler properties
---------------------------------

All logging handlers share the following set of common attributes:


.. py:attribute:: logging.handlers.type

.. py:attribute:: logging.handlers_perflog.type

   :required: Yes

   The type of handler.
   There are the following types available:

   - ``file``: This handler sends log records to file.
     See `here <#the-file-log-handler>`__ for more details.
   - ``filelog``: This handler sends performance log records to files.
     See `here <#the-filelog-log-handler>`__ for more details.
   - ``graylog``: This handler sends performance log records to Graylog.
     See `here <#the-graylog-log-handler>`__ for more details.
   - ``stream``: This handler sends log records to a file stream.
     See `here <#the-stream-log-handler>`__ for more details.
   - ``syslog``: This handler sends log records to a Syslog facility.
     See `here <#the-syslog-log-handler>`__ for more details.
   - ``httpjson``: This handler sends log records in JSON format using HTTP post requests.
     See `here <#the-httpjson-log-handler>`__ for more details.


.. py:attribute:: logging.handlers.level

.. py:attribute:: logging.handlers_perflog.level

   :required: No
   :default: ``"info"``

   The `log level <#config.logging.level>`__ associated with this handler.


.. py:attribute:: logging.handlers.format

.. py:attribute:: logging.handlers_perflog.format

   :required: No
   :default: ``"%(message)s"``

   Log record format string.

   ReFrame accepts all log record attributes from Python's `logging <https://docs.python.org/3.8/library/logging.html#logrecord-attributes>`__ mechanism and adds the following attributes:

   .. csv-table::

      ``%(check_build_locally)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.build_locally` attribute.
      ``%(check_build_time_limit)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.build_time_limit` attribute.
      ``%(check_descr)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.descr` attribute.
      ``%(check_display_name)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.display_name` attribute.
      ``%(check_environ)s``, The name of the test's :attr:`~reframe.core.pipeline.RegressionTest.current_environ`.
      ``%(check_env_vars)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.env_vars` attribute.
    ``%(check_exclusive_access)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.exclusive_access` attribute.
      ``%(check_executable)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.executable` attribute.
      ``%(check_executable_opts)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.executable_opts` attribute.
      ``%(check_extra_resources)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.extra_resources` attribute.
      ``%(check_hashcode)s``, The unique hash associated with this test.
      ``%(check_info)s``, Various information about this test; essentially the return value of the test's :func:`~reframe.core.pipeline.RegressionTest.info` function.
      ``%(check_job_completion_time)s``, Same as the ``(check_job_completion_time_unix)s`` but formatted according to ``datefmt``.
      ``%(check_job_completion_time_unix)s``, The completion time of the associated run job (see :attr:`~reframe.core.schedulers.Job.completion_time`).
      ``%(check_job_exitcode)s``, The exit code of the associated run job.
      ``%(check_job_nodelist)s``, The list of nodes that the associated run job has run on.
      ``%(check_job_submit_time)s``, The submission time of the associated run job (see :attr:`~reframe.core.schedulers.Job.submit_time`).
      ``%(check_jobid)s``, The ID of the associated run job.
      ``%(check_keep_files)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.keep_files` attribute.
      ``%(check_local)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.local` attribute.
      ``%(check_maintainers)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.maintainers` attribute.
      ``%(check_max_pending_time)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.max_pending_time` attribute.
      ``%(check_modules)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.modules` attribute.
      ``%(check_name)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.name` attribute.
      ``%(check_num_cpus_per_task)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.num_cpus_per_task` attribute.
      ``%(check_num_gpus_per_node)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.num_gpus_per_node` attribute.
      ``%(check_num_tasks)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.num_tasks` attribute.
      ``%(check_num_tasks_per_core)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.num_tasks_per_core` attribute.
      ``%(check_num_tasks_per_node)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.num_tasks_per_node` attribute.
      ``%(check_num_tasks_per_socket)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.num_tasks_per_socket` attribute.
      ``%(check_outputdir)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.outputdir` attribute.
      ``%(check_partition)s``, The name of the test's :attr:`~reframe.core.pipeline.RegressionTest.current_partition`.
      ``%(check_perfvalues)s``, All the performance variables of the test combined. These will be formatted according to :attr:`~config.logging.handlers_perflog.format_perfvars`.
      ``%(check_postbuild_cmds)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.postbuild_cmds` attribute.
      ``%(check_postrun_cmds)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.postrun_cmds` attribute.
      ``%(check_prebuild_cmds)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.prebuild_cmds` attribute.
      ``%(check_prefix)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.prefix` attribute.
      ``%(check_prerun_cmds)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.prerun_cmds` attribute.
      ``%(check_result)s``, The result of the test (``pass`` or ``fail``).
      ``%(check_readonly_files)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.readonly_files` attribute.
      ``%(check_short_name)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.short_name` attribute.
      ``%(check_sourcepath)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.sourcepath` attribute.
      ``%(check_sourcesdir)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.sourcesdir` attribute.
      ``%(check_stagedir)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.stagedir` attribute.
      ``%(check_strict_check)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.strict_check` attribute.
      ``%(check_system)s``, The name of the test's :attr:`~reframe.core.pipeline.RegressionTest.current_system`.
      ``%(check_tags)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.tags` attribute.
      ``%(check_time_limit)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.time_limit` attribute.
      ``%(check_unique_name)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.unique_name` attribute.
      ``%(check_use_multithreading)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.use_multithreading` attribute.
      ``%(check_valid_prog_environs)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.valid_prog_environs` attribute.
      ``%(check_valid_systems)s``, The value of the :attr:`~reframe.core.pipeline.RegressionTest.valid_systems` attribute.
      ``%(check_variables)s``, DEPRECATED: Please use ``%(check_env_vars)s`` instead.
      ``%(osuser)s``, The name of the OS user running ReFrame.
      ``%(osgroup)s``, The name of the OS group running ReFrame.
      ``%(version)s``, The ReFrame version.

   ReFrame allows you to log any test variable, parameter or property if they are marked as "loggable".
   The log record attribute will have the form ``%(check_NAME)s`` where ``NAME`` is the variable name, the parameter name or the property name that is marked as loggable.

   There is also the special ``%(check_#ALL)s`` format specifier which expands to all the loggable test attributes.
   These include all the above specifiers and any additional loggable variables or parameters defined by the test.
   On expanding this specifier, ReFrame will try to guess the delimiter to use for separating the different attributes based on the existing format.
   If it cannot guess it, it will default to ``|``.

   Since this can lead to very long records, you may consider using it with the :attr:`~logging.handlers_perflog..filelog..ignore_keys` parameter to filter out some attributes that are not of interest.

.. versionadded:: 3.3
   Allow arbitrary test attributes to be logged.

.. versionadded:: 3.4.2
   Allow arbitrary job attributes to be logged.

.. versionchanged:: 3.11.0
   Limit the number of attributes that can be logged. User attributes or properties must be explicitly marked as "loggable" in order to be selectable for logging.

.. versionadded:: 4.0
   The ``%(check_result)s`` specifier is added.

.. versionadded:: 4.3
   The ``%(check_#ALL)s`` special specifier is added.


.. py:attribute:: logging.handlers.format_perfvars
.. py:attribute:: logging.handlers_perflog.format_perfvars

   :required: No
   :default: ``""``

   Format specifier for logging the performance variables.

   This defines how the ``%(check_perfvalues)s`` will be formatted.
   Since a test may define multiple performance variables, the formatting specified in this field will be repeated for each performance variable sequentially in the *same* line.

   .. important::
      The last character of this format will be interpreted as the final delimiter of the formatted performance variables to the rest of the record.

   The following log record attributes are defined additionally by this format specifier:

   .. csv-table::
      :header: "Log record attribute", "Description"

      ``%(check_perf_lower_thres)s``, The lower threshold of the logged performance variable.
      ``%(check_perf_ref)s``, The reference value of the logged performance variable.
      ``%(check_perf_unit)s``, The measurement unit of the logged performance variable.
      ``%(check_perf_upper_thres)s``, The upper threshold of the logged performance variable.
      ``%(check_perf_value)s``, The actual value of the logged performance variable.
      ``%(check_perf_var)s``, The name of the logged performance variable.


   .. important::
      ReFrame versions prior to 4.0 logged a separate line for each performance variable and the ``%(check_perf_*)s`` attributes could be used directly in the ``format``.
      You can re-enable this behavior by setting the :attr:`config.logging.perflog_compat` logging configuration parameter.


   .. versionadded:: 4.0.0

.. py:attribute:: logging.handlers.datefmt

.. object:: logging.handlers_perflog.datefmt

   :required: No
   :default: ``"%FT%T"``

   Time format to be used for printing timestamps fields.
   There are two timestamp fields available: ``%(asctime)s`` and ``%(check_job_completion_time)s``.
   In addition to the format directives supported by the standard library's `time.strftime() <https://docs.python.org/3.8/library/time.html#time.strftime>`__ function, ReFrame allows you to use the ``%:z`` directive -- a GNU ``date`` extension --  that will print the time zone difference in a RFC3339 compliant way, i.e., ``+/-HH:MM`` instead of ``+/-HHMM``.


The ``file`` log handler
------------------------

This log handler handles output to normal files.
The additional properties for the ``file`` handler are the following:


.. py:attribute:: logging.handlers..file..name

.. py:attribute:: logging.handlers_perflog..file..name

   :required: No

   The name of the file where this handler will write log records.
   If not specified, ReFrame will create a log file prefixed with ``rfm-`` in the system's temporary directory.

   .. versionchanged:: 3.3
      The ``name`` parameter is no more required and the default log file resides in the system's temporary directory.


.. py:attribute:: logging.handlers..file..append

.. py:attribute:: logging.handlers_perflog..file..append

   :required: No
   :default: ``false``

   Controls whether this handler should append to its file or not.


.. py:attribute:: logging.handlers..file..timestamp

.. py:attribute:: logging.handlers_perflog..file..timestamp

   :required: No
   :default: ``false``

   Append a timestamp to this handler's log file.
   This property may also accept a date format as described in the :attr:`~config.logging.handlers.datefmt` property.
   If the handler's :attr:`~config.logging.handlers..file..name` property is set to ``filename.log`` and this property is set to ``true`` or to a specific timestamp format, the resulting log file will be ``filename_<timestamp>.log``.


.. _filelog-handler:

The ``filelog`` log handler
---------------------------

This handler is meant for performance logging only and logs the performance of a test in one or more files.
The additional properties for the ``filelog`` handler are the following:


.. py:attribute:: logging.handlers_perflog..filelog..basedir

   :required: No
   :default: ``"./perflogs"``

   The base directory of performance data log files.


.. py:attribute:: logging.handlers_perflog..filelog..ignore_keys

   A list of log record `format specifiers <#config.logging.handlers.format>`__ that will be ignored by the special ``%(check_#ALL)s`` specifier.

   .. versionadded:: 4.3


.. py:attribute:: logging.handlers_perflog..filelog..prefix

   :required: Yes

   This is a directory prefix (usually dynamic), appended to the :attr:`~config.logging.handlers_perflog..filelog..basedir`, where the performance logs of a test will be stored.
   This attribute accepts any of the check-specific `formatting placeholders <#config.logging.handlers_perflog.format>`__.
   This allows to create dynamic paths based on the current system, partition and/or programming environment a test executes with.
   For example, a value of ``%(check_system)s/%(check_partition)s`` would generate the following structure of performance log files:


   .. code-block:: none

     {basedir}/
        system1/
            partition1/
                <test_class_name>.log
            partition2/
                <test_class_name>.log
            ...
        system2/
        ...


.. py:attribute:: logging.handlers_perflog..filelog..append

   :required: No
   :default: ``true``

   Open each log file in append mode.


.. versionchanged:: 4.0.0

   The ``filelog`` handler is very cautious when generating a test log file: if a change is detected in the information that is being logged, the hanlder will not append to the same file, but it will instead create a new one, saving the old file using the ``.h<N>`` suffix, where ``N`` is an integer that is increased every time a new file is being created due to such changes.
   Examples of changes in the logged information are when the log record format changes or a new performance metric is added, deleted or has its name changed.
   This behavior guarantees that each log file is consistent and it will not break existing parsers.

.. versionchanged:: 4.3

   In the generated log file, the name of the test class name is used instead of the test's short name (which included the test's hash).
   This allows the results of different variants of a parameterized test to be stored in the same log file facilitating post-processing.


The ``graylog`` log handler
---------------------------

This handler is meant for performance logging only and sends log records to a `Graylog <https://www.graylog.org/>`__ server.
The additional properties for the ``graylog`` handler are the following:

.. py:attribute:: logging.handlers_perflog..graylog..address

   :required: Yes

   The address of the Graylog server defined as ``host:port``.


.. py:attribute:: logging.handlers_perflog..graylog..extras

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


Although the :attr:`~config.logging.handlers.format` attribute is defined for this handler, it is not only the log message that will be transmitted the Graylog server.
This handler transmits the whole log record, meaning that all the information will be available and indexable at the remote end.


The ``stream`` log handler
--------------------------

This handler sends log records to a file stream.
The additional properties for the ``stream`` handler are the following:


.. py:attribute:: logging.handlers..stream..name

.. py:attribute:: logging.handlers_perflog..stream..name

   :required: No
   :default: ``"stdout"``

   The name of the file stream to send records to.
   There are only two available streams:

   - ``stdout``: the standard output.
   - ``stderr``: the standard error.


The ``syslog`` log handler
--------------------------

This handler sends log records to UNIX syslog.
The additional properties for the ``syslog`` handler are the following:


.. py:attribute:: logging.handlers..syslog..socktype

.. py:attribute:: logging.handlers_perflog..syslog..socktype

   :required: No
   :default: ``"udp"``

   The socket type where this handler will send log records to.
   There are two socket types:

   - ``udp``: A UDP datagram socket.
   - ``tcp``: A TCP stream socket.


.. py:attribute:: logging.handlers..syslog..facility

.. py:attribute:: logging.handlers_perflog..syslog..facility

   :required: No
   :default: ``"user"``

   The Syslog facility where this handler will send log records to.
   The list of supported facilities can be found `here <https://docs.python.org/3.8/library/logging.handlers.html#logging.handlers.SysLogHandler.encodePriority>`__.


.. py:attribute:: logging.handlers..syslog..address

.. py:attribute:: logging.handlers_perflog..syslog..address

   :required: Yes

   The socket address where this handler will connect to.
   This can either be of the form ``<host>:<port>`` or simply a path that refers to a Unix domain socket.


The ``httpjson`` log handler
----------------------------

This handler sends log records in JSON format to a server using HTTP POST requests.
The additional properties for the ``httpjson`` handler are the following:

.. py:attribute:: logging.handlers_perflog..httpjson..url

   :required: Yes

   The URL to be used in the HTTP(S) request server.


.. py:attribute:: logging.handlers_perflog..httpjson..extra_headers

   :required: No
   :default: ``{}``

   A set of optional key/value pairs to be sent as HTTP message headers (e.g. API keys).
   These may depend on the server configuration.

   .. versionadded:: 4.2


.. py:attribute:: logging.handlers_perflog..httpjson..extras

   :required: No
   :default: ``{}``

   A set of optional key/value pairs to be passed with each log record to the server.
   These may depend on the server configuration.

.. py:attribute:: logging.handlers_perflog..httpjson..ignore_keys

   :required: No
   :default: ``[]``

   These keys will be excluded from the log record that will be sent to the server.


The ``httpjson`` handler sends log messages in JSON format using an HTTP POST request to the specified URL.

An example configuration of this handler for performance logging is shown here:

.. code:: python

   {
       'type': 'httpjson',
       'url': 'http://httpjson-server:12345/rfm',
       'level': 'info',
       'extra_headers': {'Authorization': 'Token YOUR_API_TOKEN'},
       'extras': {
           'facility': 'reframe',
           'data-version': '1.0'
       },
       'ignore_keys': ['check_perfvalues']
   }


This handler transmits the whole log record, meaning that all the information will be available and indexable at the remote end.

.. py:attribute:: logging.handlers_perflog..httpjson..debug

   :required: No
   :default: ``false``

   If set, the ``httpjson`` handler will not attempt to send the data to the server, but it will instead dump the JSON record in the current directory.
   The filename has the following form: ``httpjson_record_<timestamp>.json``.

   .. versionadded:: 4.1


.. py:attribute:: logging.handlers_perflog..httpjson..json_formatter

   A callable for converting the log record into JSON.

   The formatter's signature is the following:

   .. py:function:: json_formatter(record: object, extras: Dict[str, str], ignore_keys: Set[str]) -> str

      :arg record: The prepared log record.
         The log record is a simple Python object with all the attributes listed in :attr:`~config.logging.handlers.format`, as well as all the default Python `log record <https://docs.python.org/3.8/library/logging.html#logrecord-attributes>`__ attributes.
         In addition to those, there is also the special :attr:`__rfm_check__` attribute that contains a reference to the actual test for which the performance is being logged.
      :arg extras: Any extra attributes specified in :attr:`~config.logging.handlers_perflog..httpjson..extras`.
      :arg ignore_keys: The set of keys specified in :attr:`~config.logging.handlers_perflog..httpjson..ignore_keys`.
         ReFrame always adds the default Python log record attributes in this set.
      :returns: A string representation of the JSON record to be sent to the server or :obj:`None` if the record should not be sent to the server.

   .. note::
      This configuration parameter can only be used in a Python configuration file.

   .. versionadded:: 4.1



.. _exec-mode-config:

Execution Mode Configuration
============================

ReFrame allows you to define groups of command line options that are collectively called *execution modes*.
An execution mode can then be selected from the command line with the :option:`--mode` option.
The options of an execution mode will be passed to ReFrame as if they were specified in the command line.


.. py:attribute:: modes.name

   :required: Yes

   The name of this execution mode.
   This can be used with the :option:`--mode` command line option to invoke this mode.


.. py:attribute:: modes.options

   :required: No
   :default: ``[]``

   The command-line options associated with this execution mode.


.. py:attribute:: modes.target_systems

   :required: No
   :default: ``["*"]``

   A list of systems *only* that this execution mode is valid for.
   For a detailed description of this property, have a look at the :attr:`~environments.target_systems` definition for environments.


General Configuration
=====================

.. py:attribute:: general.check_search_path

   :required: No
   :default: ``["${RFM_INSTALL_PREFIX}/checks/"]``

   A list of paths (files or directories) where ReFrame will look for regression test files.
   If the search path is set through the environment variable, it should be a colon separated list.
   If specified from command line, the search path is constructed by specifying multiple times the command line option.


.. py:attribute:: general.check_search_recursive

   :required: No
   :default: ``false``

   Search directories in the `search path <#general.check_search_path>`__ recursively.



.. py:attribute:: general.clean_stagedir

   :required: No
   :default: ``true``

   Clean stage directory of tests before populating it.

   .. versionadded:: 3.1


.. py:attribute:: general.colorize

   :required: No
   :default: ``true``

   Use colors in output.
   The command-line option sets the configuration option to ``false``.


.. py:attribute:: general.compress_report

   :required: No
   :default: ``false``

   Compress the generated run report file.
   See the documentation of the :option:`--compress-report` option for more information.

   .. versionadded:: 3.12.0


.. py:attribute:: general.dump_pipeline_progress

   Dump pipeline progress for the asynchronous execution policy in ``pipeline-progress.json``.
   This option is meant for debug purposes only.

   :required: No
   :default: ``False``

   .. versionadded:: 3.10.0


.. py:attribute:: general.flex_alloc_strict

   :required: No
   :default: ``False``

   Fail flexible tests if their minimum task requirement is not satisfied.

   .. versionadded:: 4.7


.. py:attribute:: general.git_timeout

   :required: No
   :default: 5

   Timeout value in seconds used when checking if a git repository exists.


.. py:attribute:: general.pipeline_timeout

   Timeout in seconds for advancing the pipeline in the asynchronous execution policy.

   ReFrame's asynchronous execution policy will try to advance as many tests as possible in their pipeline, but some tests may take too long to proceed (e.g., due to copying of large files) blocking the advancement of previously started tests.
   If this timeout value is exceeded and at least one test has progressed, ReFrame will stop processing new tests and it will try to further advance tests that have already started.
   See :ref:`pipeline-timeout` for more guidance on how to set this.

   :required: No
   :default: ``10``

   .. versionadded:: 3.10.0


.. py:attribute:: general.perf_info_level

   :required: No
   :default: ``"info"``

   The log level at which the immediate performance info will be printed.

   As soon as a performance test is finished, ReFrame will log its performance on the standard output immediately.
   This option controls at which verbosity level this info will appear.

   For a list of available log levels, refer to the :attr:`~config.logging.level` logger configuration parameter.

   .. versionadded:: 4.0.0


.. py:attribute:: general.remote_detect

   :required: No
   :default: ``false``

   Try to auto-detect processor information of remote partitions as well.
   This may slow down the initialization of the framework, since it involves submitting auto-detection jobs to the remote partitions.

   .. versionadded:: 3.7.0


.. py:attribute:: general.remote_workdir

   :required: No
   :default: ``"."``

   The temporary directory prefix that will be used to create a fresh ReFrame clone, in order to auto-detect the processor information of a remote partition.

   .. versionadded:: 3.7.0


.. py:attribute:: general.ignore_check_conflicts

   :required: No
   :default: ``false``

   Ignore test name conflicts when loading tests.

   .. deprecated:: 3.8.0
      This option will be removed in a future version.



.. py:attribute:: general.trap_job_errors

   :required: No
   :default: ``false``

   Trap command errors in the generated job scripts and let them exit immediately.

   .. versionadded:: 3.2


.. py:attribute:: general.keep_stage_files

   :required: No
   :default: ``false``

   Keep stage files of tests even if they succeed.


.. py:attribute:: general.module_map_file

   :required: No
   :default: ``""``

   File containing module mappings.


.. py:attribute:: general.module_mappings

   :required: No
   :default: ``[]``

   A list of module mappings.
   If specified through the environment variable, the mappings must be separated by commas.
   If specified from command line, multiple module mappings are defined by passing the command line option multiple times.


.. py:attribute:: general.non_default_craype

   :required: No
   :default: ``false``

   Test a non-default Cray Programming Environment.
   This will emit some special instructions in the generated build and job scripts.
   See also :option:`--non-default-craype` for more details.


.. py:attribute:: general.purge_environment

   :required: No
   :default: ``false``

   Purge any loaded environment modules before running any tests.


.. py:attribute:: general.report_file

   :required: No
   :default: ``"${HOME}/.reframe/reports/run-report-{sessionid}.json"``

   The file where ReFrame will store its report.

   .. versionadded:: 3.1
   .. versionchanged:: 3.2
      Default value has changed to avoid generating a report file per session.
   .. versionchanged:: 4.0.0
      Default value was reverted back to generate a new file per run.


.. py:attribute:: general.report_junit

   :required: No
   :default: ``null``

   The file where ReFrame will store its report in JUnit format.
   The report adheres to the XSD schema `here <https://github.com/windyroad/JUnit-Schema/blob/master/JUnit.xsd>`__.

   .. versionadded:: 3.6.0


.. py:attribute:: general.resolve_module_conflicts

   :required: No
   :default: ``true``

   ReFrame by default resolves any module conflicts and emits the right sequence of ``module unload`` and ``module load`` commands, in order to load the requested modules.
   This option disables this behavior if set to ``false``.

   You should avoid using this option for modules system that cannot handle module conflicts automatically, such as early Tmod verions.

   Disabling the automatic module conflict resolution, however, can be useful when modules in a remote system partition are not present on the host where ReFrame runs.
   In order to resolve any module conflicts and generate the right load sequence of modules, ReFrame loads temporarily the requested modules and tracks any conflicts along the way.
   By disabling this option, ReFrame will simply emit the requested ``module load`` commands without attempting to load any module.


   .. versionadded:: 3.6.0


.. py:attribute:: general.save_log_files

   :required: No
   :default: ``false``

   Save any log files generated by ReFrame to its output directory


.. py:attribute:: general.target_systems

   :required: No
   :default: ``["*"]``

   A list of systems or system/partitions combinations that these general options are valid for.
   For a detailed description of this property, have a look at the :attr:`~environments.target_systems` definition for environments.


.. py:attribute:: general.timestamp_dirs

   :required: No
   :default: ``""``

   Append a timestamp to ReFrame directory prefixes.
   Valid formats are those accepted by the `time.strftime() <https://docs.python.org/3.8/library/time.html#time.strftime>`__ function.
   If specified from the command line without any argument, ``"%FT%T"`` will be used as a time format.


.. py:attribute:: general.unload_modules

   :required: No
   :default: ``[]``

   A list of `environment module objects <#module-objects>`__ to unload before executing any test.
   If specified using an the environment variable, a space separated list of modules is expected.
   If specified from the command line, multiple modules can be passed by passing the command line option multiple times.


.. py:attribute:: general.use_login_shell

   :required: No
   :default: ``false``

   Use a login shell for the generated job scripts.
   This option will cause ReFrame to emit ``-l`` in the shebang of shell scripts.
   This option, if set to ``true``, may cause ReFrame to fail, if the shell changes permanently to a different directory during its start up.


.. py:attribute:: general.user_modules

   :required: No
   :default: ``[]``

   A list of `environment module objects <#module-objects>`__ to be loaded before executing any test.
   If specified using an the environment variable, a space separated list of modules is expected.
   If specified from the command line, multiple modules can be passed by passing the command line option multiple times.


.. py:attribute:: general.verbose

   :required: No
   :default: 0

   Set the verbosity level of the output.
   The higher the number, the more verbose the output will be.
   If set to a negative number, this will decrease the verbosity level.


Module Objects
==============

.. versionadded:: 3.3


A *module object* in ReFrame's configuration represents an environment module.
It can either be a simple string or a JSON object with the following attributes:

.. attribute:: environments.modules.name
.. attribute:: systems.modules.name
.. attribute:: systems.partitions.modules.name
.. attribute:: systems.partitions.container_platforms.modules.name

   :required: Yes

   The name of the module.


.. attribute:: environments.modules.collection
.. attribute:: systems.modules.collection
.. attribute:: systems.partitions.modules.collection
.. attribute:: systems.partitions.container_platforms.modules.collection

   :required: No
   :default: ``false``

   A boolean value indicating whether this module refers to a module collection.
   Module collections are treated differently from simple modules when loading.

.. attribute:: environments.modules.path
.. attribute:: systems.modules.path
.. attribute:: systems.partitions.modules.path
.. attribute:: systems.partitions.container_platforms.modules.path

   :required: No
   :default: ``null``

   If the module is not present in the default ``MODULEPATH``, the module's location can be specified here.
   ReFrame will make sure to set and restore the ``MODULEPATH`` accordingly for loading the module.


   .. versionadded:: 3.5.0


.. seealso::

   Module collections with `Environment Modules <https://modules.readthedocs.io/en/latest/MIGRATING.html#module-collection>`__ and `Lmod <https://lmod.readthedocs.io/en/latest/010_user.html#user-collections>`__.


Processor Info
==============

.. versionadded:: 3.5.0

A *processor info object* in ReFrame's configuration is used to hold information about the processor of a system partition and is made available to the tests through the :attr:`processor <reframe.core.systems.SystemPartition.processor>` attribute of the :attr:`current_partition <reframe.core.pipeline.RegressionTest.current_partition>`.

.. note::
   In the following the term *logical CPUs* refers to the smallest processing unit recognized by the OS.
   Depending on the microarchitecture, this can either be a core or a hardware thread in processors that support simultaneous multithreading and this feature is enabled.
   Therefore, properties such as :attr:`num_cpus_per_core` may have a value greater than one.

.. attribute:: systems.partitions.processor.arch

   :required: No
   :default: ``None``

   The microarchitecture of the processor.


.. attribute:: systems.partitions.processor.model

   :required: No
   :default: ``None``

   The model of the processor.

   .. versionadded:: 4.6

.. attribute:: systems.partitions.processor.platform

   :required: No
   :default: ``None``

   The hardware platform for this processor (e.g., ``x86_64``, ``arm64`` etc.)

   .. versionadded:: 4.6

.. attribute:: systems.partitions.processor.num_cpus

   :required: No
   :default: ``None``

   Number of logical CPUs.


.. attribute:: systems.partitions.processor.num_cpus_per_core

   :required: No
   :default: ``None``

   Number of logical CPUs per core.


.. attribute:: systems.partitions.processor.num_cpus_per_socket

   :required: No
   :default: ``None``

   Number of logical CPUs per socket.


.. attribute:: systems.partitions.processor.num_sockets

   :required: No
   :default: ``None``

   Number of sockets.


.. attribute:: systems.partitions.processor.topology

   :required: No
   :default: ``None``

   Processor topology.
   An example follows:

   .. code-block:: python

      'topology': {
         'numa_nodes': ['0x000000ff'],
         'sockets': ['0x000000ff'],
         'cores': ['0x00000003', '0x0000000c',
                   '0x00000030', '0x000000c0'],
         'caches': [
            {
                  'type': 'L3',
                  'size': 6291456,
                  'linesize': 64,
                  'associativity': 0,
                  'num_cpus': 8,
                  'cpusets': ['0x000000ff']
            },
            {
                  'type': 'L2',
                  'size': 262144,
                  'linesize': 64,
                  'associativity': 4,
                  'num_cpus': 2,
                  'cpusets': ['0x00000003', '0x0000000c',
                              '0x00000030', '0x000000c0']
            },
            {
                  'type': 'L1',
                  'size': 32768,
                  'linesize': 64,
                  'associativity': 0,
                  'num_cpus': 2,
                  'cpusets': ['0x00000003', '0x0000000c',
                              '0x00000030', '0x000000c0']
            }
         ]
      }


Device Info
===========

.. versionadded:: 3.5.0


A *device info object* in ReFrame's configuration is used to hold information about a specific type of devices in a system partition and is made available to the tests through the :attr:`devices <reframe.core.systems.SystemPartition.processor>` attribute of the :attr:`current_partition <reframe.core.pipeline.RegressionTest.current_partition>`.


.. attribute:: systems.partitions.devices.type

   :required: No
   :default: ``None``

   The type of the device, for example ``"gpu"``.


.. attribute:: systems.partitions.devices.arch
   :noindex:

   :required: No
   :default: ``None``

   The microarchitecture of the device.

.. attribute:: systems.partitions.devices.model
   :noindex:

   :required: No
   :default: ``None``

   The model of the device.

   .. versionadded:: 4.6

.. attribute:: systems.partitions.devices.num_devices

   :required: No
   :default: ``None``

   Number of devices of this type inside the system partition.
