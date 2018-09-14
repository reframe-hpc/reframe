=================================
Configuring ReFrame for Your Site
=================================

ReFrame provides an easy and flexible way to configure new systems and new programming environments.
By default, it ships with a generic local system configured.
This should be enough to let you run ReFrame on a local computer as soon as the basic `software requirements <started.html#requirements>`__ are met.

As soon as a new system with its programming environments is configured, adapting an existing regression test could be as easy as just adding the system's name in the :attr:`valid_systems <reframe.core.pipeline.RegressionTest.valid_systems>` list and its associated programming environments in the :attr:`valid_prog_environs <reframe.core.pipeline.RegressionTest.valid_prog_environs>` list.

The Configuration File
----------------------

The configuration of systems and programming environments is performed by a special Python dictionary called ``site_configuration`` defined inside the file ``<install-dir>/reframe/settings.py``.

The ``site_configuration`` dictionary should define two entries, ``systems`` and ``environments``.
The former defines the systems that ReFrame may recognize, whereas the latter defines the available programming environments.

The following example shows a minimal configuration for the `Piz Daint <https://www.cscs.ch/computers/piz-daint/>`__ supercomputer at CSCS:

.. code-block:: python

   site_configuration = {
       'systems': {
           'daint': {
               'descr': 'Piz Daint',
               'hostnames': ['daint'],
               'modules_system': 'tmod',
               'partitions': {
                   'login': {
                       'scheduler': 'local',
                       'modules': [],
                       'access':  [],
                       'environs': ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi'],
                       'descr': 'Login nodes',
                       'max_jobs': 4
                   },

                   'gpu': {
                       'scheduler': 'nativeslurm',
                       'modules': ['daint-gpu'],
                       'access':  ['--constraint=gpu'],
                       'environs': ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi'],
                       'descr': 'Hybrid nodes (Haswell/P100)',
                       'max_jobs': 100
                   },

                   'mc': {
                       'scheduler': 'nativeslurm',
                       'modules': ['daint-mc'],
                       'access':  ['--constraint=mc'],
                       'environs': ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi'],
                       'descr': 'Multicore nodes (Broadwell)',
                       'max_jobs': 100
                   }
               }
           }
       },

       'environments': {
           '*': {
               'PrgEnv-cray': {
                   'type': 'ProgEnvironment',
                   'modules': ['PrgEnv-cray'],
               },

               'PrgEnv-gnu': {
                   'type': 'ProgEnvironment',
                   'modules': ['PrgEnv-gnu'],
               },

               'PrgEnv-intel': {
                   'type': 'ProgEnvironment',
                   'modules': ['PrgEnv-intel'],
               },

               'PrgEnv-pgi': {
                   'type': 'ProgEnvironment',
                   'modules': ['PrgEnv-pgi'],
               }
           }
       }
   }

System Configuration
--------------------

The list of supported systems is defined as a set of key/value pairs under key ``systems``.
Each system is a key/value pair, with the key being the name of the system and the value being another set of key/value pairs defining its attributes.
The valid attributes of a system are the following:

* ``descr``: A detailed description of the system (default is the system name).
* ``hostnames``: This is a list of hostname patterns that will be used by ReFrame when it tries to `auto-detect <#system-auto-detection>`__ the current system (default ``[]``).
* ``modules_system``: The modules system that should be used for loading environment modules on this system (default :class:`None`).
  Three types of modules systems are currently supported:

  - ``tmod``: The classic Tcl implementation of the `environment modules <https://sourceforge.net/projects/modules/files/Modules/modules-3.2.10/>`__.
  - ``tmod4``: The version 4 of the Tcl implementation of the `environment modules <http://modules.sourceforge.net/>`__.
  - ``lmod``: The Lua implementation of the `environment modules <https://lmod.readthedocs.io/en/latest/>`__.
* ``prefix``: Default regression prefix for this system (default ``.``).
* ``stagedir``: Default stage directory for this system (default :class:`None`).
* ``outputdir``: Default output directory for this system (default :class:`None`).
* ``perflogdir``: Default directory prefix for storing performance logs for this system (default :class:`None`).
* ``logdir``: `Deprecated since version 2.14 please use` ``perflogdir`` `instead.`
* ``resourcesdir``: Default directory for storing large resources (e.g., input data files, etc.) needed by regression tests for this system (default ``.``).
* ``partitions``: A set of key/value pairs defining the partitions of this system and their properties (default ``{}``).
  Partition configuration is discussed in the `next section <#partition-configuration>`__.

.. note::
  .. versionadded:: 2.8
    The ``modules_system`` key was introduced for specifying custom modules systems for different systems.

For a more detailed description of the ``prefix``, ``stagedir``, ``outputdir`` and ``perflogdir`` directories, please refer to the `"Configuring ReFrame Directories" <running.html#configuring-reframe-directories>`__ and `"Performance Logging" <running.html#performance-logging>`__ sections.

Partition Configuration
-----------------------

From the ReFrame's point of view, each system consists of a set of logical partitions.
These partitions need not necessarily correspond to real scheduler partitions.
For example, Piz Daint on the above example is split in *virtual partitions* using Slurm constraints.
Other systems may be indeed split into real scheduler partitions.

The partitions of a system are defined similarly to systems as a set of key/value pairs with the key being the partition name and the value being another set of key/value pairs defining the partition's attributes.
The available partition attributes are the following:

* ``descr``: A detailed description of the partition (default is the partition name).

* ``scheduler``: The job scheduler and parallel program launcher combination that is used on this partition to launch jobs.
  The syntax of this attribute is ``<scheduler>+<launcher>``.
  A list of the supported `schedulers <#supported-scheduler-backends>`__ and `parallel launchers <#supported-parallel-launchers>`__ can be found at the end of this section.

* ``access``: A list of scheduler options that will be passed to the generated job script for gaining access to that logical partition (default ``[]``).

* ``environs``: A list of environments, with which ReFrame will try to run any regression tests written for this partition (default ``[]``).
  The environment names must be resolved inside the ``environments`` section of the ``site_configuration`` dictionary (see `Environments Configuration <#environments-configuration>`__ for more information).

* ``modules``: A list of modules to be loaded before running a regression test on that partition (default ``[]``).

* ``variables``: A set of environment variables to be set before running a regression test on that partition (default ``{}``).
  Environment variables can be set as follows (notice that both the variable name and its value are strings):

  .. code-block:: python

    'variables': {
        'MYVAR': '3',
        'OTHER': 'foo'
    }

* ``max_jobs``: The maximum number of concurrent regression tests that may be active (not completed) on this partition.
  This option is relevant only when ReFrame executes with the `asynchronous execution policy <running.html#asynchronous-execution-of-regression-checks>`__.

* ``resources``: A set of custom resource specifications and how these can be requested from the partition's scheduler (default ``{}``).

  This variable is a set of key/value pairs with the key being the resource name and the value being a list of options to be passed to the partition's job scheduler.
  The option strings can contain *placeholders* of the form ``{placeholder_name}``.
  These placeholders may be replaced with concrete values by a regression tests through the :attr:`extra_resources` attribute.

  For example, one could define a ``gpu`` resource for a multi-GPU system that uses Slurm as follows:

  .. code-block:: python

    'resources': {
        'gpu': ['--gres=gpu:{num_gpus_per_node}']
    }

  A regression test then may request this resource as follows:

  .. code-block:: python

    self.extra_resources = {'gpu': {'num_gpus_per_node': '8'}}

  And the generated job script will have the following line in its preamble:

  .. code-block:: bash

    #SBATCH --gres=gpu:8

  A resource specification may also start with ``#PREFIX``, in which case ``#PREFIX`` will replace the standard job script prefix of the backend scheduler of this partition.
  This is useful in cases of job schedulers like Slurm, that allow alternative prefixes for certain features.
  An example is the `DataWarp <https://www.cray.com/datawarp>`__ functionality of Slurm which is supported by the ``#DW`` prefix.
  One could then define DataWarp related resources as follows:

  .. code-block:: python

   'resources': {
       'datawarp': [
           '#DW jobdw capacity={capacity} access_mode={mode} type=scratch',
           '#DW stage_out source={out_src} destination={out_dst} type={stage_filetype}'
       ]
   }

  A regression test that wants to make use of that resource, it can set its :attr:`extra_resources` as follows:

  .. code-block:: python

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
   For the `PBS <#supported-scheduler-backends>`__ backend, options accepted in the ``access`` and ``resources`` attributes may either refer to actual ``qsub`` options or be just resources specifications to be passed to the ``-l select`` option.
   The backend assumes a ``qsub`` option, if the options passed in these attributes start with a ``-``.

.. note::
  .. versionchanged:: 2.8
     A new syntax for the ``scheduler`` values was introduced as well as more parallel program launchers.
     The old values for the ``scheduler`` key will continue to be supported.

   .. versionchanged:: 2.9
      Better support for custom job resources.



Supported scheduler backends
============================

ReFrame supports the following job schedulers:


* ``slurm``: Jobs on the configured partition will be launched using `Slurm <https://www.schedmd.com/>`__.
  This scheduler relies on job accounting (``sacct`` command) in order to reliably query the job status.
* ``squeue``: *[new in 2.8.1]*
  Jobs on the configured partition will be launched using `Slurm <https://www.schedmd.com/>`__, but no job accounting is required.
  The job status is obtained using the ``squeue`` command.
  This scheduler is less reliable than the one based on the ``sacct`` command, but the framework does its best to query the job state as reliably as possible.

* ``pbs``: *[new in 2.13]* Jobs on the configured partition will be launched using a `PBS-based <https://en.wikipedia.org/wiki/Portable_Batch_System>`__ scheduler.
* ``local``: Jobs on the configured partition will be launched locally as OS processes.


Supported parallel launchers
============================

ReFrame supports the following parallel job launchers:

* ``srun``: Programs on the configured partition will be launched using a bare ``srun`` command *without* any job allocation options passed to it.
  This launcher may only be used with the ``slurm`` scheduler.
* ``srunalloc``: Programs on the configured partition will be launched using the ``srun`` command *with* job allocation options passed automatically to it.
  This launcher may also be used with the ``local`` scheduler.
* ``alps``: Programs on the configured partition will be launched using the ``aprun`` command.
* ``mpirun``: Programs on the configured partition will be launched using the ``mpirun`` command.
* ``mpiexec``: Programs on the configured partition will be launched using the ``mpiexec`` command.
* ``local``: Programs on the configured partition will be launched as-is without using any parallel program launcher.

There exist also the following aliases for specific combinations of job schedulers and parallel program launchers:

* ``nativeslurm``: This is equivalent to ``slurm+srun``.
* ``local``: This is equivalent to ``local+local``.


Environments Configuration
--------------------------

The environments available for testing in different systems are defined under the ``environments`` key of the top-level ``site_configuration`` dictionary.
The ``environments`` key is associated to a special dictionary that defines scopes for looking up an environment. The ``*`` denotes the global scope and all environments defined there can be used by any system.
Instead of ``*``, you can define scopes for specific systems or specific partitions by using the name of the system or partition.
For example, an entry ``daint`` will define a scope for a system called ``daint``, whereas an entry ``daint:gpu`` will define a scope for a virtual partition named ``gpu`` on the system ``daint``.
When an environment name is used in the ``environs`` list of a system partition (see `Partition Configuration <#partition-configuration>`__), it is first looked up in the entry of that partition, e.g., ``daint:gpu``.
If no such entry exists, it is looked up in the entry of the system, e.g., ``daint``.
If not found there, it is looked up in the global scope denoted by the ``*`` key.
If it cannot be found even there, an error will be issued.
This look up mechanism allows you to redefine an environment for a specific system or partition.
In the following example, we redefine ``PrgEnv-gnu`` for a system named ``foo``, so that whenever ``PrgEnv-gnu`` is used on that system, the module ``openmpi`` will also be loaded and the compiler variables should point to the MPI wrappers.

.. code-block:: python

  'foo': {
      'PrgEnv-gnu': {
          'type': 'ProgEnvironment',
          'modules': ['PrgEnv-gnu', 'openmpi'],
          'cc':  'mpicc',
          'cxx': 'mpicxx',
          'ftn': 'mpif90',
      }
  }

An environment is also defined as a set of key/value pairs with the key being its name and the value being a dictionary of its attributes.
The possible attributes of an environment are the following:

* ``type``: The type of the environment to create. There are two available environment types (note that names are case sensitive):

  * ``'Environment'``: A simple environment.
  * ``'ProgEnvironment'``: A programming environment.

* ``modules``: A list of modules to be loaded when this environment is used (default ``[]``, valid for all environment types)
* ``variables``: A set of variables to be set when this environment is used (default ``{}``, valid for all environment types)
* ``cc``: The C compiler (default ``'cc'``, valid for ``'ProgEnvironment'`` only).
* ``cxx``: The C++ compiler (default ``'CC'``, valid for ``'ProgEnvironment'`` only).
* ``ftn``: The Fortran compiler (default ``'ftn'``, valid for ``'ProgEnvironment'`` only).
* ``cppflags``: The default preprocessor flags (default :class:`None`, valid for ``'ProgEnvironment'`` only).
* ``cflags``: The default C compiler flags (default :class:`None`, valid for ``'ProgEnvironment'`` only).
* ``cxxflags``: The default C++ compiler flags (default :class:`None`, valid for ``'ProgEnvironment'`` only).
* ``fflags``: The default Fortran compiler flags (default :class:`None`, valid for ``'ProgEnvironment'`` only).
* ``ldflags``: The default linker flags (default :class:`None`, valid for ``'ProgEnvironment'`` only).

.. note:: When defining programming environment flags, :class:`None` is treated differently from ``''`` for regression tests that are compiled through a Makefile.
   If a flags variable is not :class:`None` it will be passed to the Makefile, which may affect the compilation process.

System Auto-Detection
---------------------

When the ReFrame is launched, it tries to auto-detect the current system based on its site configuration. The auto-detection process is as follows:

ReFrame first tries to obtain the hostname from ``/etc/xthostname``, which provides the unqualified *machine name* in Cray systems.
If this cannot be found the hostname will be obtained from the standard ``hostname`` command. Having retrieved the hostname, ReFrame goes through all the systems in its configuration and tries to match the hostname against any of the patterns in the ``hostnames`` attribute of `system configuration <#system-configuration>`__.
The detection process stops at the first match found, and the system it belongs to is considered as the current system.
If the system cannot be auto-detected, ReFrame will fail with an error message.
You can override completely the auto-detection process by specifying a system or a system partition with the ``--system`` option (e.g., ``--system daint`` or ``--system daint:gpu``).
