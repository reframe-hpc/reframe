ReFrame provides an easy and flexible way to configure new systems and new programming environments.
By default, it ships with two systems configured already: a generic local system, which is also used in the front-end unit tests, and Cray's Swan TDS for demonstration purposes.

As soon as a new system with its programming environments is configured, adapting an existing regression test could be as easy as just adding the system's name in the `valid_systems` list and its associated programming environments in the `valid_prog_environs` list.

# The configuration file

The configuration of systems and programming environments is performed by a special Python dictionary called `site_configuration` defined inside the file `<install-dir>/reframe/settings.py`.

The `site_configuration` dictionary should define two entries,`systems` and `environments`.
The former defines the available systems to the regression tests and the latter the available programming environments.

An example of how the configuration for [Piz Daint](http://www.cscs.ch/computers/piz_daint/index.html) at CSCS looks like:

```python
site_configuration = ReadOnlyField({
    'systems' : {
        'daint' : {
            'descr' : 'Piz Daint',
            'hostnames' : [ 'daint' ],
            'outputdir' : '$APPS/UES/jenkins/regression/maintenance',
            'logdir'    : '$APPS/UES/jenkins/regression/maintenance/logs',
            'stagedir'  : '$SCRATCH/regression/stage',
            'partitions' : {
                'login' : {
                    'scheduler' : 'local',
                    'modules'   : [],
                    'access'    : [],
                    'environs'  : [ 'PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi' ],
                    'descr'     : 'Login nodes'
                    'max_jobs'  : 4
                },
                'gpu' : {
                    'scheduler' : 'nativeslurm',
                    'modules'   : [ 'daint-gpu' ],
                    'access'    : [ '--constraint=gpu' ],
                    'environs'  : [ 'PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi' ],
                    'descr'     : 'Hybrid nodes (Haswell/P100)',
                    'max_jobs'  : 100
                },
                'mc' : {
                    'scheduler' : 'nativeslurm',
                    'modules'   : [ 'daint-mc' ],
                    'access'    : [ '--constraint=mc' ],
                    'environs'  : [ 'PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel', 'PrgEnv-pgi' ],
                    'descr'     : 'Multicore nodes (Broadwell)',
                    'max_jobs'  : 100
                }
            }
        }
    },

    'environments' : {
        'kesch' : {
            'PrgEnv-gnu' : {
                'type' : 'ProgEnvironment',
                'modules' : [ 'PrgEnv-gnu' ],
                'cc'      : 'mpicc',
                'cxx'     : 'mpicxx',
                'ftn'     : 'mpif90',
            },
        },
        '*' : {
            'PrgEnv-cray' : {
                'type' : 'ProgEnvironment',
                'modules' : [ 'PrgEnv-cray' ],
            },
            'PrgEnv-gnu' : {
                'type' : 'ProgEnvironment',
                'modules' : [ 'PrgEnv-gnu' ],
            },
            'PrgEnv-intel' : {
                'type' : 'ProgEnvironment',
                'modules' : [ 'PrgEnv-intel' ],
            },
            'PrgEnv-pgi' : {
                'type' : 'ProgEnvironment',
                'modules' : [ 'PrgEnv-pgi' ],
            }
        }
    }
})
```

# System configuration
The list of supported systems is defined as a set of key/value pairs under the global configuration key `systems`.
Each system is a key/value pair, with the key being the name of the system and the value being another set of key/value pairs defining its attributes.
The valid attributes of a system are the following:

* `descr`: A detailed description of the system (default is the system name).
* `hostnames`: This is a list of hostname patterns that will be used by ReFrame when it tries to [auto-detect](#system-auto-detection) the system it runs on (default `[]`).
* `prefix`: Default regression prefix for this system (default `.`).
* `stagedir`: Default stage directory for this system (default `None`).
* `outputdir`: Default output directory for this system (default `None`).
* `logdir`: Default log directory for this system (default `None`).
* `partitions`: A set of key/value pairs defining the partitions of this system and their properties (default `{}`).
  See [Partition configuration section](#partition-configuration) on how to define system partitions.


# Partition configuration
From the ReFrame's point of view each system consists of a set of logical partitions.
These partitions need not necessarily correspond to real scheduler partitions.
For example, CSCS' Piz Daint comprises three logical partitions: the login nodes (named `login`), the hybrid nodes (named `gpu`) and the multicore nodes (named `mc`), but these do not correspond to actual Slurm partitions.

The partitions of a system are defined similarly as a set of key/value pairs with the key being the partition name and the value being another set of key/value pairs defining the partition's attributes.
The available partition attributes are the following:
* `descr`: A detailed description of the partition (default is the partition name).
* `scheduler`: The job scheduler to use for launching jobs on this partition.
   Available values are the following:
   * `local`: Jobs on this partition will be launched locally as OS processes.
   When a job is launched locally, ReFrame will create a wrapper shell script for running the check on the current node.
   This is default scheduler if none is specified.
   * `nativeslurm`: Jobs on this partition will be launched using Slurm and the `srun` command for creating MPI processes.
   * `slurm+alps`: Jobs on this partition will be launched using Slurm and the `aprun` command for creating MPI processes (this scheduler is not thoroughly tested, due to lack of support on CSCS' systems).
* `access`: A list of scheduler options that will be passed to the generated job script for gaining access to that logical partition (default `[]`).
  You can see that for the Piz Daint logical partitions we use the `--constraint` feature of Slurm to get access, since the logical partitions do not actually correspond to Slurm partitions.
* `environs`: A list of environments that ReFrame will try to use for running each regression test (default `[]`).
  The environment names must be resolved inside the `environments` section of the `site_configuration` dictionary (see [Environment configuration](#environment-configuration) for more information).
* `modules`: A list of modules to be loaded each time before running a regression test on that partition (default `[]`).
* `variables`: A set of environment variables to be set each time before running a regression test on that partition (default `{}`).
  This is how you can set environment variables (notice that both the variable name and its value are strings):

```python
'variables' : {
    'MYVAR' : '3',
    'OTHER' : 'foo'
}
```
* `max_jobs`: (new in [2.4](https://github.com/eth-cscs/reframe/releases/tag/v2.4)) The maximum number of concurrent regression checks that may be active (not completed) on this partition.
   This option is relevant only when ReFrame executes with the [asynchronous execution policy](/running#asynchronous-execution-of-regression-checks).

* `resources`: A set of custom resource specifications and how these can be requested from the partition's scheduler (default `{}`).
  This variable is a set of key/value pairs with the key being the resource name and the value being a list of options to be passed to the partition's job scheduler.
  The option strings can contain "references" to the resource being required using the syntax `{resource_name}`.
  In such cases, the `{resource_name}` will be replaced by the value of that resource defined in the regression test that is being run.
  For example, here is how the resources are specified on [Kesch](http://www.cscs.ch/computers/kesch_escha_meteoswiss/index.html), a system with 16 GPUs per node, for requesting a number of GPUs:

```python
'resources' : {
    'num_gpus_per_node' : [
        '--gres=gpu:{num_gpus_per_node}'
    ]
}
```
When ReFrame runs a test that defines `self.num_gpus_per_node = 8`, the generated job script for that test will have in its preamble the following line:
```bash
#SBATCH --gres=gpu:8
```


# Environment configuration

The environments available to different systems are defined under the `environments` key of the top-level `site_configuration` dictionary.
The `environments` of the `site_configuration` is a special dictionary that defines scopes for looking up an environment.
The `*` denotes the global scope and all environments defined there can be used by any system.
You can define a dictionary only for a specific system by placing it under an entry keyed with the name of that system, e.g., `daint`, or even for a specific partition, e.g., `daint:gpu`.
When an environment is used in the `environs` attribute of a system partition (see [Partition configuration](#partition-configuration)), it is looked up first in the entry of that partition, e.g., `daint:gpu`.
If no such entry exists, it is looked up in the entry of the system, e.g., `daint`.
If not found there yet, it is looked up in the global scope denoted by the `*` key.
If it cannot be found even there, then an error will be issued.
This look up mechanism allows you to redefine an environment for a specific system or partition.
In the example shown above, the `PrgEnv-gnu` is redefined for the system `kesch` (any partition), so as to use different compiler wrappers.

An environment is defined as a set of key/value pairs with the key being its name and the value being a dictionary of its attributes.
The possible attributes of an environment are the following:
* `type`: The type of the environment to create. There are two available environment types (note that names are case sensitive):
  * `Environment`: A simple environment.
  * `ProgEnvironment`: A programming environment.
* `modules`: A list of modules to be loaded when this environment is loaded (default `[]`, valid for all types)
* `variables`: A set of variables to be set when this environment is loaded (default `{}`, valid for all types)
* `cc`: The C compiler (default `cc`, valid for `ProgEnvironment` only).
* `cxx`: The C++ compiler (default `CC`, valid for `ProgEnvironment` only).
* `ftn`: The Fortran compiler (default `ftn`, valid for `ProgEnvironment` only).
* `cppflags`: The default preprocessor flags (default `''`, valid for `ProgEnvironment` only).
* `cflags`: The default C compiler flags (default `''`, valid for `ProgEnvironment` only).
* `cxxflags`: The default C++ compiler flags (default `''`, valid for `ProgEnvironment` only).
* `fflags`: The default Fortran compiler flags (default `''`, valid for `ProgEnvironment` only).
* `ldflags`: The default linker flags (default `''`, valid for `ProgEnvironment` only).
