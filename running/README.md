# Running the Framework

Before going into the details of the regression framework's frontend, the simplest way to invoke the regression suite is the following:
```bash
./reframe -r
```
This will run all the available regression tests suitable for the current system for all the supported programming environments.

The regression frontend comprises three phases which are executed in order:
1. Regression check discovery
2. Regression check selection
3. Action on the selected regression checks

We will describe each of these phases in detail in the following. For each phase there is a distinct set of options that control it.
`./reframe -h` will give you a detailed listing of all the options grouped by phase.

## Supported actions
Although this is the last phase the frontend goes through, I list it first since an action is always required.
Otherwise, you will only get the regression's help message.
Currently there are only two available actions:
1. Listing of the selected checks
2. Execution of the selected checks

### Listing of regression checks
To retrieve a listing of the selected checks, you must specify the `-l` or `--list` options.
An example listing of checks is the following:

```bash
./run_regression -l
```
```
Command line: ./run_regression.py -l
Regression paths
================
    Check prefix      : /users/karakasv/Devel/PyRegression
    Check search path : 'checks/'
    Stage dir prefix  : /scratch/snx3000/karakasv/regression/stage/2017-03-03T11:50:08
    Output dir prefix : /apps/daint/UES/jenkins/regression/maintenance/2017-03-03T11:50:08
    Logging dir       : /apps/daint/UES/jenkins/regression/maintenance/logs
List of matched checks
======================
  * amber_gpu_check (Amber parallel GPU check)
        tags: [ scs, production, maintenance ], maintainers: [ VH, VK ]
  * amber_cpu_check (Amber parallel CPU check)
        tags: [ production ], maintainers: [ VH, VK ]
  * cpmd_check (CPMD check (C4H6 metadynamics))
        tags: [ production ], maintainers: [ AJ, LM ]
  * cp2k_cpu_check (CP2K check CPU)
        tags: [ production, scs, maintenance ], maintainers: [ LM, CB ]
  * cp2k_gpu_check (CP2K check GPU)
        tags: [ production, scs, maintenance ], maintainers: [ LM, CB ]
  * namd_cpu_check (NAMD 2.11 check (cpu))
        tags: [ production ], maintainers: [ CB, LM ]
  * namd_gpu_check (NAMD 2.11 check (gpu))
        tags: [ scs, production, maintenance ], maintainers: [ CB, LM ]
  ...
Found 115 check(s).
```
The listing contains the name of the check, its description, the tags associated with it (see [Discovery of regression checks](discovery-of-regression-checks)) and a list of its maintainers.
Note that this listing may also contain checks that are not supported by the current system.
These checks will be just skipped if you try to run them.

### Execution of regression checks

To run the regression checks you should specify the `run` action though the `-r` or `--run` options.
The listing action takes precedence over the execution one, meaning that if you specify both `-l -r`, only the listing action will be performed.
The output of a regression run looks like the following:

```bash
./run_regression.py --notimestamp -c checks/cuda/cuda_checks.py --prefix . -r
```
```
Command line: ./run_regression.py --notimestamp -c checks/cuda/cuda_checks.py --prefix . -r
Regression paths
================
    Check prefix      :
    Check search path : 'checks/cuda/cuda_checks.py'
    Stage dir prefix  : /users/karakasv/Devel/PyRegression/stage/
    Output dir prefix : /users/karakasv/Devel/PyRegression/output/
    Logging dir       : /users/karakasv/Devel/PyRegression/logs
Regression test started by karakasv on dom
===> start date Fri Mar  3 11:48:31 2017
>>>> Running regression on partition: login
Skipping unsupported test cuda_bandwidth_check...
Skipping unsupported test cuda_concurrentkernels_check...
Skipping unsupported test cuda_devicequery_check...
Skipping unsupported test cuda_matrixmulcublas_check...
Skipping unsupported test cuda_simplempi_check...
=============================================================================
>>>> Running regression on partition: mc
Skipping unsupported test cuda_bandwidth_check...
Skipping unsupported test cuda_concurrentkernels_check...
Skipping unsupported test cuda_devicequery_check...
Skipping unsupported test cuda_matrixmulcublas_check...
Skipping unsupported test cuda_simplempi_check...
=============================================================================
>>>> Running regression on partition: gpu
Test: CUDA bandwidthTest compile and run for PrgEnv-cray
=============================================================================
  | Setting up ...                                                   [ OK ]
  | Compiling ...                                                    [ OK ]
  | Submitting job ...                                               [ OK ]
  | Waiting job (id=640085) ...                                      [ OK ]
  | Checking sanity ...                                              [ OK ]
  | Verifying performance ...                                        [ OK ]
  | Cleaning up ...                                                  [ OK ]
| Result: CUDA bandwidthTest compile and run                       [ PASSED ]
Test: CUDA bandwidthTest compile and run for PrgEnv-gnu
=============================================================================
  | Setting up ...                                                   [ OK ]
  | Compiling ...                                                    [ OK ]
  | Submitting job ...                                               [ OK ]
  | Waiting job (id=640086) ...                                      [ OK ]
  | Checking sanity ...                                              [ OK ]
  | Verifying performance ...                                        [ OK ]
  | Cleaning up ...                                                  [ OK ]
| Result: CUDA bandwidthTest compile and run                       [ PASSED ]
...
=============================================================================
Stats for partition: login
  | Ran 0 case(s) of 0 supported check(s) (0 failure(s))
Stats for partition: mc
  | Ran 0 case(s) of 0 supported check(s) (0 failure(s))
Stats for partition: gpu
  | Ran 10 case(s) of 5 supported check(s) (0 failure(s))
===> end date Fri Mar  3 11:50:09 2017
```

The first thing the regression does, even before loading any check, is to try to auto-detect the system it runs on.
From PyRegression [2.1](https://madra.cscs.ch/scs/PyRegression/tags/v2.1) onward, the regression supports heterogeneous clusters consisting of multiple partitions.
The term "partition" here does not refer to a partition in the job scheduler, but rather to a logical partition of the system (e.g., GPU nodes vs. multicore-only nodes vs. KNL nodes etc.).
The way a job gets access to that logical partition can be performed in different ways by the job scheduler (e.g., using partitions, constraints, resource specifications etc.).
For more information on how a new system is configured inside the regression, please have a look [here](#site-configuration).

After the current system is [auto-detected](#system-auto-detection) successfully, the regression will go over all its partitions and try to run all the checks skipping those that are not supported.
Each test is executed repeatedly for all the programming environments it supports and a status line for each individual phase of the execution of the check is printed.
Generally, each regression check passes through a set of phases during its execution, which are summarized below.
We will not get into all the details of each phase, since these are relevant only to the implementors of new regression checks.
There are seven phases a regression check goes through during its execution:
1. *Setup*
   * During this phase the check is set up for the current partition and the current programming environment.
     The check's stage and output directories as well as its job descriptor are set up.
     The job descriptor contains all the necessary information needed to launch the regression check.
2. *Compilation*
   * Here the source code of the check, if any, is compiled. Some tests may not need to compile anything, in which case the status of this phase is always success.
3. *Job submission*
   * At this phase the regression check is launched.
     How the check will be launched depends on the job scheduler that serves the current system partition.
     A system partition (e.g., the login nodes of the system) may only accept local jobs (see [Site configuration](#site-configuration) for more information), in which case a local OS process will be launched for running the check.
     You can also force the regression to run all checks locally using the `--force-local` option.

4. *Job wait*
   * During this phase the previously launched job or process is waited for until it finishes and the job ID or the process ID are reported respectively.
     No check is performed whether the job or process finished gracefully.
     It is responsibility for the check to judge this.
     In practice, this means that this phase should always pass, unless something catastrophic has happened (bug in the framework or malfunctioning job scheduler).

5. *Sanity checking*
   * At this phase the regression check verifies whether it has finished successfully or not.

6. *Performance verification*
   * This phase is only relevant for performance regression checks, in which case the check verifies whether it has met its performance requirements.
     For simple regression checks, this phase is always a success.

7. *Clean up*
   * This phase is responsible for cleaning up the resources of the regression check.
     This includes copying some important files of the check to the output directory (e.g., generated job scripts, standard output/error etc.), removing its temporary stage directory and unloading its environment.


## Discovery of regression checks

When the regression frontend is invoked it tries to locate regression checks in a predefined path.
This path can be retrieved with

```bash
./run_regression.py -l | grep 'Check search path'
```

If the path line is prefixed with `(R)`, every directory in the path will search recursively.
From version 2.1 onward, the default behavior of the regression is to search recursively for checks under the `checks/` directory.

User checks are essentially python source files that provide a special function, which returns the actual regression check instances.
A single source file may provide multiple regression checks.
The front-end loads the python source files and tries to call this special function;
if this function cannot be found, the source file will be ignored.
At the end of this phase the front-end will have instantiated all the checks found in the path.

You can override the default check search path by specifying the `-c` or `--checkpath` options.
The following command will list all the checks found in `checks/apps/`:

```bash
./run_regression.py -c checks/apps/ -l
```

Note that by default the front-end does *not* search recursively into directories specified with the `-c` option.
If you need such behavior you should use the `-R` or `--recurse` options.

The `-c` option completely overrides the default path.
Currently, there is no option to prepend or append to the default regression path.
However, you can build your own check path by specifying multiple times the `-c` option.
The `-c`option accepts also regular files.
This is very useful when you are implementing new regression checks, since it allows you to run only your check:

```bash
./run_regression.py -c /path/to/my/new/check.py -r
```

## Selection of regression checks

After the discovery phase, all the discovered checks will be loaded and ready to be run or listed.
At this phase you can select which regression checks should be finally run or listed.
There are two ways to select regression checks: (a) by programming environment and (b) by tags.

### Selecting checks by programming environment

To select tests by the programming environment, use the `-p` or `--prgenv` options:

```bash
./run_regression.py -p PrgEnv-gnu -l
```

This will select all the checks that support the `PrgEnv-gnu` environment.

You can also specify multiple times the `-p` option, in which case a test will be selected if it support all the programming environments specified in the command line.
For example the following will select all the checks that can run with both `PrgEnv-cray` and `PrgEnv-gnu`:

```bash
./run_regression.py -p PrgEnv-gnu -p PrgEnv-cray -l
```

Note here that specifying the `-p` option will run the selected checks only for the specified programming environments and not for the supported programming environments by the system, which is the default behavior.


### Selecting checks by tags

Each regression check may be associated with a set of tags.
Using the `-t` or `--tag` option you can select the regression checks associated with a specific tag.
For example the following will list all the checks that have a `maintenance` tag:

```bash
./run_regression.py -t maintenance -l
```

Similarly to the `-p` option, you can chain multiple `-t` options together, in which case a regression check will be selected if it is associated with all the tags specified in the command line.
The list of tags associated with a check can be viewed in the listing output when specifying the `-l` option.

Currently, we have two major "official" tags:
(a) `production` for specifying checks to be run daily, while the system is in production and (b) `maintenance` for specifying checks to be run on system maintenance sessions.


### Selecting checks by name

It is possible to select or exclude checks by name through the `--name` or `-n` and `--exclude` or `-x` options.
For example, you can select only the `amber_cpu_check` as follows:
```bash
./run_regression.py -n amber_cpu_check -l
```

```
List of matched checks
======================
  * amber_cpu_check (Amber parallel CPU check)
        tags: [ production ], maintainers: [ VH, VK ]
Found 1 check(s).
```

Similarly, you can exclude this check by passing `-x amber_cpu_check` option.


## Controlling execution of regression checks

There are several options for controlling the execution of the regression checks.
Keep in mind that these options will have effect on all the tests that will run with the current invocation.
They are summarized below:

* `-A ACCOUNT`, `--account ACCOUNT`: Submit regression check jobs using `ACCOUNT`.
   By default the default account for the user launching the regression will be used.
* `-P PART`, `--partition PART`: Submit regression check jobs in partition `PART`.
  By default the default partition for the system the regression is running on will be used.
* `--reservation RES`: Submit regression check jobs in reservation `RES`.
* `--nodelist NODELIST`: Run regression check jobs on the nodes specified in `NODELIST`.
* `--exclude-nodes NODELIST`: Do not run the regression check jobs on any of the nodes specified in `NODELIST`.
* `--job-option OPT`: Pass option `OPT` directly to the back-end job scheduler.
  This option *must* be used with care, since you may break the submission mechanism.
  All the above job submission related options could be expressed with this option.
  For example, the `-n NODELIST` is equivalent to `--job-option='--nodelist=NODELIST'` for a Slurm job scheduler.
  If you pass an option that is already defined by the framework, the framework will *not* explicitly override it; this is up to scheduler.

  All extra options defined from the command line are appended to the automatically generated options in the generated batch script file.
  So if you redefine one of them, e.g., `--output`, it is up to Slurm how to interpret multiple definitions of the same options.
  In this example, Slurm's policy is that later definitions of options override previous ones.
  So, in this case, way you would override the standard output for all the submitted jobs!

* `--force-local`: Forces a local execution of the selected checks. No jobs will be submitted.
* `--skip-sanity-check`: Skips sanity checking phase.
* `--skip-performance-check`: Skips performance verification phase.
* `--relax-performance-check`: (since [2.0.2]((https://madra.cscs.ch/scs/PyRegression/tags/v2.0.2)) This option is similar to the `--skip-performance-check` in that the regression will not fail if the performance of the check is not the expected.
There are, however, two differences:
  1. The performance check *will be* performed and its performance will be logged (see [Logging](#logging)), but the regression will always report success for this phase.
  2. This option affects only those tests that can accept such a relaxed behavior (`strict_check = False` property).
     The application regression checks are such an example.
     This option is useful if you don't want the regression to fail in a production environment, where some checks might experience performance fluctuations due to random factors.
     However, it allows you to still keep a log of their performance for monitoring their behavior in time.
* `--skip-system-check`: Skips the system check and run the selected checks whether they support the current system or not.
  This option is useful when you need to quickly verify if a regression check supports a new system.
* `--skip-prgenv-check`: Skips programming environment check and run the selected checks whether they support the specified programming environments or not.
  This option is useful when you need to quickly verify if a regression check supports another programming environment.
  For example, if you know that a tests supports only `PrgEnv-cray` and you need to check if it works out-of-the-box with `PrgEnv-gnu`, you can test is as follows:

```bash
./run_regression.py -c /path/to/my/check.py -p PrgEnv-gnu --skip-prgenv-check -r
```

## Configuring regression directories

The regression framework uses three basic directories during the execution of tests:
* The stage directory
  * Each regression check is executed in a "sandbox";
    all of its resources (source files, resources) are copied over to a stage directory and executed from there.
    This will also be the working directory for the check.
* The output directory
  * After a regression check finishes execution some important files will be copied from the stage directory to the output directory.
    By default these are the standard output, standard error and the generated job script file.
    The regression may also specify to keep additional files.
* The log directory (since [2.0.2](https://madra.cscs.ch/scs/PyRegression/tags/v2.0.2))
  * This is where the log files of the individual performance checks are placed (see [Logging](#logging) for more information)

By default all these directories are placed under a common prefix, which defaults to `.`.
The rest of the directories are organized as follows:

* Stage directory: `${prefix}/stage/<timestamp>`
* Output directory: `${prefix}/output/<timestamp>`
* Log directory (since [2.0.2](https://madra.cscs.ch/scs/PyRegression/tags/v2.0.2)): `${prefix}/logs`

A timestamp directory will be appended to the stage and output directories, unless you specify the `--notimestamp` option.
The default format of the timestamp is `yyyy-mm-ddThh:mm:ss`.
Since PyRegression [2.1](https://madra.cscs.ch/scs/PyRegression/tags/v2.1), you can change the timestamp format using the `--timefmt` option, which accepts a `strftime()` compatible string.


You can override either the default global prefix or any of the default individual regression directories using the corresponding options.

* `--prefix DIR`: set regression's prefix to `DIR`.
* `--output DIR`: set regression's output directory to `DIR`.
* `--stage DIR`: set regression's stage directory to `DIR`.
* `--logdir DIR`: (since [2.0.2](https://madra.cscs.ch/scs/PyRegression/tags/v2.1)) set regression's log directory to `DIR`.

The stage and output directories are created only when you run a regression check.
However you can view the directories that will be created even when you do a listing of the available checks with the `-l` option.
This is useful if you want to check the directories that regression will create.

```bash
./run_regression.py --prefix /foo -l
```
```
Command line: ./run_regression.py --prefix /foo -t foo -l
Regression paths
================
    Check prefix      : /users/karakasv/Devel/PyRegression
(R) Check search path : 'checks/'
    Stage dir prefix  : /foo/stage/2017-03-09T15:01:11
    Output dir prefix : /foo/output/2017-03-09T15:01:11
    Logging dir       : /foo/logs
List of matched checks
======================
Found 0 check(s).
```

You can also define different default regression directories per system by specifying them in the [site configuration](#site-configuration) settings file.
However, the command line options take always precedence over any default directory.

## Logging

Since PyRegression [2.1](https://madra.cscs.ch/scs/PyRegression/tags/v2.1), it is supported a very simple logging of the performance values of the performance checks.
For each performance check, a log file of the form `<check-name>.log` is created under the regression's [log directory](#configuring-regression-directories) for logging the check's performance.
Regression always appends to this file, so you can keep a history of the performance of the check.
This file looks like the following:
```
[2017-02-27T16:32:48] regression.checks.namd_cpu_check: INFO: submitted job (id=637895)
[2017-02-27T16:42:42] regression.checks.namd_cpu_check: INFO: value: 4.440446666666666, reference: (1.37, None, 0.15)
```

The interpretation of the performance values depend on the individual checks. In the above check, for example, the performance value refers to days/ns.
The reference value is a three-element tuple of the form `(<reference>, <low-threshold>, <high-threshold>)`, where the `low-threshold` and `high-threshold` are the acceptable tolerance thresholds expressed in percentages. For example, the performance check shown above has a reference value of 1.37 days/ns with a maximum high tolerance of 15%.
There is no low tolerance, since lower values denote higher performance.

## Site configuration

PyRegression [2.1](https://madra.cscs.ch/scs/PyRegression/tags/v2.1) introduces an easy way for configuring the regression suite for new sites.
In the past this was only possible by hacking inside the regression internals.
Now the configuration of systems and programming environments is performed by a special Python dictionary called `site_configuration` defined in `<install-dir>/regression/settings.py`. Here is how the configuration for Daint looks like:

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
                    },
                    'gpu' : {
                        'scheduler' : 'nativeslurm',
                        'modules'   : [ 'daint-gpu' ],
                        'access'    : [ '--constraint=gpu' ],
                        'environs'  : [ 'PrgEnv-cray', 'PrgEnv-gnu',
                                        'PrgEnv-intel', 'PrgEnv-pgi' ],
                        'descr'     : 'Hybrid nodes (Haswell/P100)',
                    },
                    'mc' : {
                        'scheduler' : 'nativeslurm',
                        'modules'   : [ 'daint-mc' ],
                        'access'    : [ '--constraint=mc' ],
                        'environs'  : [ 'PrgEnv-cray', 'PrgEnv-gnu',
                                        'PrgEnv-intel', 'PrgEnv-pgi' ],
                        'descr'     : 'Multicore nodes (Broadwell)',
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

### System configuration
The list of supported systems is defined as a set of key/value pairs under the global configuration key `systems`.
Each system is a key/value pair, with the key being the name of the system and the value being another set of key/value pairs defining its attributes.
The valid attributes of a system are the following:

* `descr`: A detailed description of the system (default is the system name).
* `hostnames`: This is a list of hostname patterns that will be used by the regression when it tries to [auto-detect](#system-auto-detection) the system it runs on (default `[]`).
* `prefix`: Default regression prefix for this system (default `.`).
* `stagedir`: Default stage directory for this system (default `None`).
* `outputdir`: Default output directory for this system (default `None`).
* `logdir`: Default log directory for this system (default `None`).
* `partitions`: A set of key/value pairs defining the partitions of this system and their properties (default `{}`).
  See [next section](#partition-configuration) on how to define system partitions.


### Partition configuration
From the regression's point of view each system consists of a set of logical partitions.
These partitions need not necessarily correspond to real scheduler partitions.
For example, Daint comprises three logical partitions: the login nodes (named `login`), the hybrid nodes (named `gpu`) and the multicore nodes (named `mc`), but these do not correspond to actual Slurm partitions.

The partitions of a system are defined similarly as a set of key/value pairs with the key being the partition name and the value being another set of key/value pairs defining the partition's attributes.
The available partition attributes are the following:
* `descr`: A detailed description of the partition (default is the partition name).
* `scheduler`: The job scheduler to use for launching jobs on this partition.
   Available values are the following:
   * `local`: Jobs on this partition will be launched locally as OS processes.
   When a job is launched locally, the regression will create a wrapper shell script for running the check on the current node.
   This is default scheduler if none is specified.
   * `nativeslurm`: Jobs on this partition will be launched using Slurm and the `srun` command for creating MPI processes.
   * `slurm+alps`: Jobs on this partition will be launched using Slurm and the `aprun` command for creating MPI processes (this scheduler is not thoroughly tested, due to lack of support on CSCS' systems).
* `access`: A list of scheduler options that will be passed to the generated job script for gaining access to that logical partition (default `[]`).
  You can see that for the Daint logical partitions we use the `--constraint` feature of Slurm to get access, since the logical partitions do not actually correspond to Slurm partitions.
* `environs`: A list of environments that the regression will try to use for running each check (default `[]`).
  The environment names must be resolved inside the `environments` section of the `site_configuration` dictionary (see [Environment configuration](#environment-configuration) for more information).
* `modules`: A list of modules to be loaded each time before running a regression check on that partition (default `[]`).
* `variables`: A set of environment variables to be set each time before running a regression check on that partition (default `{}`).
  This is how you can set environment variables (notice that both the variable name and its value are strings):
```python
'variables' : {
    'MYVAR' : '3',
    'OTHER' : 'foo'
}
```

* `resources`: A set of custom resource specifications and how these can be requested from the partition's scheduler (default `{}`).
  This variable is a set of key/value pairs with the key being the resource name and the value being a list of options to be passed to the partition's job scheduler.
  The option strings can contain "references" to the resource being required using the syntax `{resource_name}`.
  In such cases, the `{resource_name}` will be replaced by the value of that resource defined in the regression check that is being run.
  For example, here is how the resources are specified on Kesch, a system with 16 GPUs per node, for requesting a number of GPUs:
```python
'resources' : {
    'num_gpus_per_node' : [
        '--gres=gpu:{num_gpus_per_node}'
    ]
}
```
When the regression runs a check that defines `num_gpus_per_node = 8`, the generated job script for that check will have in its preamble the following line:
```bash
#SBATCH --gres=gpu:8
```
therefore requesting from the resource scheduler to allocate it 8 GPUs.


### Environment configuration

The environments available for testing to the different systems are defined under the `environments` key of the top-level `site_configuration` dictionary.
The `environments` of the `site_configuration` is a special dictionary that defines scopes for looking up an environment.
The `*` denotes the global scope and all environments defined there can be used by any system.
You can define a dictionary only for a specific system by placing it under an entry keyed with the name of that system, e.g., `daint`, or even for a specific partition, e.g., `daint:gpu`.
When an environment is used in the `environs` attribute of a system partition (see [Partition configuration](#partition-configuration)), it is looked up first in the entry of that partition, e.g., `daint:gpu`.
If no such entry exists, it is looked up in the entry of the system, e.g., `daint`.
If not found there yet, it is looked up in the global scope denoted by the `*` key.
If it cannot be found even there, then an error will be issued.
This look up mechanism allows you to redefine an environment for a specific system or partition.
In the example shown above, the `PrgEnv-gnu` is redefined for the system `kesch` (any partition), so as to use different compiler wrappers.

An environment is defined as key/value pair with the key being its name and the value being a dictionary of its attributes.
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


### System auto-detection
When the regression is launched, it tries to auto-detect the system it runs on based on its site configuration.
The auto-detection process is as follows:

The regression first tries to obtain the hostname from `/etc/xthostname`, which provides the unqualified "machine name" in Cray systems.
If this cannot be found the hostname will be obtained from the standard `hostname` command.
Having retrieved the hostname, the regression goes through all the systems in its configuration and tries to match the hostname against any of the patterns in the `hostnames` attribute.
The detection process stops at the first match found, and the system it belongs to is considered as the current system.
If the system cannot be auto-detected, regression will fail with an error message.
You can override completely the auto-detection process by specifying a system or a system partition with the `--system` option (e.g., `--system daint` or `--system daint:gpu`).


## Examples of usage

1. Run all tests with the `production` tag and place the output of the regression in your home directory:
```bash
./run_regression.py -o $HOME/regression/output -t production -r
```

2. List all tests with the `maintenance` and `slurm` tags:
```bash
./run_regression.py -t maintenance -t slurm -l
```

2. Run all the maintenance checks on the `syscheckout` reservation:
```bash
./run_regression.py -t maintenance --reservation=syscheckout -r
```

2. List all production tests supporting `PrgEnv-gnu` and having the `production` tag:
```bash
./run_regression.py -p PrgEnv-gnu -t production -l
```

3. Run all the checks from a specific check file and relocate both output and stage directories under your current working directory without using timestamps:
```bash
./run_regression.py --prefix . --no-timestamp -c /path/to/my/check.py -r
```
This is a useful setup while developing new regression checks, since you don't want to "contaminate" the default stage and output locations or end up with lots of directories with different timestamps.

4. Run a specific check on a new system that is not officially supported by the check:
```bash
./run_regression.py -c /path/to/my/check.py --skip-system-check -r
```

5. Run a specific check on a programming environment (e.g., `PrgEnv-pgi`) that is not officially supported by the check:
```bash
./run_regression.py -c /path/to/my/check.py -p PrgEnv-pgi --skip-prgenv-check -r
```

# Limitations

* Currently, there is no way for a check to specify that it needs to run on the full extent of a partition or a reservation.


# Understanding the framework internals

A developer's guide will be put in place [here](framework).

You can find the minutes of developers' meetings [here](pyregression-dev-meetings).

# Further reading

* *PyRegression 2.0 &ndash; User training*, 7.12.2016 @ CSCS [ [PyRegression tutorial](/uploads/dac1d80ec1dd838e0c7d03c4f147efca/PyRegression2.0_tutorial_cscs.pdf) [regression for Daint MC part](/uploads/532fb57389e06b35ccdd3747027c26b4/regression_mc_part.pdf) ].
