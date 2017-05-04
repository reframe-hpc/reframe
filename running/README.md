# Running the Framework

Before going into any details about the framework front-end and command-line interfaces, the simplest way to invoke ReFrame is the following:
```bash
./reframe -c /path/to/checks -R --run
```
This will search recursively for test files in `/path/to/checks` and will start running them on the current system.

ReFrame's front-end goes through three phases:
* test discovery,
*  test selection and
* action.

In the following, we will elaborate on these phases and the key command-line options controlling them.
A detailed listing of all the options grouped by phase is given by `reframe -h`.

## System auto-detection
When the regression is launched, it tries to auto-detect the system it runs on based on its site configuration.
The auto-detection process is as follows:

The regression first tries to obtain the hostname from `/etc/xthostname`, which provides the unqualified *machine name* in Cray systems.
If this cannot be found the hostname will be obtained from the standard `hostname` command.
Having retrieved the hostname, the regression goes through all the systems in its configuration and tries to match the hostname against any of the patterns in the `hostnames` attribute (see the [Configure your site section](/configure) for further details).
The detection process stops at the first match found, and the system it belongs to is considered as the current system.
If the system cannot be auto-detected, regression will fail with an error message.
You can override completely the auto-detection process by specifying a system or a system partition with the `--system` option (e.g., `--system daint` or `--system daint:gpu`).


## Supported actions
Even though the actions is the last phase the front-end goes through, it is listed first since an action is always required.
Otherwise, you will only get the regression's help message.
Currently there are only two available actions:
1. Listing of the selected checks
2. Execution of the selected checks


### Listing of regression checks
To retrieve a listing of the selected checks, you must specify the `-l` or `--list` options.
An example listing of checks is the following:

```bash
reframe -l
```

The ouput looks like:
```
Command line: reframe -l
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
The listing contains the name of the check, its description, the tags associated with it (see [Discovery of regression checks](#discovery-of-regression-checks)) and a list of its maintainers.
Note that this listing may also contain checks that are not supported by the current system.
These checks will be just skipped if you try to run them.

### Execution of regression checks

To run the regression checks you should specify the `run` action though the `-r` or `--run` options.
The listing action takes precedence over the execution one, meaning that if you specify both `-l -r`, only the listing action will be performed.
```bash
reframe --notimestamp -c checks/cuda/cuda_checks.py --prefix . -r
```

The output of the regression run looks like the following:
```
Command line: reframe --notimestamp -c checks/cuda/cuda_checks.py --prefix . -r
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

## Discovery of regression checks

When the regression frontend is invoked it tries to locate regression checks in a predefined path.
This path can be retrieved with

```bash
reframe -l | grep 'Check search path'
```

If the path line is prefixed with `(R)`, every directory in the path will search recursively.

User checks are essentially python source files that provide a special function, which returns the actual regression check instances.
A single source file may provide multiple regression checks.
The front-end loads the python source files and tries to call this special function;
if this function cannot be found, the source file will be ignored.
At the end of this phase the front-end will have instantiated all the checks found in the path.

You can override the default check search path by specifying the `-c` or `--checkpath` options.
The following command will list all the checks found in `checks/apps/`:

```bash
reframe -c checks/apps/ -l
```

Note that by default the front-end does *not* search recursively into directories specified with the `-c` option.
If you need such behavior you should use the `-R` or `--recurse` options.

The `-c` option completely overrides the default path.
Currently, there is no option to prepend or append to the default regression path.
However, you can build your own check path by specifying multiple times the `-c` option.
The `-c`option accepts also regular files.
This is very useful when you are implementing new regression checks, since it allows you to run only your check:

```bash
reframe -c /path/to/my/new/check.py -r
```

## Selection of regression checks

After the discovery phase, all the discovered checks will be loaded and ready to be run or listed.
At this phase you can select which regression checks should be finally run or listed.
There are two ways to select regression checks: (a) by programming environment and (b) by tags.

### Selecting checks by programming environment

To select tests by the programming environment, use the `-p` or `--prgenv` options:

```bash
reframe -p PrgEnv-gnu -l
```

This will select all the checks that support the `PrgEnv-gnu` environment.

You can also specify multiple times the `-p` option, in which case a test will be selected if it support all the programming environments specified in the command line.
For example the following will select all the checks that can run with both `PrgEnv-cray` and `PrgEnv-gnu`:

```bash
reframe -p PrgEnv-gnu -p PrgEnv-cray -l
```

Note here that specifying the `-p` option will run the selected checks only for the specified programming environments and not for the supported programming environments by the system, which is the default behavior.


### Selecting checks by tags

Each regression check may be associated with a set of tags.
Using the `-t` or `--tag` option you can select the regression checks associated with a specific tag.
For example the following will list all the checks that have a `maintenance` tag:

```bash
reframe -t maintenance -l
```

Similarly to the `-p` option, you can chain multiple `-t` options together, in which case a regression check will be selected if it is associated with all the tags specified in the command line.
The list of tags associated with a check can be viewed in the listing output when specifying the `-l` option.

Currently, we have two major "official" tags:
(a) `production` for specifying checks to be run daily, while the system is in production and (b) `maintenance` for specifying checks to be run on system maintenance sessions.


### Selecting checks by name

It is possible to select or exclude checks by name through the `--name` or `-n` and `--exclude` or `-x` options.
For example, you can select only the `amber_cpu_check` as follows:
```bash
reframe -n amber_cpu_check -l
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
* `--relax-performance-check`: This option is similar to the `--skip-performance-check` in that the regression will not fail if the performance of the check is not the expected.
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
reframe -c /path/to/my/check.py -p PrgEnv-gnu --skip-prgenv-check -r
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
* The log directory
  * This is where the log files of the individual performance checks are placed (see [Logging](#logging) for more information)

By default all these directories are placed under a common prefix, which defaults to `.`.
The rest of the directories are organized as follows:

* Stage directory: `${prefix}/stage/<timestamp>`
* Output directory: `${prefix}/output/<timestamp>`
* Log directory: `${prefix}/logs`

A timestamp directory will be appended to the stage and output directories, unless you specify the `--notimestamp` option.
The default format of the timestamp is `yyyy-mm-ddThh:mm:ss`.
You can change the timestamp format using the `--timefmt` option, which accepts a `strftime()` compatible string.


You can override either the default global prefix or any of the default individual regression directories using the corresponding options.

* `--prefix DIR`: set regression's prefix to `DIR`.
* `--output DIR`: set regression's output directory to `DIR`.
* `--stage DIR`: set regression's stage directory to `DIR`.
* `--logdir DIR`: set regression's log directory to `DIR`.

The stage and output directories are created only when you run a regression check.
However you can view the directories that will be created even when you do a listing of the available checks with the `-l` option.
This is useful if you want to check the directories that regression will create.

```bash
reframe --prefix /foo -l
```
```
Command line: reframe --prefix /foo -t foo -l
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

You can also define different default regression directories per system by specifying them in the [site configuration](/configure/#new-system-configuration) settings file.
However, the command line options take always precedence over any default directory.

## Logging

ReFrame supports a very simple logging of the performance values of the performance checks.
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


## Examples of usage

1. Run all tests with the `production` tag and place the output of the regression in your home directory:
```bash
reframe -o $HOME/regression/output -t production -r
```

2. List all tests with the `maintenance` and `slurm` tags:
```bash
reframe -t maintenance -t slurm -l
```

2. Run all the maintenance checks on the `syscheckout` reservation:
```bash
reframe -t maintenance --reservation=syscheckout -r
```

2. List all production tests supporting `PrgEnv-gnu` and having the `production` tag:
```bash
reframe -p PrgEnv-gnu -t production -l
```

3. Run all the checks from a specific check file and relocate both output and stage directories under your current working directory without using timestamps:
```bash
reframe --prefix . --no-timestamp -c /path/to/my/check.py -r
```
This is a useful setup while developing new regression checks, since you don't want to "contaminate" the default stage and output locations or end up with lots of directories with different timestamps.

4. Run a specific check on a new system that is not officially supported by the check:
```bash
reframe -c /path/to/my/check.py --skip-system-check -r
```

5. Run a specific check on a programming environment (e.g., `PrgEnv-pgi`) that is not officially supported by the check:
```bash
reframe -c /path/to/my/check.py -p PrgEnv-pgi --skip-prgenv-check -r
```
