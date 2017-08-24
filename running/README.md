# Running the Framework

Before going into any details about the framework front-end and command-line interfaces, the simplest way to invoke ReFrame is the following:
```bash
./bin/reframe -c /path/to/checks -R --run
```
This will search recursively for test files in `/path/to/checks` and will start running them on the current system.

ReFrame's front-end goes through three phases:
* test discovery,
*  test selection and
* action.

In the following, we will elaborate on these phases and the key command-line options controlling them.
A detailed listing of all the options grouped by phase is given by `./bin/reframe -h`.

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
./bin/reframe -l
```

The ouput looks like:
```
Command line: ./bin/reframe -l
Reframe version: 2.4
Launched by user: karakasv
Launched on host: daint101
Reframe paths
=============
    Check prefix      : /users/karakasv/Devel/reframe
(R) Check search path : 'checks/'
    Stage dir prefix  : /scratch/snx3000/karakasv/regression/stage/2017-06-27T23:48:26
    Output dir prefix : /apps/daint/UES/jenkins/regression/maintenance/2017-06-27T23:48:26
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
./bin/reframe -c checks/cuda/cuda_checks.py -r
```

The output of the regression run looks like the following:
```
Command line: ./bin/reframe -c checks/cuda/cuda_checks.py -r
Reframe version: 2.5
Launched by user: karakasv
Launched on host: daint103
Reframe paths
=============
    Check prefix      :
    Check search path : 'checks/cuda/cuda_checks.py'
    Stage dir prefix  : /users/karakasv/Devel/reframe/stage/
    Output dir prefix : /users/karakasv/Devel/reframe/output/
    Logging dir       : /users/karakasv/Devel/reframe/logs
[==========] Running 5 check(s)
[==========] Started on Thu Aug 24 15:30:30 2017

[----------] started processing cuda_bandwidth_check (CUDA bandwidthTest compile and run)
[   SKIP   ] skipping daint:login
[ RUN      ] cuda_bandwidth_check on daint:gpu using PrgEnv-cray
[       OK ] cuda_bandwidth_check on daint:gpu using PrgEnv-cray
[ RUN      ] cuda_bandwidth_check on daint:gpu using PrgEnv-gnu
[       OK ] cuda_bandwidth_check on daint:gpu using PrgEnv-gnu
[   SKIP   ] skipping PrgEnv-intel for daint:gpu
[   SKIP   ] skipping PrgEnv-pgi for daint:gpu
[   SKIP   ] skipping daint:mc
[----------] finished processing cuda_bandwidth_check (CUDA bandwidthTest compile and run)

[----------] started processing cuda_concurrentkernels_check (Use of streams for concurrent execution)
[   SKIP   ] skipping daint:login
[ RUN      ] cuda_concurrentkernels_check on daint:gpu using PrgEnv-cray
[       OK ] cuda_concurrentkernels_check on daint:gpu using PrgEnv-cray
[ RUN      ] cuda_concurrentkernels_check on daint:gpu using PrgEnv-gnu
[       OK ] cuda_concurrentkernels_check on daint:gpu using PrgEnv-gnu
[   SKIP   ] skipping PrgEnv-intel for daint:gpu
[   SKIP   ] skipping PrgEnv-pgi for daint:gpu
[   SKIP   ] skipping daint:mc
[----------] finished processing cuda_concurrentkernels_check (Use of streams for concurrent execution)

[----------] started processing cuda_devicequery_check (Queries the properties of the CUDA devices)
[   SKIP   ] skipping daint:login
[ RUN      ] cuda_devicequery_check on daint:gpu using PrgEnv-cray
[       OK ] cuda_devicequery_check on daint:gpu using PrgEnv-cray
[ RUN      ] cuda_devicequery_check on daint:gpu using PrgEnv-gnu
[       OK ] cuda_devicequery_check on daint:gpu using PrgEnv-gnu
[   SKIP   ] skipping PrgEnv-intel for daint:gpu
[   SKIP   ] skipping PrgEnv-pgi for daint:gpu
[   SKIP   ] skipping daint:mc
[----------] finished processing cuda_devicequery_check (Queries the properties of the CUDA devices)

[----------] started processing cuda_matrixmulcublas_check (Implements matrix multiplication using CUBLAS)
[   SKIP   ] skipping daint:login
[ RUN      ] cuda_matrixmulcublas_check on daint:gpu using PrgEnv-cray
[       OK ] cuda_matrixmulcublas_check on daint:gpu using PrgEnv-cray
[ RUN      ] cuda_matrixmulcublas_check on daint:gpu using PrgEnv-gnu
[       OK ] cuda_matrixmulcublas_check on daint:gpu using PrgEnv-gnu
[   SKIP   ] skipping PrgEnv-intel for daint:gpu
[   SKIP   ] skipping PrgEnv-pgi for daint:gpu
[   SKIP   ] skipping daint:mc
[----------] finished processing cuda_matrixmulcublas_check (Implements matrix multiplication using CUBLAS)

[----------] started processing cuda_simplempi_check (Simple example demonstrating how to use MPI with CUDA)
[   SKIP   ] skipping daint:login
[ RUN      ] cuda_simplempi_check on daint:gpu using PrgEnv-cray
[       OK ] cuda_simplempi_check on daint:gpu using PrgEnv-cray
[ RUN      ] cuda_simplempi_check on daint:gpu using PrgEnv-gnu
[       OK ] cuda_simplempi_check on daint:gpu using PrgEnv-gnu
[   SKIP   ] skipping PrgEnv-intel for daint:gpu
[   SKIP   ] skipping PrgEnv-pgi for daint:gpu
[   SKIP   ] skipping daint:mc
[----------] finished processing cuda_simplempi_check (Simple example demonstrating how to use MPI with CUDA)

[  PASSED  ] Ran 10 test case(s) from 5 check(s) (0 failure(s))
[==========] Finished on Thu Aug 24 17:36:55 2017
```

## Discovery of regression checks

When the regression frontend is invoked it tries to locate regression checks in a predefined path.
This path can be retrieved with

```bash
./bin/reframe -l | grep 'Check search path'
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
./bin/reframe -c checks/apps/ -l
```

Note that by default the front-end does *not* search recursively into directories specified with the `-c` option.
If you need such behavior you should use the `-R` or `--recurse` options.

The `-c` option completely overrides the default path.
Currently, there is no option to prepend or append to the default regression path.
However, you can build your own check path by specifying multiple times the `-c` option.
The `-c`option accepts also regular files.
This is very useful when you are implementing new regression checks, since it allows you to run only your check:

```bash
./bin/reframe -c /path/to/my/new/check.py -r
```

## Selection of regression checks

After the discovery phase, all the discovered checks will be loaded and ready to be run or listed.
At this phase you can select which regression checks should be finally run or listed.
There are two ways to select regression checks: (a) by programming environment and (b) by tags.

### Selecting checks by programming environment

To select tests by the programming environment, use the `-p` or `--prgenv` options:

```bash
./bin/reframe -p PrgEnv-gnu -l
```

This will select all the checks that support the `PrgEnv-gnu` environment.

You can also specify multiple times the `-p` option, in which case a test will be selected if it support all the programming environments specified in the command line.
For example the following will select all the checks that can run with both `PrgEnv-cray` and `PrgEnv-gnu`:

```bash
./bin/reframe -p PrgEnv-gnu -p PrgEnv-cray -l
```

Note here that specifying the `-p` option will run the selected checks only for the specified programming environments and not for the supported programming environments by the system, which is the default behavior.


### Selecting checks by tags

Each regression check may be associated with a set of tags.
Using the `-t` or `--tag` option you can select the regression checks associated with a specific tag.
For example the following will list all the checks that have a `maintenance` tag:

```bash
./bin/reframe -t maintenance -l
```

Similarly to the `-p` option, you can chain multiple `-t` options together, in which case a regression check will be selected if it is associated with all the tags specified in the command line.
The list of tags associated with a check can be viewed in the listing output when specifying the `-l` option.

Currently, we have two major "official" tags:
(a) `production` for specifying checks to be run daily, while the system is in production and (b) `maintenance` for specifying checks to be run on system maintenance sessions.


### Selecting checks by name

It is possible to select or exclude checks by name through the `--name` or `-n` and `--exclude` or `-x` options.
For example, you can select only the `amber_cpu_check` as follows:
```bash
./bin/reframe -n amber_cpu_check -l
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
./bin/reframe -c /path/to/my/check.py -p PrgEnv-gnu --skip-prgenv-check -r
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

* Stage directory: `${prefix}/stage/`
* Output directory: `${prefix}/output/`
* Log directory: `${prefix}/logs`

You can append timestamp directory to the stage and output directories using the `--timestamp` option.
The default format of the timestamp is `yyyy-mm-ddThh:mm:ss`.
You can change the timestamp by passing a time format to the `--timestamp` option.
The time format is any `strftime()` compatible string.

You can override either the default global prefix or any of the default individual regression directories using the corresponding options.

* `--prefix DIR`: set regression's prefix to `DIR`.
* `--output DIR`: set regression's output directory to `DIR`.
* `--stage DIR`: set regression's stage directory to `DIR`.
* `--logdir DIR`: set regression's log directory to `DIR`.

The stage and output directories are created only when you run a regression check.
However you can view the directories that will be created even when you do a listing of the available checks with the `-l` option.
This is useful if you want to check the directories that regression will create.

```bash
./bin/reframe --prefix /foo -l
```
```
Command line: ./bin/reframe --prefix /foo -l
Reframe version: 2.4
Launched by user: karakasv
Launched on host: daint101
Reframe paths
=============
    Check prefix      : /users/karakasv/Devel/reframe
(R) Check search path : 'checks/'
    Stage dir prefix  : /foo/stage/2017-06-27T23:51:26
    Output dir prefix : /foo/output/2017-06-27T23:51:26
    Logging dir       : /foo/logs
List of matched checks
======================
...
```

You can also define different default regression directories per system by specifying them in the [site configuration](/configure/#new-system-configuration) settings file.
However, the command line options take always precedence over any default directory.

## Logging

From version 2.4 onward, Reframe supports logging of its actions.
Reframe creates two files inside the current working directory every time it is run:
* `reframe.out`: This file stores the output of the reframe run as it was printed in the standard output.
* `reframe.log`: This file stores more detailed of information on Reframe's actions.

By default, the output in `reframe.log` looks like the following:
```
[2017-07-09T21:22:05] info: reframe: [ RUN      ] amber_gpu_check on daint:gpu using PrgEnv-gnu
[2017-07-09T21:22:05] debug: amber_gpu_check: setting up the environment
[2017-07-09T21:22:05] debug: amber_gpu_check: loading environment for partition daint:gpu
[2017-07-09T21:22:05] debug: amber_gpu_check: loading environment PrgEnv-gnu
[2017-07-09T21:22:06] debug: amber_gpu_check: setting up paths
[2017-07-09T21:22:06] debug: amber_gpu_check: setting up the job descriptor
[2017-07-09T21:22:06] debug: amber_gpu_check: job scheduler backend: nativeslurm
[2017-07-09T21:22:06] debug: amber_gpu_check: setting up performance logging
[2017-07-09T21:22:06] debug: amber_gpu_check: copying /apps/common/regression/resources/Amber/ to stage directory (/users/karakasv/Devel/reframe/stage/gpu/amber_gpu_check/PrgEnv-gnu)
[2017-07-09T21:22:06] debug: amber_gpu_check: symlinking files: []
[2017-07-09T21:22:06] debug: amber_gpu_check: spawned job (jobid=2251529)
[2017-07-09T21:22:54] debug: amber_gpu_check: spawned job finished
[2017-07-09T21:22:54] debug: amber_gpu_check: sanity check result: True
[2017-07-09T21:22:54] debug: amber_gpu_check: performance check result: True
[2017-07-09T21:22:54] debug: amber_gpu_check: copying interesting files to output directory
[2017-07-09T21:22:54] debug: amber_gpu_check: removing stage directory
[2017-07-09T21:22:54] info: reframe: [       OK ] amber_gpu_check on daint:gpu using PrgEnv-gnu
```
Each line starts with a timestamp, the level of this message (`info`, `debug` etc.), the context that the framework is currently executing in (either `reframe` or the name of the check, of which behalf it executes) and, finally, the actual message.

Every time Reframe is run, both the `reframe.out` and `reframe.log` files will be rewritten.
However, you can ask Reframe to copy them to the output directory before exiting by passing it the `--save-log-files` option.

### Configuring logging

You can configure several aspects of logging in Reframe and even how the output will look like.
Reframe's logging mechanism is built upon Python's [logging](https://docs.python.org/3.5/library/logging.html) framework adding extra logging levels and more formatting capabilities.

Logging in Reframe is configured by the `logging_config` variable in the `reframe/settings.py` file.
The default configuration looks as follows:
```
logging_config = {
    'level': 'DEBUG',
    'handlers': {
        'reframe.log' : {
            'level'     : 'DEBUG',
            'format'    : '[%(asctime)s] %(levelname)s: '
                          '%(check_name)s: %(message)s',
            'append'    : False,
        },

        # Output handling
        '&1': {
            'level'     : 'INFO',
            'format'    : '%(message)s'
        },
        'reframe.out' : {
            'level'     : 'INFO',
            'format'    : '%(message)s',
            'append'    : False,
        }
    }
}
```

Please not that this configuration dictionary is not the same as the one used by Python's logging framework.
It is a simplified version adapted to the needs of Reframe.

The `logging_config` dictionary has two main key entries:
* `level`: [default: `'INFO'`] This is the lowest level of messages that will be passed down to the different log record handlers.
   Any message with a lower level than that, it will be filtered out immediately and will not be passed to any handler.
   Reframe defines the following logging levels with a decreasing severity: `CRITICAL`, `ERROR`, `WARNING`, `INFO`, `VERBOSE` and `DEBUG`.
   Note that the level name is *not* case sensitive in Reframe.
* `handlers`: A dictionary defining the properties of the handlers that are attached to Reframe's logging mechanism.
   The key is either filename or a special character combination denoting standard output (`&1`) or standard error (`&2`).
   You can attach as many handlers as you like.
   The value of each handler key is another dictionary that holds the properties of the corresponding handler as key/value pairs.

The configurable properties of a log record handler are the following:
* `level`: [default: `'debug'`] The lowest level of log records that this handler can process.
* `format`: [default: `'%(message)s'`] Format string for the printout of the log record.
   Reframe supports all the [format strings](https://docs.python.org/3.5/library/logging.html#logrecord-attributes) from Python's logging library and provides two additional ones:
     - `check_name`: Prints the name of the regression check on behalf of which Reframe is currently executing.
       If Reframe is not in the context of regression check, `reframe` will be printed.
     - `check_jobid`: Prints the job or process id of the job or process associated to currently executing regression check.
       If a job or process is not yet created, `-1` will be printed.
* `datefmt`: [default: `'%FT%T'`] The format that will be used for outputting timestamps (i.e., the `%(asctime)s` field).
  Acceptable formats must conform to standard library's [time.strftime()](https://docs.python.org/3.5/library/time.html#time.strftime) function.
* `append`: [default: `False`] Controls whether Reframe should append to this file or not.
  This is ignored for the standard output/error handlers.
* `timestamp` [default: `None`]: Append a timestamp to this log filename.
  This property may accept any date format as the `datefmt` property.
  If set for a `filename.log` handler entry, the resulting log file name will be `filename_<timestamp>.log`.
  This property is ignored for the standard output/error handlers.

### Performance Logging

ReFrame supports an additional level of logging for performance checks specifically, in order to record historical performance data.
For each performance check, a log file of the form `<check-name>.log` is created under the regression's [log directory](#configuring-regression-directories) where the check's performance is record.
The default format used for this file is `'[%(asctime)s] %(check_name)s (jobid=%(check_jobid)s): %(message)s'` and Reframe always appends to this file.
Currently, it is not possible for users to configure performance logging.

This resulting log file looks like the following:
```
[2017-07-09T21:22:54] amber_gpu_check (jobid=2251529): value: 21.45, reference: (21.0, -0.05, None)
[2017-07-09T21:37:55] amber_gpu_check (jobid=2251565): value: 21.56, reference: (21.0, -0.05, None)
[2017-07-09T21:44:06] amber_gpu_check (jobid=2251588): value: 21.49, reference: (21.0, -0.05, None)
```

The interpretation of the performance values depend on the individual checks.
In the above check, for example, the performance value refers to ns/day.
The reference value is a three-element tuple of the form `(<reference>, <low-threshold>, <high-threshold>)`, where the `low-threshold` and `high-threshold` are the acceptable tolerance thresholds expressed in percentages. For example, the performance check shown above has a reference value of 21.0 ns/day with a maximum high tolerance of 5%.
There is no upper tolerance, since higher values denote higher performance in this case.


## Asynchronous execution of regression checks

From version [2.4](https://github.com/eth-cscs/reframe/releases/tag/v2.4), Reframe supports asynchronous execution of the regression checks.
This execution policy can be enabled by passing the option `--exec-policy=async` to the command line.
The default execution policy is `serial` which enforces a sequential execution of the selected regression checks.
The asynchronous execution policy parallelizes the "run" phase of the checks only.
The rest of the phases remain sequential.

A limit of concurrent jobs (pending and running) may be configured for each virtual system partition.
As soon as the concurrency limit of a partition is reached, Reframe will hold the execution of the regression check until a slot is released in that partition.

When executing in asynchronous mode, Reframe's output differs from the sequential execution.
The final result of the checks will be printed at the end and additional messages may be printed to indicate that a check is held.
Here is an example output of Reframe using asynchronous execution policy:

```
Command line: ./bin/reframe --exec-policy=async -c checks/cuda/cuda_checks.py -r
Reframe version: 2.5
Launched by user: karakasv
Launched on host: daint103
Reframe paths
=============
    Check prefix      :
    Check search path : 'checks/cuda/cuda_checks.py'
    Stage dir prefix  : /users/karakasv/Devel/reframe/stage/
    Output dir prefix : /users/karakasv/Devel/reframe/output/
    Logging dir       : /users/karakasv/Devel/reframe/logs
[==========] Running 5 check(s)
[==========] Started on Thu Aug 24 17:53:39 2017

[----------] started processing cuda_bandwidth_check (CUDA bandwidthTest compile and run)
[   SKIP   ] skipping daint:login
[   SKIP   ] skipping daint:mc
[ RUN      ] cuda_bandwidth_check on daint:gpu using PrgEnv-cray
[ RUN      ] cuda_bandwidth_check on daint:gpu using PrgEnv-gnu
[   SKIP   ] skipping PrgEnv-intel for daint:gpu
[   SKIP   ] skipping PrgEnv-pgi for daint:gpu
[----------] finished processing cuda_bandwidth_check (CUDA bandwidthTest compile and run)

[----------] started processing cuda_concurrentkernels_check (Use of streams for concurrent execution)
[   SKIP   ] skipping daint:login
[   SKIP   ] skipping daint:mc
[ RUN      ] cuda_concurrentkernels_check on daint:gpu using PrgEnv-cray
[ RUN      ] cuda_concurrentkernels_check on daint:gpu using PrgEnv-gnu
[   SKIP   ] skipping PrgEnv-intel for daint:gpu
[   SKIP   ] skipping PrgEnv-pgi for daint:gpu
[----------] finished processing cuda_concurrentkernels_check (Use of streams for concurrent execution)

[----------] started processing cuda_matrixmulcublas_check (Implements matrix multiplication using CUBLAS)
[   SKIP   ] skipping daint:login
[   SKIP   ] skipping daint:mc
[ RUN      ] cuda_matrixmulcublas_check on daint:gpu using PrgEnv-cray
[ RUN      ] cuda_matrixmulcublas_check on daint:gpu using PrgEnv-gnu
[   SKIP   ] skipping PrgEnv-intel for daint:gpu
[   SKIP   ] skipping PrgEnv-pgi for daint:gpu
[----------] finished processing cuda_matrixmulcublas_check (Implements matrix multiplication using CUBLAS)

[----------] started processing cuda_simplempi_check (Simple example demonstrating how to use MPI with CUDA)
[   SKIP   ] skipping daint:login
[   SKIP   ] skipping daint:mc
[ RUN      ] cuda_simplempi_check on daint:gpu using PrgEnv-cray
[ RUN      ] cuda_simplempi_check on daint:gpu using PrgEnv-gnu
[   SKIP   ] skipping PrgEnv-intel for daint:gpu
[   SKIP   ] skipping PrgEnv-pgi for daint:gpu
[----------] finished processing cuda_simplempi_check (Simple example demonstrating how to use MPI with CUDA)

[----------] waiting for spawned checks
[       OK ] cuda_concurrentkernels_check on daint:gpu using PrgEnv-gnu
[       OK ] cuda_bandwidth_check on daint:gpu using PrgEnv-gnu
[       OK ] cuda_bandwidth_check on daint:gpu using PrgEnv-cray
[       OK ] cuda_concurrentkernels_check on daint:gpu using PrgEnv-cray
[       OK ] cuda_devicequery_check on daint:gpu using PrgEnv-cray
[       OK ] cuda_devicequery_check on daint:gpu using PrgEnv-gnu
[       OK ] cuda_matrixmulcublas_check on daint:gpu using PrgEnv-cray
[       OK ] cuda_matrixmulcublas_check on daint:gpu using PrgEnv-gnu
[       OK ] cuda_simplempi_check on daint:gpu using PrgEnv-cray
[       OK ] cuda_simplempi_check on daint:gpu using PrgEnv-gnu
[----------] all spawned checks finished
[  PASSED  ] Ran 10 test case(s) from 5 check(s) (0 failure(s))
[==========] Finished on Thu Aug 24 18:04:18 2017
```

The asynchronous execution policy may provide significant overall performance benefits for run-only regression tests.
For compile-only and normal tests that require a compilation, the execution time will be bound by the total compilation time of the test.


### Setting concurrency limits
As mentioned earlier, it is possible to specify different limits for concurrent jobs per configured virtual partitioned.
This can be achieved by setting the `max_jobs` property of a partition in the `site_configuration` dictionary as follows:

```python
...
'systems' : {
    # Generic system used for cli unit tests
    'generic' : {
        'descr' : 'Generic example system',
        'partitions' : {
            'login' : {
                'scheduler' : 'local',
                'modules'   : [],
                'access'    : [],
                'environs'  : [ 'builtin-gcc' ],
                'descr'     : 'Login nodes',
                'max_jobs'  : 4
            }
        }
    }
},
...

```

## Execution modes

From version [2.5](https://github.com/eth-cscs/reframe/releases/tag/v2.5) onward, Reframe permits users to define different *execution modes* of the framework.
An execution mode is merely a set of predefined command-line options that will be passed to Reframe when this mode is invoked.
You can define execution modes per system in the Reframe's configuration file.
For example, one could define a global `maintenance` execution mode as follows in the `settings.py` file:

```python
site_configuration = ReadOnlyField({
    'systems' : {
        ...
    },
    'environments' : {
        ...
    },
    'modes' : {
        *' : {
            'maintenance' : [
                '--exec-policy=async',
                '--output=$STORE/regression/maintenance',
                '--logdir=$STORE/regression/maintenance/logs',
                '--stage=$WORK/regression/maintenance/stage',
                '--reservation=maintenance',
                '--save-log-files',
                '--tag=maintenance',
                '--timestamp=%F_%H-%M-%S',
            ],
        }
    }
})
```

Whenever a user invokes Reframe with `--mode=maintenance`, all of the predefined options of that mode will be passed to the invocation.
Note that the framework will expand any shell variables specified inside a mode.
The user may also pass additional command line options.
Command line options always override the options set by the mode, so that with the above configuration, the following will reset the execution policy to `serial`:
```
./bin/reframe --mode=maintenance --exec-policy=serial -r
```

It should be noted here that if a boolean option is defined in an execution mode, this may not be overriden if the inverse option is not provided by the framework.
For example, with the above definition of the `maintenance` mode the `--save-log-files` option cannot be overriden, since there is no option currently in the framework to invert its action.

Execution modes may be defined or redefined per system as it is the case also with the [programming environments](/configure#environment-configuration).

## Examples of usage

1. Run all tests with the `production` tag and place the output of the regression in your home directory:
```bash
./bin/reframe -o $HOME/regression/output -t production -r
```

2. List all tests with the `maintenance` and `slurm` tags:
```bash
./bin/reframe -t maintenance -t slurm -l
```

2. Run all the maintenance checks on the `foo` reservation:
```bash
./bin/reframe -t maintenance --reservation=foo -r
```

2. List all production tests supporting `PrgEnv-gnu` and having the `production` tag:
```bash
./bin/reframe -p PrgEnv-gnu -t production -l
```

3. Run a specific check on a new system that is not officially supported by the check:
```bash
./bin/reframe -c /path/to/my/check.py --skip-system-check -r
```

4. Run a specific check on a programming environment (e.g., `PrgEnv-pgi`) that is not officially supported by the check:
```bash
./bin/reframe -c /path/to/my/check.py -p PrgEnv-pgi --skip-prgenv-check -r
```
