# Running ReFrame

Before getting into any details, the simplest way to invoke ReFrame is the following:
```bash
./bin/reframe -c /path/to/checks -R --run
```
This will search recursively for test files in `/path/to/checks` and will start running them on the current system.

ReFrame's front-end goes through three phases:

1. Load tests
2. Filter tests
3. Act on tests

In the following, we will elaborate on these phases and the key command-line options controlling them.
A detailed listing of all the command-line options grouped by phase is given by `./bin/reframe -h`.

## Supported Actions
Even though an action is the last phase that the front-end goes through, we are listing it first since an action is always required.
Currently there are only two available actions:

1. Listing of the selected checks
2. Execution of the selected checks


### Listing of the regression tests
To retrieve a listing of the selected checks, you must specify the `-l` or `--list` options.
An example listing of checks is the following that lists all the tests found under the `tutorial/` folder:

```bash
./bin/reframe -c tutorial -l
```

The ouput looks like:
```
Command line: ./bin/reframe -c tutorial/ -l
Reframe version: 2.6.1
Launched by user: karakasv
Launched on host: daint103
Reframe paths
=============
    Check prefix      :
    Check search path : 'tutorial/'
    Stage dir prefix  : /users/karakasv/Devel/reframe/stage/
    Output dir prefix : /users/karakasv/Devel/reframe/output/
    Logging dir       : /users/karakasv/Devel/reframe/logs
List of matched checks
======================
  * example1_check (Simple matrix-vector multiplication example)
        tags: [tutorial], maintainers: [you-can-type-your-email-here]
  * example2a_check (Matrix-vector multiplication example with OpenMP)
        tags: [tutorial], maintainers: [you-can-type-your-email-here]
  * example2b_check (Matrix-vector multiplication example with OpenMP)
        tags: [tutorial], maintainers: [you-can-type-your-email-here]
  * example3_check (Matrix-vector multiplication example with MPI)
        tags: [tutorial], maintainers: [you-can-type-your-email-here]
  * example4_check (Matrix-vector multiplication example with OpenACC)
        tags: [tutorial], maintainers: [you-can-type-your-email-here]
  * example5_check (Matrix-vector multiplication example with Cuda)
        tags: [tutorial], maintainers: [you-can-type-your-email-here]
  * example6_check (Matrix-vector multiplication with L2 norm check)
        tags: [tutorial], maintainers: [you-can-type-your-email-here]
  * example7_check (Matrix-vector multiplication example with Cuda)
        tags: [tutorial], maintainers: [you-can-type-your-email-here]
  * example8_serial_check (Serial matrix-vector multiplication)
        tags: [tutorial], maintainers: [you-can-type-your-email-here]
  * example8_openmp_check (OpenMP matrix-vector multiplication)
        tags: [tutorial], maintainers: [you-can-type-your-email-here]
  * example8_mpi_check (MPI matrix-vector multiplication)
        tags: [tutorial], maintainers: [you-can-type-your-email-here]
  * example8_openacc_check (OpenACC matrix-vector multiplication)
        tags: [tutorial], maintainers: [you-can-type-your-email-here]
  * example8_cuda_check (Cuda matrix-vector multiplication)
        tags: [tutorial], maintainers: [you-can-type-your-email-here]
Found 13 check(s).
```
The listing contains the name of the check, its description, the tags associated with it and a list of its maintainers.
Note that this listing may also contain checks that are not supported by the current system.
These checks will be just skipped if you try to run them.

### Execution of the regression tests

To run the regression tests you should specify the _run_ action though the `-r` or `--run` options.
> NOTE: The listing action takes precedence over the execution, meaning that if you specify both `-l -r`, only the listing action will be performed.

```bash
./bin/reframe --notimestamp -c checks/cuda/cuda_checks.py --prefix . -r
```

The output of the regression run looks like the following:
```
Command line: ./bin/reframe -c tutorial/example1.py -r
Reframe version: 2.6.1
Launched by user: karakasv
Launched on host: daint103
Reframe paths
=============
    Check prefix      :
    Check search path : 'tutorial/example1.py'
    Stage dir prefix  : /users/karakasv/Devel/reframe/stage/
    Output dir prefix : /users/karakasv/Devel/reframe/output/
    Logging dir       : /users/karakasv/Devel/reframe/logs
[==========] Running 1 check(s)
[==========] Started on Tue Oct 24 18:13:33 2017

[----------] started processing example1_check (Simple matrix-vector multiplication example)
[ RUN      ] example1_check on daint:mc using PrgEnv-cray
[       OK ] example1_check on daint:mc using PrgEnv-cray
[ RUN      ] example1_check on daint:mc using PrgEnv-gnu
[       OK ] example1_check on daint:mc using PrgEnv-gnu
[ RUN      ] example1_check on daint:mc using PrgEnv-intel
[       OK ] example1_check on daint:mc using PrgEnv-intel
[ RUN      ] example1_check on daint:mc using PrgEnv-pgi
[       OK ] example1_check on daint:mc using PrgEnv-pgi
[ RUN      ] example1_check on daint:gpu using PrgEnv-cray
[       OK ] example1_check on daint:gpu using PrgEnv-cray
[ RUN      ] example1_check on daint:gpu using PrgEnv-gnu
[       OK ] example1_check on daint:gpu using PrgEnv-gnu
[ RUN      ] example1_check on daint:gpu using PrgEnv-intel
[       OK ] example1_check on daint:gpu using PrgEnv-intel
[ RUN      ] example1_check on daint:gpu using PrgEnv-pgi
[       OK ] example1_check on daint:gpu using PrgEnv-pgi
[ RUN      ] example1_check on daint:login using PrgEnv-cray
[       OK ] example1_check on daint:login using PrgEnv-cray
[ RUN      ] example1_check on daint:login using PrgEnv-gnu
[       OK ] example1_check on daint:login using PrgEnv-gnu
[ RUN      ] example1_check on daint:login using PrgEnv-intel
[       OK ] example1_check on daint:login using PrgEnv-intel
[ RUN      ] example1_check on daint:login using PrgEnv-pgi
[       OK ] example1_check on daint:login using PrgEnv-pgi
[----------] finished processing example1_check (Simple matrix-vector multiplication example)

[  PASSED  ] Ran 12 test case(s) from 1 check(s) (0 failure(s))
[==========] Finished on Tue Oct 24 18:15:06 2017
```

## Discovery of Regression Tests

When ReFrame is invoked, it tries to locate regression tests in a predefined path.
By default, this path is the `<reframe-install-dir>/checks`.
You can also retrieve this path as follows:

```bash
./bin/reframe -l | grep 'Check search path'
```

If the path line is prefixed with `(R)`, every directory in that path will be searched recursively for regression tests.

As described extensively in the ["ReFrame Tutorial"](tutorial.html), regression tests in ReFrame are essentially Python source files that provide a special function, which returns the actual regression test instances.
A single source file may also provide multiple regression tests.
ReFrame loads the python source files and tries to call this special function;
if this function cannot be found, the source file will be ignored.
At the end of this phase, the front-end will have instantiated all the tests found in the path.

You can override the default search path for tests by specifying the `-c` or `--checkpath` options.
We have already done that already when listing all the tutorial tests:

```bash
./bin/reframe -c tutorial/ -l
```

ReFrame the does not search recursively into directories specified with the `-c` option, unless you explicitly specify the `-R` or `--recurse` options.

The `-c` option completely overrides the default path.
Currently, there is no option to prepend or append to the default regression path.
However, you can build your own check path by specifying multiple times the `-c` option.
The `-c`option accepts also regular files.
This is very useful when you are implementing new regression tests, since it allows you to run only your test:

```bash
./bin/reframe -c /path/to/my/new/test.py -r
```

## Filtering of Regression Tests

At this phase you can select which regression tests should be run or listed.
There are several ways to select regression tests, which we describe in more detail here:

### Selecting tests by programming environment

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
If you are going to run a set of tests selected by programming environment, they will run only for the selected programming environment(s).

### Selecting tests by tags

As we have seen in the ["ReFrame tutorial"](tutorial.html), every regression test may be associated with a set of tags.
Using the `-t` or `--tag` option you can select the regression tests associated with a specific tag.
For example the following will list all the tests that have a `maintenance` tag:

```bash
./bin/reframe -t maintenance -l
```

Similarly to the `-p` option, you can chain multiple `-t` options together, in which case a regression test will be selected if it is associated with all the tags specified in the command line.
The list of tags associated with a check can be viewed in the listing output when specifying the `-l` option.


### Selecting tests by name

It is possible to select or exclude tests by name through the `--name` or `-n` and `--exclude` or `-x` options.
For example, you can select only the `example7_check` from the tutorial as follows:
```bash
./bin/reframe -c tutorial n example7_check -l
```

```
Command line: ./bin/reframe -c tutorial/ -n example7_check -l
Reframe version: 2.6.1
Launched by user: karakasv
Launched on host: daint103
Reframe paths
=============
    Check prefix      :
    Check search path : 'tutorial/'
    Stage dir prefix  : /users/karakasv/Devel/reframe/stage/
    Output dir prefix : /users/karakasv/Devel/reframe/output/
    Logging dir       : /users/karakasv/Devel/reframe/logs
List of matched checks
======================
  * example7_check (Matrix-vector multiplication example with Cuda)
        tags: [tutorial], maintainers: [you-can-type-your-email-here]
Found 1 check(s).
```

Similarly, you can exclude this test by passing the `-x example7_check` option:

```
Command line: ./bin/reframe -c tutorial/ -x example7_check -l
Reframe version: 2.6.1
Launched by user: karakasv
Launched on host: daint103
Reframe paths
=============
    Check prefix      :
    Check search path : 'tutorial/'
    Stage dir prefix  : /users/karakasv/Devel/reframe/stage/
    Output dir prefix : /users/karakasv/Devel/reframe/output/
    Logging dir       : /users/karakasv/Devel/reframe/logs
List of matched checks
======================
  * example1_check (Simple matrix-vector multiplication example)
        tags: [tutorial], maintainers: [you-can-type-your-email-here]
  * example2a_check (Matrix-vector multiplication example with OpenMP)
        tags: [tutorial], maintainers: [you-can-type-your-email-here]
  * example2b_check (Matrix-vector multiplication example with OpenMP)
        tags: [tutorial], maintainers: [you-can-type-your-email-here]
  * example3_check (Matrix-vector multiplication example with MPI)
        tags: [tutorial], maintainers: [you-can-type-your-email-here]
  * example4_check (Matrix-vector multiplication example with OpenACC)
        tags: [tutorial], maintainers: [you-can-type-your-email-here]
  * example5_check (Matrix-vector multiplication example with Cuda)
        tags: [tutorial], maintainers: [you-can-type-your-email-here]
  * example6_check (Matrix-vector multiplication with L2 norm check)
        tags: [tutorial], maintainers: [you-can-type-your-email-here]
  * example8_serial_check (Serial matrix-vector multiplication)
        tags: [tutorial], maintainers: [you-can-type-your-email-here]
  * example8_openmp_check (OpenMP matrix-vector multiplication)
        tags: [tutorial], maintainers: [you-can-type-your-email-here]
  * example8_mpi_check (MPI matrix-vector multiplication)
        tags: [tutorial], maintainers: [you-can-type-your-email-here]
  * example8_openacc_check (OpenACC matrix-vector multiplication)
        tags: [tutorial], maintainers: [you-can-type-your-email-here]
  * example8_cuda_check (Cuda matrix-vector multiplication)
        tags: [tutorial], maintainers: [you-can-type-your-email-here]
Found 12 check(s).
```

## Controlling the Execution of Regression Tests

There are several options for controlling the execution of regression tests.
Keep in mind that these options will affect all the tests that will run with the current invocation.
They are summarized below:

* `-A ACCOUNT`, `--account ACCOUNT`: Submit regression test jobs using `ACCOUNT`.
* `-P PART`, `--partition PART`: Submit regression test jobs in the _scheduler partition_ `PART`.
* `--reservation RES`: Submit regression test jobs in reservation `RES`.
* `--nodelist NODELIST`: Run regression test jobs on the nodes specified in `NODELIST`.
* `--exclude-nodes NODELIST`: Do not run the regression test jobs on any of the nodes specified in `NODELIST`.
* `--job-option OPT`: Pass option `OPT` directly to the back-end job scheduler.
  This option *must* be used with care, since you may break the submission mechanism.
  All of the above job submission related options could be expressed with this option.
  For example, the `-n NODELIST` is equivalent to `--job-option='--nodelist=NODELIST'` for a Slurm job scheduler.
  If you pass an option that is already defined by the framework, the framework will *not* explicitly override it; this is up to scheduler.
  All extra options defined from the command line are appended to the automatically generated options in the generated batch script file.
  So if you redefine one of them, e.g., `--output` for the Slurm scheduler, it is up the job scheduler on how to interpret multiple definitions of the same options.
  In this example, Slurm's policy is that later definitions of options override previous ones.
  So, in this case, way you would override the standard output for all the submitted jobs!

* `--force-local`: Force the local execution of the selected tests. No jobs will be submitted.
* `--skip-sanity-check`: Skip sanity checking phase.
* `--skip-performance-check`: Skip performance verification phase.
* `--strict`: Force strict performance checking.
  Some tests may set their `strict_check` attribute to `False` (see ["Reference Guide"](reference.html)) in order to just let their performance recorded but not yield an error.
  This option overrides this behavior and forces all tests to be strict.
* `--skip-system-check`: Skips the system check and run the selected tests even if they do not support the current system.
  This option is sometimes useful when you need to quickly verify if a regression test supports a new system.
* `--skip-prgenv-check`: Skips programming environment check and run the selected tests for even if they do not support a programming environment.
  This option is useful when you need to quickly verify if a regression check supports another programming environment.
  For example, if you know that a tests supports only `PrgEnv-cray` and you need to check if it also works with `PrgEnv-gnu`, you can test is as follows:

```bash
./bin/reframe -c /path/to/my/check.py -p PrgEnv-gnu --skip-prgenv-check -r
```

## Configuring ReFrame Directories

ReFrame uses three basic directories during the execution of tests:

1. The stage directory
    - Each regression test is executed in a "sandbox";
      all of its resources (source files, resources) are copied over to a stage directory and executed from there.
      This will also be the working directory for the test.
2. The output directory
    - After a regression test finishes some important files will be copied from the stage directory to the output directory.
      By default these are the standard output, standard error and the generated job script file.
      A regression test may also specify to keep additional files.
3. The log directory
    - This is where the performance log files of the individual performance tests are placed (see [Logging](#logging) for more information)

By default, all these directories are placed under a common prefix, which defaults to `.`.
The rest of the directories are organized as follows:

* Stage directory: `${prefix}/stage/<timestamp>`
* Output directory: `${prefix}/output/<timestamp>`
* Performance log directory: `${prefix}/logs`

You can optionally append a timestamp directory component to the above paths (except the logs directory), by using the `--timestamp` option.
This options takes an optional argument to specify the timestamp format.
The default [time format](http://man7.org/linux/man-pages/man3/strftime.3.html) is `%FT%T`, which results into timestamps of the form `2017-10-24T21:10:29`.

You can override either the default global prefix or any of the default individual directories using the corresponding options.

* `--prefix DIR`: set prefix to `DIR`.
* `--output DIR`: set output directory to `DIR`.
* `--stage DIR`: set stage directory to `DIR`.
* `--logdir DIR`: set performance log directory to `DIR`.

The stage and output directories are created only when you run a regression test.
However you can view the directories that will be created even when you do a listing of the available checks with the `-l` option.
This is useful if you want to check the directories that ReFrame will create.

```bash
./bin/reframe --prefix /foo -l
```
```
Command line: ./bin/reframe --prefix /foo -t foo -l
Reframe version: 2.6.1
Launched by user: karakasv
Launched on host: daint103
Reframe paths
=============
    Check prefix      : /users/karakasv/Devel/reframe
(R) Check search path : 'checks/'
    Stage dir prefix  : /foo/stage/
    Output dir prefix : /foo/output/
    Logging dir       : /foo/logs
List of matched checks
======================
Found 0 check(s).
```

You can also define different default directories per system by specifying them in the [site configuration](configure.html#the-configuration-file) settings file.
The command line options, though, take always precedence over any default directory.

## Logging

From version 2.4 onward, ReFrame supports logging of its actions.
ReFrame creates two files inside the current working directory every time it is run:

* `reframe.out`: This file stores the output of a run as it was printed in the standard output.
* `reframe.log`: This file stores more detailed of information on ReFrame's actions.

By default, the output in `reframe.log` looks like the following:
```
[2017-10-24T21:19:04] info: reframe: [----------] started processing example7_check (Matrix-vector mult
iplication example with Cuda)
[2017-10-24T21:19:04] info: reframe: [   SKIP   ] skipping daint:mc
[2017-10-24T21:19:04] info: reframe: [ RUN      ] example7_check on daint:gpu using PrgEnv-cray
[2017-10-24T21:19:04] debug: example7_check: setting up the environment
[2017-10-24T21:19:04] debug: example7_check: loading environment for partition daint:gpu
[2017-10-24T21:19:05] debug: example7_check: loading environment PrgEnv-cray
[2017-10-24T21:19:05] debug: example7_check: setting up paths
[2017-10-24T21:19:05] debug: example7_check: setting up the job descriptor
[2017-10-24T21:19:05] debug: example7_check: job scheduler backend: nativeslurm
[2017-10-24T21:19:05] debug: example7_check: setting up performance logging
[2017-10-24T21:19:05] debug: example7_check: compilation started
[2017-10-24T21:19:06] debug: example7_check: compilation stdout:

[2017-10-24T21:19:06] debug: example7_check: compilation stderr:
nvcc warning : The 'compute_20', 'sm_20', and 'sm_21' architectures are deprecated, and may be removed
in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).

[2017-10-24T21:19:06] debug: example7_check: compilation finished
[2017-10-24T21:19:09] debug: example7_check: spawned job (jobid=4163846)
[2017-10-24T21:19:21] debug: example7_check: spawned job finished
[2017-10-24T21:19:21] debug: example7_check: copying interesting files to output directory
[2017-10-24T21:19:21] debug: example7_check: removing stage directory
[2017-10-24T21:19:21] info: reframe: [       OK ] example7_check on daint:gpu using PrgEnv-cray
```
Each line starts with a timestamp, the level of the message (`info`, `debug` etc.), the context in which the framework is currently executing (either `reframe` or the name of the current test and, finally, the actual message.

Every time ReFrame is run, both `reframe.out` and `reframe.log` files will be rewritten.
However, you can ask ReFrame to copy them to the output directory before exiting by passing it the `--save-log-files` option.

### Configuring logging

You can configure several aspects of logging in ReFrame and even how the output will look like.
ReFrame's logging mechanism is built upon Python's [logging](https://docs.python.org/3.6/library/logging.html) framework adding extra logging levels and more formatting capabilities.

Logging in ReFrame is configured by the `_logging_config` variable in the `reframe/settings.py` file.
The default configuration looks as follows:
```
_logging_config = {
    'level': 'DEBUG',
    'handlers': {
        'reframe.log' : {
            'level'     : 'DEBUG',
            'format'    : '[%(asctime)s] %(levelname)s: '
                          '%(testcase_name)s: %(message)s',
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

Note that this configuration dictionary is not the same as the one used by Python's logging framework.
It is a simplified version adapted to the needs of ReFrame.

The `_logging_config` dictionary has two main key entries:

* `level` (default: `'INFO'`): This is the lowest level of messages that will be passed down to the different log record handlers.
   Any message with a lower level than that, it will be filtered out immediately and will not be passed to any handler.
   ReFrame defines the following logging levels with a decreasing severity: `CRITICAL`, `ERROR`, `WARNING`, `INFO`, `VERBOSE` and `DEBUG`.
   Note that the level name is *not* case sensitive in ReFrame.
* `handlers`: A dictionary defining the properties of the handlers that are attached to ReFrame's logging mechanism.
   The key is either a filename or a special character combination denoting standard output (`&1`) or standard error (`&2`).
   You can attach as many handlers as you like.
   The value of each handler key is another dictionary that holds the properties of the corresponding handler as key/value pairs.

The configurable properties of a log record handler are the following:

* `level` (default: `'debug'`): The lowest level of log records that this handler can process.
* `format` (default: `'%(message)s'`): Format string for the printout of the log record.
   ReFrame supports all the [format strings](https://docs.python.org/3.6/library/logging.html#logrecord-attributes) from Python's logging library and provides the following additional ones:
     - `check_name`: Prints the name of the regression test on behalf of which ReFrame is currently executing.
       If ReFrame is not in the context of regression test, `reframe` will be printed.
     - `check_jobid`: Prints the job or process id of the job or process associated with currently executing regression test.
       If a job or process is not yet created, `-1` will be printed.
     - `testcase_name`: Print the name of the test case that is currently executing.
        Test case is essentially a tuple consisting of the test name, the current system and partition and the current programming envrinoment.
        This format string prints out like `<test-name>@<partition> using <environ>`.

* `datefmt` (default: `'%FT%T'`) The format that will be used for outputting timestamps (i.e., the `%(asctime)s` field).
  Acceptable formats must conform to standard library's [time.strftime()](https://docs.python.org/3.6/library/time.html#time.strftime) function.
* `append` (default: `False`) Controls whether ReFrame should append to this file or not.
  This is ignored for the standard output/error handlers.
* `timestamp` (default: `None`): Append a timestamp to this log filename.
  This property may accept any date format as the `datefmt` property.
  If set for a `filename.log` handler entry, the resulting log file name will be `filename_<timestamp>.log`.
  This property is ignored for the standard output/error handlers.

### Performance Logging

ReFrame supports additional logging for performance tests specifically, in order to record historical performance data.
For each performance test, a log file of the form `<test-name>.log` is created under the ReFrame's [log directory](#configuring-reframe-directories) where the test's performance is recorded.
The default format used for this file is `'[%(asctime)s] %(check_name)s (jobid=%(check_jobid)s): %(message)s'` and ReFrame always appends to this file.
Currently, it is not possible for users to configure performance logging.

The resulting log file looks like the following:
```
[2017-10-21T00:48:42] example7_check (jobid=4073910): value: 49.253851, reference: (50.0, -0.1, 0.1)
[2017-10-24T21:19:21] example7_check (jobid=4163846): value: 49.690761, reference: (50.0, -0.1, 0.1)
[2017-10-24T21:19:33] example7_check (jobid=4163852): value: 50.037254, reference: (50.0, -0.1, 0.1)
[2017-10-24T21:20:00] example7_check (jobid=4163856): value: 49.622199, reference: (50.0, -0.1, 0.1)
```

The interpretation of the performance values depends on the individual tests.
The above output is from the CUDA performance test we presented in the [tutorial](tutorial.html#writing-a-performance-test), so the value refers to the achieved Gflop/s.
The reference value is a three-element tuple of the form `(<reference>, <lower-threshold>, <upper-threshold>)`, where the `lower-threshold` and `upper-threshold` are the acceptable tolerance thresholds expressed in percentages. For example, the performance check shown above has a reference value of 50 Gflop/s ± 10%.


## Asynchronous Execution of Regression Checks

From version [2.4](https://github.com/eth-cscs/reframe/releases/tag/v2.4), ReFrame supports asynchronous execution of regression tests.
This execution policy can be enabled by passing the option `--exec-policy=async` to the command line.
The default execution policy is `serial` which enforces a sequential execution of the selected regression tests.
The asynchronous execution policy parallelizes only the [running phase](pipeline.html#the-run-phase) of the tests.
The rest of the phases remain sequential.

A limit of concurrent jobs (pending and running) may be [configured](configure.html#partition-configuration) for each virtual system partition.
As soon as the concurrency limit of a partition is reached, ReFrame will hold the execution of new regression tests until a slot is released in that partition.

When executing in asynchronous mode, ReFrame's output differs from the sequential execution.
The final result of the tests will be printed at the end and additional messages may be printed to indicate that a test is held.
Here is an example output of ReFrame using asynchronous execution policy:

```
ommand line: ./reframe.py -c tutorial/ --exec-policy=async -r
Reframe version: 2.6.1
Launched by user: karakasv
Launched on host: daint104
Reframe paths
=============
    Check prefix      :
    Check search path : 'tutorial/'
    Stage dir prefix  : /users/karakasv/Devel/reframe/stage/
    Output dir prefix : /users/karakasv/Devel/reframe/output/
    Logging dir       : /users/karakasv/Devel/reframe/logs
[==========] Running 13 check(s)
[==========] Started on Sun Nov  5 19:37:09 2017

[----------] started processing example1_check (Simple matrix-vector multiplication example)
[ RUN      ] example1_check on daint:login using PrgEnv-cray
[ RUN      ] example1_check on daint:login using PrgEnv-gnu
[ RUN      ] example1_check on daint:login using PrgEnv-intel
[ RUN      ] example1_check on daint:login using PrgEnv-pgi
[ RUN      ] example1_check on daint:gpu using PrgEnv-cray
[ RUN      ] example1_check on daint:gpu using PrgEnv-gnu
[ RUN      ] example1_check on daint:gpu using PrgEnv-intel
[ RUN      ] example1_check on daint:gpu using PrgEnv-pgi
[ RUN      ] example1_check on daint:mc using PrgEnv-cray
[ RUN      ] example1_check on daint:mc using PrgEnv-gnu
[ RUN      ] example1_check on daint:mc using PrgEnv-intel
[ RUN      ] example1_check on daint:mc using PrgEnv-pgi
[----------] finished processing example1_check (Simple matrix-vector multiplication example)

...

[----------] started processing example8_cuda_check (Cuda matrix-vector multiplication)
[   SKIP   ] skipping daint:login
[ RUN      ] example8_cuda_check on daint:gpu using PrgEnv-cray
[ RUN      ] example8_cuda_check on daint:gpu using PrgEnv-gnu
[   SKIP   ] skipping PrgEnv-intel for daint:gpu
[ RUN      ] example8_cuda_check on daint:gpu using PrgEnv-pgi
[   SKIP   ] skipping daint:mc
[----------] finished processing example8_cuda_check (Cuda matrix-vector multiplication)

[----------] waiting for spawned checks
[       OK ] example1_check on daint:login using PrgEnv-cray
[       OK ] example1_check on daint:login using PrgEnv-gnu
[       OK ] example1_check on daint:login using PrgEnv-intel
[       OK ] example1_check on daint:login using PrgEnv-pgi
[       OK ] example1_check on daint:gpu using PrgEnv-cray
[       OK ] example1_check on daint:gpu using PrgEnv-gnu
[       OK ] example1_check on daint:gpu using PrgEnv-intel
[       OK ] example1_check on daint:gpu using PrgEnv-pgi
[       OK ] example1_check on daint:mc using PrgEnv-cray
[       OK ] example1_check on daint:mc using PrgEnv-gnu
[       OK ] example1_check on daint:mc using PrgEnv-intel
[       OK ] example1_check on daint:mc using PrgEnv-pgi
...
[       OK ] example8_openacc_check on daint:gpu using PrgEnv-cray
[       OK ] example8_openacc_check on daint:gpu using PrgEnv-pgi
[       OK ] example8_cuda_check on daint:gpu using PrgEnv-cray
[       OK ] example8_cuda_check on daint:gpu using PrgEnv-gnu
[       OK ] example8_cuda_check on daint:gpu using PrgEnv-pgi
[----------] all spawned checks finished
[  PASSED  ] Ran 97 test case(s) from 13 check(s) (0 failure(s))
[==========] Finished on Sun Nov  5 19:42:23 2017
```

The asynchronous execution policy may provide significant overall performance benefits for run-only regression tests.
For compile-only and normal tests that require a compilation, the execution time will be bound by the total compilation time of the test.
