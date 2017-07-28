# Basics of Regression Checks

The regression pipeline described in [Pipeline section](/pipeline) is implemented in a base class called `RegressionTest`, from which all user regression tests must eventually inherit.
There exist also base classes (inheriting also from `RegressionTest`) that implement the special regression test types described in [Types of Regression Checks](#types-of-regression-checks).
A user test may inherit from any of the base regression tests depending on the type of check (*normal*, *run-only* or *compile-only*).

The initialization phase of a regression test is implemented in the test's constructor, i.e., the `__init__()` method of the regression test class.
The constructor of a user regression test is only required to allow keyword arguments to be passed to it.
These are needed to initialize the base `RegressionTest`.
Of course, a user regression test may accept any number of positional arguments that are specific to the test and are used to control its construction.
The following listing shows the boiler plate code for implementing new regression tests classes:
```python
class HelloTest(RegressionTest):
    def __init__(self, **kwargs):
        super().__init__(
            'test_name',
            os.path.dirname(__file__),
            **kwargs
        )
        # test's specification
```

The base class' constructor needs two positional arguments that must be supplied by the user tests:
* the name of the test and
* its prefix.

The prefix of a regression test is normally the directory it resides in and it will be used in later phases for resolving relative paths for accessing the test's resources.

The rest of the regression pipeline stages are implemented as different methods by the base class `RegressionTest`.
Normally, a user test does not need to override them, unless it needs to modify the default behavior.
Even in this case though, a user test need not care about any of the phase implementation details, since it can delegate the actual implementation to the base class after or before its intervention.
We will show several examples for modifying the default behavior of the pipeline phases in this section.
A list of the actual `RegressionTest`'s methods that implement the different pipeline stages follows:
* `setup(self, system, environ, **job_opts)`:
  Implements the setup phase.
  The `system` and `environ` arguments refer to the current system partition and environment that the regression test will run.
  The `job_opts` arguments will be passed through the job scheduler backend.
  This is used by the front-end in special invocation of the regression suite, e.g., submit all regression test jobs in a specific reservation.
* `compile(self)`:
  Implements the compilation phase.
* `run(self)` and `wait(self)`:
  These two implement the run phase.
  The first call is asynchronous;
  it returns as soon as the associated job or process is submitted or created, respectively.
* `check_sanity(self)`:
  Implements the sanity checking phase.
* `check_performance(self)`:
  Implements the performance checking phase.
* `cleanup(self, remove_files=False)`:
  Cleans up the regression tests resources and unloads its environment.
  The `remove_files` flag controls whether the stage directory of the regression test should be removed or not.

As we shall see later in this section, user regression tests that override any of these methods usually do a minimal setup (e.g., setup compilation flags, adapt their internal state based on the current programming environment) and call the base class' corresponding method passing through all the arguments.

The following listing shows a complete user regression test that compiles a *Hello, World!* C program for different programming environments:

```python
import os

from reframe.core.checks import RegressionTest

class HelloWorldTest(RegressionTest):
    def __init__(self, **kwargs):
        super().__init__(
            'hello_world',
            os.path.dirname(__file__),
            **kwargs
        )
        self.descr = 'Hello World C Test'
        self.sourcepath = 'hello.c'
        self.valid_systems = [
            'daint:gpu',
            'daint:mc'
        ]
        self.valid_prog_environs = [
            'PrgEnv-cray',
            'PrgEnv-gnu',
            'PrgEnv-intel',
            'PrgEnv-pgi'
        ]
        self.sanity_patterns = {
            '-': {'Hello, World\!': []}
        }

def _get_checks(**kwargs):
    return [ HelloWorldTest(**kwargs) ]
```

After the base class' constructor is called with the boiler plate code we showed before, the specification of the test needs to be set.
The `RegressionTest` base class makes available a set of member variables that can be used to set up the test.
All these variables are dynamically type checked;
if they are assigned a value of different type, a runtime error will be raised and the regression test will be skipped from execution.

The *Hello, World!* example shown above shows a minimal set of variables that needs to be set for describing a test.
`descr` is an optional arbitrary textual description of the test that defaults to the test's name if not set.
`sourcepath` is a path to either a source file or a source directory.
By default, source paths are resolved against `'<testprefix>/src'`, where `testprefix` is stored in the `prefix` member variable.
If `sourcepath` refers to a file, it will be compiled picking the correct compiler based on its extension (C/C++/Fortran/CUDA). If it refers to a directory, ReFrame will invoke `make` inside that directory.

`valid_systems` and `valid_prog_environs` are the variables that basically enable a test to run on certain systems and programming environments.
They are both simple list of names.
The names need not to necessarily correspond to a configured system/partition or programming environment, in which case the test will be ignored (see [Configuration](/configure) on how a new systems and programming environments are configured).
The system name specification follows the syntax `<sysname>[:<partname>]`, i.e., you can either a specify a whole system or a specific partition in that system.
In our previous example, this test will run on both the *gpu* and *mc* partitions of Piz Daint.
If we specified simply *daint* in our example, then the above test would be eligible to run on any configured partition for that system.

Next, you need to define the `sanity_patterns` variable, which tells the framework what to look for in the standard output of the test to verify the sanity of the check.
We will cover the output parsing capabilities of ReFrame in detail in [Output Parsing and Performance Assesment](#output-parsing-and-performance-assessment).

Finally, each regression test file must provide the special method `_get_checks()`, which instantiates the user tests of the file.
This method is the entry point of the framework into the user tests and should return a list of `RegressionTest` instances.
From the framework's point of view, a regression test file is simply a Python module file that is loaded by the framework and has its `_get_checks()` method called.
This allows for maximum flexibility in writing regression tests, since a user can create his own hierarchies of tests and even test factories for generating sequences of related tests.
In fact, we have used this capability extensively while developing the Piz Daint's regression tests and has allowed us to considerably reduce code duplication and maintenance costs of regression tests.

# Types of Regression Checks

As mentioned earlier, a regression test may skip any of the above pipeline stages.
The ReFrame framework provides three basic types of regression tests:
* __Normal test__: This is the typical test that goes through all the phases of the pipeline.
* __Run-only test__: This test skips the compilation phase.
  This is quite a common type of test, especially when you want to test the functionality of installed software.
  Note that in this type of test the copying of the resources to the stage directory happens during the run phase and not during the compilation phase, as is the case of a normal test.
* __Compile-only test__: This test a special test that skips completely the run phase.
  However it makes available the standard output and standard error of the compilation, so that their output can be parsed during the check sanity phase in the same way that this happens for the other type of tests.

## Normal Regression test

This type of test implements all phases of the regression pipeline. The majority of regression tests at CSCS are of this type.

```python
class HelloTest(RegressionTest):
    def __init__(self, **kwargs):
        super().__init__(
            'test_name',
            os.path.dirname(__file__),
            **kwargs
        )
        # test's specification
```

## Run Only Regression Test

This type of test skips the compilation phase of the regression pipeline.

In order to make the test a run-only test, the test must inherit from `RunOnlyRegressionTest` as in the example below. Examples of this types of test are compiled application tests that are supported by the HPC support team.

```python
class MyRunOnlyTest(RunOnlyRegressionTest):
    def __init__(self, **kwargs):
        super().__init__(
            'test_name',
            os.path.dirname(__file__),
            **kwargs
        )
        # test's specification
```


## Compile Only Regression Test

This type of test skips the run phase of the regression pipeline and makes available the standard output and standard error of the compilation, so that their output can be parsed during the check sanity phase in the same way that this happens for the other type of tests.

In order to make the test a compile-only test, the test must inherit from `CompileOnlyRegressionTest` as in the example below. There a few case scenarios for this type of tests. At CSCS it is only used for tests where one wants to resolve the version of the linked libraries.

```python
class MyCompileOnlyTest(CompileOnlyRegressionTest):
    def __init__(self, **kwargs):
        super().__init__(
            'test_name',
            os.path.dirname(__file__),
            **kwargs
        )
        # test's specification
```

## Defining your Own Class of Regression Tests

An advantage of writing regression tests in a high-level language, such as Python, is that one can take advantage of features not present in classical shell scripting.
For example, one can create groups of related tests that share common characteristics and/or functionality by implementing them in a base class, from which all the related concrete tests inherit.
This eliminates unnecessary code duplication and reduces significantly the maintenance cost.
An example could be the implementation of system acceptance tests, where longer wall clock times may be required compared to the regular everyday production tests.
For such tests, one could define a base class, like in the example below, that would implement the longer wall clock time feature instead of modifying each job individually:
```python
class AcceptanceTest(RegressionTest):
    def __init__(self, **kwargs):
        ...
        self.time_limit = (24, 0, 0)

class HPLTest(AcceptanceTest):
        ...
```

# Select Systems and Programming Environments

Each test written with ReFrame should define the variables `valid_systems` and `valid_prog_environs`.
These variables allow the fine-grained control of in which systems and programming environments a given regression tests is allowed to run.
As the variable names suggest, the variable `valid_systems` defines the list of valid systems and the variable `valid_prog_environs` defines the list of valid programming environments.
The names defined inside these variables need not to necessarily correspond to a configured system/partition or programming environment, in which case the test will be ignored (see [Configuration](/configure) on how a new systems and programming environments are configured).
The system name specification follows the syntax `<sysname>[:<partname>]`, i.e., you can either a specify a whole system or a specific partition in that system.
```python
self.valid_systems = [
    'daint:gpu', # Piz Daint gpu virtual partition
    'dom'        # All Piz Dom's virtual partitions
]
self.valid_prog_environs = [
    'PrgEnv-cray',
    'PrgEnv-gnu',
    'PrgEnv-intel',
    'PrgEnv-pgi'
]
```
In this example, this test will run only on the *gpu* partitions of Piz Daint and in all virtual partitions of Piz Dom.

# Setting up Job Submission

ReFrame aims to be job scheduler agnostic and to support different job options using a unified interface.
To the regression check developer, the interface is manifested by a simple collection of member variables defined inside the regression check. The next section describes the supported job options.

## Job Options

Table below shows a listing of these variables and their interpretation in SLURM.

`RegressionTest`'s member variable | Interpreted SLURM option
--- | ---
`self.time_limit = (10, 20, 30)`  | `#SBATCH --time=10:20:30`
`self.use_multithreading = True`  | `#SBATCH --hint=multithread`
`self.use_multithreading = False` | `#SBATCH --hint=nomultithread`
`self.exclusive = False`          | `#SBATCH --exclusive`
`self.num_tasks=72`               | `#SBATCH --ntasks=72`
`self.num_tasks_per_node=36`      | `#SBATCH --ntasks-per-node=36`
`self.num_cpus_per_task=2`        | `#SBATCH --cpus-per-task=2`
`self.num_tasks_per_core=2`       | `#SBATCH --ntasks-per-core=2`
`self.num_tasks_per_socket=36`    | `#SBATCH --ntasks-per-socket=36`

These variables are converted internally by the framework to express the appropriate job options for a given scheduler.
Normally, these variables are set during the initialization phase of a regression test.
Since the system that the framework is running on is already known during the initialization phase of the regression test, it is possible to customize these variables based on the current system without the need of overwriting any method. An example is shown in the following listing:
```python
def __init__(self, **kwargs):
    ...
    if self.current_system.name == 'systemName':
        self.num_tasks = 72
    else:
        self.num_tasks = 192
```



## Additional Job Options

Even though the set of variables described on Table above are enough to accommodate most of the common regression scenarios, some regression tests, especially those related to a scheduler, may require additional job options.
Supporting all job options from all schedulers is a virtually impossible task.
Therefore ReFrame allows the definition of custom job options.
These options can be appended to the test's job descriptor during the test's setup phase.
In the following example, a memory limit is passed explicitly to the backend scheduler, here SLURM:
```python
class MyTest(RegressionTest):
    ...
    def setup(self, system, environ, **job_opts):
        super().setup(system, environ, **job_opts)
        self.job.options += [ '--mem=120000' ]
```
Note that the option is appended after the call to the superclass' `setup()` method, since this is responsible for initializing the job descriptor.
Keep in mind that adding custom job options tights the regression test to the scheduler making it less portable, unless proper action is taken.
Of course, if there is no need to support multiple schedulers, adding any job option becomes trivial as shown in the example above.

# Setting up the Environment

ReFrame allows the customization of the environment of the regression tests.
This can be achieved by loading and unloading environment modules and by defining environment variables.

## Loading Modules

Every regression test may define its required modules using the `self.modules` variable.
```python
self.modules = [
    'cudatoolkit',
    'cray-libsci_acc',
    'fftw/3.3.4.10'
]
```
These modules will be loaded during the test's setup phase after the programming environment and any other environment associated to the current system partition are loaded.
The test's environment setup is a three-step process.
The modules associated to the current system partition are loaded first, followed by the modules associated to the programming environment and, finally, the regression test's modules as described in its `self.modules` variable.
If there is any conflict between the listed modules and the currently loaded modules, ReFrame will automatically unload the conflicting ones.
The same sequence of module loads and unloads performed during the setup phase is generated in the job script that is submitted to the job scheduler.
Note that programming environments modules need not be listed in the `self.modules` variable, since they are defined inside ReFrame's configuration file and the framework takes automatically care of their loading during the test's setup phase.

Since the actual loading of environment modules happens during the setup phase of the regression test, it is important to define the `self.modules` list before calling the `RegressionTest`'s `setup()` method.
The common scenario is to define the list of modules in the initialization phase, but on certain occasions, the modules of a test might need to change depending on the programming environment.
In these situations, it is better to create a mapping between the module name and the programming environment and override the `setup()` method to set the `self.modules` according to the current programming environment.
Following is an actual example from CSCS' Score-P regression tests:
```python
def __init__(self, **kwargs):
    ...
    self.valid_prog_environs = [
        'PrgEnv-cray',
        'PrgEnv-gnu',
        'PrgEnv-intel',
        'PrgEnv-pgi'
    ]

    self.scorep_modules = {
        'PrgEnv-cray'  : 'Score-P/3.0-CrayCCE-2016.11',
        'PrgEnv-gnu'   : 'Score-P/3.0-CrayGNU-2016.11',
        'PrgEnv-intel' : 'Score-P/3.0-CrayIntel-2016.11',
        'PrgEnv-pgi'   : 'Score-P/3.0-CrayPGI-2016.11'
    }

def setup(self, system, environ, **job_opts):
    self.modules = [
        self.scorep_modules[ environ.name ]
    ]
    super().setup(system, environ, **job_opts)
```

## Environment Variables

In addition to custom modules, users can also define environment variables for their regression tests.
In this case, the variable `self.variables` is used, which is as a dictionary where the keys are the names of the environment variables and the values match the environment variables' values:
```python
self.variables = {
    'ENVVAR' : 'env_value'
}
```
This dictionary can be used, for example, to define the value of the `OMP_NUM_THREADS` environment variable.
In order to set it to the number of cpus per tasks of the regression test, one can set it as follows:
```python
self.variables = {
    'OMP_NUM_THREADS' : str(self.num_cpus_per_task)
}
```

# Customising the Compilation Phase

ReFrame supports the compilation of the source code associated with the test.
As discussed in the hello world example, if the provided source code is a single source file (defined by the member variable `self.sourcepath`), the language will be detected from its extension and the file will be compiled.
By default, the source files should be placed inside a special folder named `src/` inside the regression test's folder.
In the case of a single source file, the name of the generated executable is name of the regression test.
ReFrame passes no flags to the programming environment's compiler;
it is left to the writer of a regression test to specify any if needed.
This can be achieved by overriding the `compile()` method as shown in the example below:
```python
def compile(self):
    self.current_environ.cflags   = '-O3'
    self.current_environ.cxxflags = '-O3'
    self.current_environ.fflags   = '-O3'
    super().compile()
```
Note that it is not possible to specify the compilation flags during the initialization phase of the test, since the current programming environment is not set yet.

If the compilation flags depend on the programming environment, like for example the OpenMP flags for different compilers, the same trick as with the `self.modules` described above can be used, by defining the flag mapping during the initialization phase and using it during the compilation phase:
```python
def __init__(self, **kwargs):
    ...
    self.prgenv_flags = {
        'PrgEnv-cray'  : '-homp',
        'PrgEnv-gnu'   : '-fopenmp',
        'PrgEnv-intel' : '-openmp',
        'PrgEnv-pgi'   : '-mp'
    }

def compile(self):
    flag = self.prgenv_flags[
        self.current_environ.name
    ]
    self.current_environ.cflags   = flag
    self.current_environ.cxxflags = flag
    self.current_environ.fflags   = flag
    super().compile()
```
Of course, one could just differentiate inside the `compile()` method, but the approach shown above is cleaner and moves more information inside the test's specification.

If the test comprises multiple source files, a `Makefile` must be provided and `self.sourcepath` must refer to a directory (this is the default behavior if not specified at all).
ReFrame will issue the `make` command inside the source directory.
Note that in this case it is not possible for ReFrame to guess the executable's name, so this must be provided explicitly through the `self.executable` variable.
Additional options can be passed to the the `make` command and even non-standard makefiles may be used as it is demonstrated in the example below:
```python
def __init__(self, **kwargs):
    ...
    self.executable = './executable'

def compile(self):
    self.current_environ.cflags = '-O3'
    super().compile(makefile='build.mk', options="PREP='scorep'")
```

The generated compilation command in this case will be
```bash
make -C <stagedir> -f build.mk PREP='scorep' CC='cc' CXX='CC' FC='ftn' CFLAGS='' CXXFLAGS='' FFLAGS='' LDFLAGS=''
```
Finally, pre- and post-compilation steps can be added through special variables (e.g., a `configure` step may be needed before compilation), however, ReFrame is not designed to be an automatic compilation and deployment tool.

# Customising the Run of a Test

ReFrame offers several other options for customizing the behavior of regression tests.

## Executable Options

ReFrame allows a list of options to be passed to the regression check executable.
```python
def __init__(self, **kwargs):
    ...
    self.executable = './a.out'
    self.executable_opts = [
        '-i inputfile',
        '-o outputfile'
    ]
```
These options are passed to the executable, which will be invoked by the scheduler launcher.
In the above example, the executable will be launched as follows with the SLURM scheduler:
```bash
srun ./a.out -i inputfile -o outputfile
```

## Pre- and Post-run Commands

The framework allows the execution of additional commands before and/or after the scheduler launcher invocation.
This can be useful for invoking  pre- or post-processing tools.
We use this feature in our Score-P tests, where we need to print out and check the produced traces:
```python
def setup(self, system, environ, **job_opts):
    super().setup(system, environ, **job_opts)
    self.job.pre_run = [
        'ulimit -s unlimited'
    ]
    ...
    self.job.post_run = [
        'otf2-print traces.otf2'
    ]
```

## Changing the Regression Test Resources Path

ReFrame allows individual regressions tests to define a custom folder for their resources, different than the default `src/` described in [Folder Structure](/structure).
This is especially important for applications with a large number of input files or large input files, where these input files may need to be saved in a different filesystem due to several reasons, such as filesystem size, I/O performance, network configuration, backup policy etc.
The location of this folder can be changed be redefining the \shinline{self.sourcesdir} variable:
```python
def __init__(self, **kwargs):
    ...
    self.sourcesdir = '/apps/input/folders'
```

## Launcher Wrappers

In some cases, it is necessary to wrap the scheduler launcher call with another program.
This is the typical case with debuggers of distributed programs, e.g., `ddt`.
This can be achieved in ReFrame by changing the job launcher using the special `LauncherWrapper` object.
This object wraps a launcher with custom command:
```python
def __init__(self, **kwargs):
    ...
    self.ddt_options = '--offline'

def setup(self, system, environ, **job_opts):
    super().setup(system, environ, **job_opts)
    self.job.launcher = LauncherWrapper(self.job.launcher, 'ddt', self.ddt_options)
```

Note that this test remains portable across different job launchers.
If it runs on a system with native SLURM it will be translated to
```bash
ddt --offline srun ...
```
whereas if it run on a system with ALPS it will be translated to
```bash
ddt --offline aprun ...
```

## Custom Launchers

From version 2.5 onward, ReFrame permits the simple addition of custom scheduler launchers. 
A launcher is basically a program that sets up the distributed execution of another program. Example launchers are `srun` and `mpirun`.

Launchers in Reframe are instances of the `JobLauncher` class and are responsible for setting up the command line to execute a program distributedly. 
The command line is the concatenation of (a) the launcher executable (e.g. `mpirun`), (b) fixed launcher options (e.g. `-np <num_tasks>`), (c) user launcher options, (d) the application exectutable and (e) application options; for example: `mpirun -np <num_tasks> -hostfile myhostfile hostname -s`.

A launcher that invokes `mpirun` as in the above example be implemented as follows:

```python
class MpirunLauncher(JobLauncher):
    @property
    def executable(self):
        return 'mpirun'

    @property
    def fixed_options(self):
        return [ '-np %s' % self.job.num_tasks ]
```

While the definition of the property `executable` is obviously mandatory, the definition of `fixed_options` is optional; it defaults to no options. 
Note that each launcher has a job descriptor associated (`self.job`); the launcher class may use the job submission information contained there.

A custom launcher as the above defined `MpirunLauncher` may be used in a regression test as follows:

```python
class MpirunTest(RegressionTest):
    ...
    def setup(self, system, environ, **job_opts):
        super().setup(system, environ, **job_opts)
        self.job.launcher = MpirunLauncher(self.job)
```

ReFrame will provide a collection of custom launchers (added uppon request). 
Currently it provides only one: a launcher for the [VisIt visualisation software](https://visit.llnl.gov/). 
The VisIt launcher can be used in a regression test the same way as the above `MpirunLauncher`; only the following import statement is required to make it available:
```python
from regression.core.launchers import VisitLauncher

```

# Output Parsing and Performance Assessment

ReFrame provides a powerful mechanism for describing the patterns to look for inside the output and/or performance file of a test without the need to override the `check_sanity()` or `check_performance()` methods.
It allows you to search for multiple different patterns in different files and also associate callback functions for interpreting the matched values and for deciding on the validity of the match.

## Syntax
Both the `sanity_patterns` and `perf_patterns` follow the exact same syntax and the framework's parsing algorithm is the same as well; the only difference is that a reference value is looked for in the case of performance matching.
In the following, we will use `sanity_patterns` in our examples and we will elaborate only on the additional traits of the `perf_patterns` when necessary.

The syntax of the `sanity_patterns` is the following:
```python
self.sanity_patterns = {
  <filepatt> : {
    <pattern_with_tags> : [
      (
        <tag>,
        <conv>,
        <action>
      ),
      # rest of tags in pattern
    ],
    # more patterns to look for in file
  },
  # more files
}
```

The `<filepatt>` is any valid shell file pattern but it can also take special values for denoting standard output and error:
* `'&1'` or `'-'` for standard output and
* `'&2'` for standard error.

The `<pattern_with_tags>` is a Python regular expression with named groups.
For example, the regular expression `'(?P<lineno>\\d+): result: (?P<result>\\S+)'` matches a line starting with an integer number followed by `': result: '`, followed by a non-whitespace string.
The integer number is stored in match group named `lineno` and the second in a group named `result`.
These group names will be used as tags by the framework to invoke different actions on each match.
The regular expression pattern in the `sanity_patterns`' syntax is followed by a list of tuples that associate a tag with a conversion function (`<conv>`) and an action to be taken.
For each of the tags of the matched pattern, the framework will call `action(conv(tagvalue), reference)`.
The reference is set to `None` for sanity patterns checking, but for performance patterns it is looked up in a special dictionary of reference values (we will discuss this later in this section).
The `<action>` callable must return a boolean denoting whether the tag should be actually considered as matched or not.

If you are not interested in associating actions with tags, you can place an empty list instead.
In that case, a named group in the regular expression pattern is not needed either, as is the case of the *Hello, World* example, where the parsing mechanism behaves like a simple `grep` command invocation:
```python
self.sanity_patterns = {
    '-' : { 'Hello, World\!' : [] }
}
```

## End of File Pattern
To better support stateful parsing (see below), ReFrame also define a special regex pattern (`'\\e'`) that matches the end-of-file.
This pattern can only be associated with a callback function taking no arguments, which will be called after the processing of the file has finished.

## Output Scanning
The Figure below shows general concept of the algorithm used by the framework for matching the sanity and performance patterns.

![Output scanning](img/output-scanning.png)
__Figure: Output Parsing.__ The regression framework's algorithm for scanning sanity and performance patterns in the output of regression tests.
The algorithm here is a bit simplified, since it does not show the resolution of performance references.

The algorithm starts by expanding the file patterns inside `sanity_patterns`, and for each file it tries to match all the regex patterns associated with it.
As soon as a pattern is matched, it is marked and every of its associated tag values is converted and passed to the action callback along with its corresponding reference value (or just `None` for sanity checking).
If the action callback returns `True`, then the tag is marked as matched.
The process succeeds if all patterns and all tags of every file have been matched *at least once* and the end-of-file callback, if it has been provided, returns also `True`.

A more complex `sanity_patterns` example is shown below:
```python
self.sanity_patterns = {
  '-' : {
    'final result:\s+(?P<res>\d+\.?\d*)':
    [
      ('res', float,
       lambda value, **kwargs: \
         standard_threshold(
           value, (1., -1e-5, 1e-5)))
     ],
  }
}
```

This is an excerpt from an actual [OpenACC](http://www.openacc.org/) regression test from our suite.
The test program computes an axpy product and prints `final result: <number>`.
Since the output of different compilers may differ slightly, we need a float comparison.
Achieving that with the `sanity_patterns` is quite easy.
We tag the matched value with `res` and we then associate an action with this tag.
The action is a simple lambda function that checks if the value of the tag is `1.0e-5`.
`standard_threshold` is a function provided by the framework that checks if a value is between some tolerance limits.
Note that inside the lambda function `value` is already converted to a float number by the conversion callable `float`.


## Tag Resolution and Reference Lookup

The only difference between `sanity_patterns` and `perf_patterns` is that for the latter the framework looks up the tags in a special dictionary that holds reference values and picks up the correct one, whereas for `sanity_patterns`, the reference value is always set to `None`.
A representative reference value dictionary is shown below:
```python
self.reference = {
    'dom:gpu' : {
        'perf' : (258, None, 0.15)
    },
    'dom:mc' : {
        'perf' : (340, None, 0.15)
    },
    'daint:gpu' : {
        'perf' : (130, None, 0.15)
    },
    'daint:mc' : {
        'perf' : (135, None, 0.15)
    }
}
```

This is a real example of the reference value dictionary for our [CP2K](http://onlinelibrary.wiley.com/doi/10.1002/wcms.1159/full) regression test.
A reference value dictionary consists of sub-dictionaries per system, which in turn map tags to reference values.
A reference value can be anything that the action callback function could understand.
In this case the reference value required by the `standard_threshold` function is a 3-element tuple containing the actual reference value and lower and upper thresholds expressed in decimal fractions.

Although reference value dictionaries behave like normal Python dictionaries, they have an additional trait that makes them quite flexible:
they allow you to define scopes.
We call such dictionaries *scoped dictionaries*.
Currently, they are only used for holding the reference values, but we plan to expand their use in other API variables.
A scoped dictionary is a two-level dictionary, where the higher level defines the scope (or namespace) and the lower level holds the actual key/value mappings for a scope.
Scopes are defined by the outer level dictionary keys, which have the form `'s1:s2:...'` or `'*'` for denoting the global scope.
When you request a key from a scoped dictionary, you can prefix it with a scope, e.g., `'daint:mc:perf'`.
The last component of the fully qualified name is always the key to be looked up, here `'perf'`.
This key will be first looked up in the deepest scope, and if not found it will be looked up recursively in the parent scopes, until the global scope is reached.
If not found even there, a `KeyError` will be raised.
The reference value dictionary uses two scopes: the system and the system partition.
In the above example, we provide different reference values for different system partitions, but one could provide a single reference value per system.
Although the global scope `'*'` seems not to offer anything in a reference value dictionary (what would be the need to have a global reference for any system?), it is quite useful when we need stateful parsing for performance patterns.


## Stateful Parsing of Output

It is often the case with the output of a regression test that you cannot judge its outcome by just looking for single occurrences of patterns in the output file.
Consider the case that you have to count the lines of the output before deciding about success of the test or not.
You could also only care about the `n-th` occurrence of a pattern, in which case you would call the `standard_threshold` function for checking the performance outcome.
In such cases, you need to keep a state of the parsing process and defer the final decision until all the required information is gathered.
In a shell script world, you would achieve this by piping the `grep` output to an `awk` program that would take care of the state bookkeeping.

Thanks to the callback mechanism of the ReFrame's output parsing facility, you can define your own custom output parser that would hold the state of the parsing procedure.
The framework provides two entry points to any custom parser:
* the action callback function and
* the end-of-file callback

The action callback must be a function accepting at least two keyword arguments (`value` and `reference`), whereas the the eof callback need not have any named keyword argument.
Below is a concrete example of how you would count exactly 10 occurrences of the `'Hello, World\!'` pattern:
```python
class Counter:
    def __init__(self, max_patt):
        self.max_patt = max_patt
        self.count = 0

    def match_line(self, value,
                   reference, **kwargs):
        self.count += 1
        return True

    def match_eof(self, **kwargs):
        return self.count == max_patt

parser = Counter(10)
self.sanity_patterns = {
    '-' : {
        '(?P<line>Hello, World\!)' : [
            ('line',
             str, parser.match_line)
        ]
    },
    '\e' : parser.match_eof
}
```

Note that it is the end-of-file callback that actually decides on the final outcome of the sanity checking in this case.
If you wouldn't need to count the exact number of occurrences, but just a minimum number, you could then omit completely the `match_eof()` function and have the `match_line()` just return `self.count >= self.max_patt`.

The ability that the framework offers you to leverage the callback mechanism of sanity and performance output checking in order to perform stateful parsing is quite important, since it abstracts away the *boring* details of managing the output files, thus adding to the clarity of the regression test description.
Additionally, you may not even need to implement your own parser, since the framework provides a set of predefined parsers for common aggregate and reduction operations.
A parser is an object carrying a state and a callback function that will be called if a match is detected.
The default callback function is always returning `True`.
The parsers offer a simple API that it can be used as an action callback in `sanity_patterns` or `perf_patterns`:
* `on(**kwargs)`:
  Switches on the parser.
  By default, all parsers are in the *off* state, meaning that their matching functions will always return `False`.
* `off(**kwargs)`:
  Switches off the parser.
* `match(value, reference, **kwargs)`:
  This is the function that performs the state update and is to be called when a match is found.
  The `value` is the value of the match and `reference` is the reference value for the current system (or `None` for sanity checking).
  It returns `True` if the match is valid.
  All parsers determine the validity of a match in two stages.
  First, they check against their state (e.g., *is this the fifth match?*) and if this check is successful, then they call their callback function to finally determine the validity of the match.
  This allows validity checks of the form *the average performance of the first 100 steps must be within 10% of the reference value for this system*.
* `match_eof(**kwargs)`:
  This function is to be called as an eof handler.
  Again, here, the validity of the match is checked in two stages as in the `match()` method.
  This method clears also parser's state before returning.
* `clear(**kwargs)`:
  Clears the parser's state.

ReFrame offers the following parsers to the users:
* `StatefulParser`:
  This parser is very basic, storing only an on/off state.
  Its `match()` method simply delegates the decision to the parser's callback function, if the parser is on.
* `SingleOccurenceParser`:
  This parser checks for the `n-th` occurence of a pattern and calls its callback function if it's found.
* `CounterParser`:
  This parser counts the number of occurrences of a pattern and calls its callback function if a certain count is met.
  The parser can be configured to either count a minimum number of occurrences or an exact number.
* `UniqueOccurrencesParser`:
  This parser counts the unique occurrences of a pattern and calls its callback function if a certain count is met.
* `MaxParser`:
  This parser applies its callback function to the maximum value of its matched patterns.
* `MinParser`:
  This parser applies its callback function to the minimum value of its matched patterns.
* `SumParser`:
  This parser applies its callback function to the sum of the values of its matched patterns.
* `AverageParser`:
  This parser applies its callback function to the average of the values of its matched patterns.

The following listing shows an example usage of the `AverageParser` from the actual NAMD check used in CSCS' regression test suite for Piz Daint:
```python
self.parser = AverageParser(
    standard_threshold)
self.parser.on()
self.perf_patterns = {
    '-' : {
        'long_pattern (?P<days_ns>\S+) '
        'days/ns' : [
            ('days_ns', float,
             self.parser.match)
        ],
        '\e' : self.parser.match_eof,
    }
}
```


# Check Tagging

To facilitate the organization of regression tests inside a test suite, ReFrame allows to assign tags to regression tests.
You can later select specific tests to run based on their tags.
```python
    self.tags = { 'production',
                  'gpu' }
```

It is also possible to associate a list of persons that are responsible for maintaining a regression test.
This list will be printed in case of a test failure.
```python
self.maintainers = [ 'bob@a.com',
                     'john@a.com' ]
```

# Examples of Regression Tests

A list contaitning a collection of example regression tests that highlight some implementation details are can be found [here](/examples). The list sorted by feature should contain:
* Minimal Regression Test:
  * See the simple [hello world](/examples/#hello-world) check from the unit tests.
* Setting compilation flags:
  * Simple case, see [CUDA](/examples/#cuda-regression-tests) checks.
  * Based on the environment, see [OpenACC](/examples/#openacc-regression-tests) checks.
* Performance check using statefull parses:
  * See [CP2K](/examples/#application-regression-tests) check.
* Differentiate behavior based on the current system and programming environment:
  * See [OpenACC](/examples/#openacc-regression-tests) checks.
* Stateful output and/or performance parsing:
  * See [CP2K](/examples/#application-regression-tests) check.
* Compile-only checks:
  * See the [libsci_resolve](/examples/#compile-only-regression-tests) check.
