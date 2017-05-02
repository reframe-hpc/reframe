# Writing Regression Checks

The regression pipeline described in [here](/pipeline) is implemented in a base class called `RegressionTest`, from which all user regression tests must eventually inherit.
There exist also base classes (inheriting also from `RegressionTest`) that implement the special regression test types described in Section~\ref{sec:reg-test-types}.
A user test may inherit from any of the base regression tests depending on the type of check (*normal*, *run-only* or *compile-only*).

The initialization phase of a regression test is implemented in the test's constructor, i.e., the \pyinline{__init__()} method of the regression test class.
The constructor of a user regression test is only required to allow keyword arguments to be passed to it.
These are needed to initialize the base \pyinline{RegressionTest}.
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
Due to space limitations, we will not go into all of the `RegressionTest`'s member variables.
We will only discuss the most important ones that come up in our examples.

The *Hello, World!* example shown above shows a minimal set of variables that needs to be set for describing a test.
`descr` is an optional arbitrary textual description of the test that defaults to the test's name if not set.
`sourcepath` is a path to either a source file or a source directory.
By default, source paths are resolved against `'<testprefix>/src'`, where `testprefix` is stored in the `prefix` member variable.
If `sourcepath` refers to a file, it will be compiled picking the correct compiler based on its extension (C/C++/Fortran/CUDA).
If it refers to a directory, ReFrame will invoke `make` inside that directory.
`valid_systems` and `valid_prog_environs` are the variables that basically enable a test to run on certain systems and programming environments.
They are both simple list of names.
The names need not necessarily correspond to a configured system/partition or programming environment, in which case the test will be ignored (see Section~\ref{sec:reg-configuring-new-site} on how a new systems and programming environments are configured).
The system name specification follows the syntax `<sysname>[:<partname>]`, i.e., you can either a specify a whole system or a specific partition in that system.
In our example, this test will run on both the *gpu* and *mc* partitions of Daint.
If we specified simply *daint* in our example, then the above test would be eligible to run on any configured partition for that system.
Next, you need to define the `sanity_patterns` variable, which tells the framework what to look for in the standard output of the test to verify the sanity of the check.
We will cover the output parsing capabilities of ReFrame in detail in Section~\ref{sec:reg-output-parsing}.

Finally, each regression test file must provide the special method `_get_checks()`, which instantiates the user tests of the file.
This method is the entry point of the framework into the user tests and should return a list of `RegressionTest` instances.
From the framework's point of view, a regression test file is simply a Python module file that is loaded by the framework and has its `_get_checks()` method called.
This allows for maximum flexibility in writing regression tests, since a user can create his own hierarchies of tests and even test factories for generating sequences of related tests.
In fact, we have used this capability extensively while developing the Piz Daint's regression tests and has allowed us to considerably reduce code duplication and maintenance costs of regression tests.


# Setting up job submission

\RegressionName aims to be job scheduler agnostic and to support different job options using a unified interface.
To the regression check developer, the interface is manifested by a simple collection of member variables defined inside the regression check.
These variables are converted internally by the framework to express the appropriate job options for a given scheduler.
Table~\ref{tab:job-options} shows a listing of these variables and their interpretation in SLURM.
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


Regression checks' member variables related to job options and their SLURM scheduler associated options

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

Even though the set of variables described on Table~\ref{tab:job-options} are enough to accommodate most of the common regression scenarios, some regression tests, especially those related to a scheduler, may require additional job options.
Supporting all job options from all schedulers is a virtually impossible task.
Therefore \RegressionName allows the definition of custom job options.
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

# Setting up the environment

ReFrame allows the customization of the environment of the regression tests.
This can be achieved by loading and unloading environment modules and by defining environment variables.
Every regression test may define its required modules using the `self.modules` variable.
```python
self.modules = [ 
    'cudatoolkit',
    'cray-libsci_acc',
    'fftw/3.3.4.10' 
]
```
These modules will be loaded during the test's setup phase after the programming environment and any other environment associated to the current system partition are loaded.
Recall from section~\ref{sec:setup-phase} that the test's environment setup is a three-step process.
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

# Customising the compilation phase

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
make -C <stagedir> -f build.mk PREP='scorep' CC='cc' CXX='CC' FC='ftn' CFLAGS='' \CXXFLAGS='' FFLAGS='' LDFLAGS=''
```
Finally, pre- and post-compilation steps can be added through special variables (e.g., a `configure` step may be needed before compilation), however, ReFrame is not designed to be an automatic compilation and deployment tool.

# Customising the run of a test

ReFrame offers several other options for customizing the behavior of regression tests.

## Executable options

ReFrame allows a list of options to be passed to regression check executable.
```python
def __init__(self, **kwargs):
    ...
    self.executable = './a.out'
    self.executable_opts = [
        '-i inputfile',
        '-o outputfile'
    ]
```
These options are passed to the executable, which will be invoked but the scheduler launcher.
In the above example, the executable will be launched as follows with the SLURM scheduler:
```bash
srun ./a.out -i inputfile -o outputfile
```

## Pre- and post-run commands

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

## Changing the regression test resources path

ReFrame allows individual regressions tests to define a custom folder for their resources, different than the default `src/` described in Section~\ref{sec:reg-folder-structure}.
This is especially important for applications with a large number of input files or large input files, where these input files may need to be saved in a different filesystem due to several reasons, such as filesystem size, I/O performance, network configuration, backup policy etc.
The location of this folder can be changed be redefining the \shinline{self.sourcesdir} variable:
```python
def __init__(self, **kwargs):
    ...
    self.sourcesdir = '/apps/input/folders'
```

## Launcher wrappers

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
```python
ddt --offline srun ...
```
whereas if it run on a system with ALPS it will be translated to
```python
ddt --offline aprun ...
```