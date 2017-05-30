This is the complete reference of the various fields and classes that a user of ReFrame may need when writing a regression test.

# `Environment` class

| Field name | Description | Default value | Type | Version |
| ---------- | ----------- | ------------- | ---- |---------|
| `name`     | Name of the environment. | Read-only* | non-whitespace string | 2.2 |


# `Job` class

| Field name | Description | Default value | Type | Version |
| ---------- | ----------- | ------------- | ---- | ------- |
| `launcher` | The job launcher associated to this job descriptor, e.g., "srun", "mpirun" etc. | Read-only* | `JobLauncher` | 2.2 |
| `post_run` | List of shell commands to emit after the actual job launch. | `[]` | list of strings | 2.2 |
| `pre_run`  | List of shell commands to emit before the actual job launch. | `[]` | list of strings | 2.2 |

# `JobLauncher` class

| Field name | Description | Default value | Type | Version |
| ---------- | ----------- | ------------- | ---- | ------- |
| `options`  | List of options to be passed to this job launcher. | `[]` | list of strings | 2.2 |

The only `JobLauncher` that users may use is the `LauncherWrapper`, which wraps an existing launcher by prepending a user program to its invocation.
This is useful for parallel debuggers.

```python
LauncherWrapper(launcher, wrapper_cmd, wrapper_options)
```

| Argument | Description | Type | Version |
| -------- | ----------- | ---- | ------- |
| `launcher` | The existing launcher to wrap. | `JobLauncher` | 2.2 |
| `wrapper_cmd` | The wrapping command. | string | 2.2 |
| `wrapper_options` | List of options to pass to the wrapping command. | list of strings | 2.2 |

Example usage below:

```python
def setup(self, system, environ, **job_opts):
    super().setup(system, environ, **job_opts)
    self.job.launcher = LauncherWrapper(self.job.launcher, 'ddt', [ '--offline' ])
```


# `ProgEnvironment` class

Fields additional to those defined by [Environment](#environment-class):

| Field name | Description | Default value | Type | Version |
| ---------- | ----------- | ------------- | ---- | ------- |
| `cc`       | The C compiler used by this environment. | `cc` | string | 2.2 |
| `cxx`      | The C++ compiler used by this environment. | `CC` | string | 2.2 |
| `ftn`      | The Fortran compiler used by this environment. | `ftn` | string | 2.2 |
| `cppflags` | Preprocessor flags. | `None` | string | 2.2 |
| `cflags`   | The C compiler flags. | `None` | string | 2.2 |
| `cxxflags` | The C++ compiler flags. | `None` | string | 2.2 |
| `fflags`   | The Fortran compiler flags. | `None` | string | 2.2 |
| `ldflags`  | Linker flags. | `None` | string | 2.2 |
| `propagate` | Propagate the compilation flags to the `make` invocation | `True` | boolean | 2.3 |

The correct compiler is picked by examining the source file extensions based on GCC's recognition [patterns](https://gcc.gnu.org/onlinedocs/gcc/Overall-Options.html).
Additionally, source files with the `.cu` extension are recognized as CUDA source files and are compiled with `nvcc`.

The generated compilation command has the following pattern (example for C, but other languages are treated the same way):
```
cc $cppflags $cflags -I<srcdir> <srcfile> -o <executable> $ldflags
```
* Notice that `cppflags` are passed to all languages.

In case that a whole directory is compiled and `propagate` is `True`, the generated command is the following:
```
make -C <srcdir> -f <makefile> CC=$cc CXX=$cxx FC=$ftn \
    CPPFLAGS=$cppflags CFLAGS=$cflags CXXFLAGS=$cxxflags FFLAGS=$fflags LDFLAGS=$ldflags
```
If `propagate` is set to `False`, no flags will be passed to `make`.

> Note: From version 2.3 onward all the compilation flags default to `None`.
> This has no difference to the previous versions when you compile a single file, but the behavior changes in case a directory is compiled and `make` is invoked.
> In this case, if a flag variable is set to `None`, it will not be passed over to the `make` invocation.
> Note that if a flag variable is set to the empty string, it *will* be passed to the `make` invocation.


# `RegressionTest` class

| Field name | Description | Default value | Type | Version |
| ---------- | ----------- | ------------- | ---- | ------- |
| `current_environ` | The programming environment that the regression test is currently executing with. This is set by the framework during the regression test's setup phase | Read-only* | `Environment` | 2.2 |
| `current_partition` | The system partition the regression test is currently executing on. This is set by the framework during the regression test's setup phase. | Read-only* | `SystemPartition` | 2.2 |
| `current_system` | The system the regression test is currently executing on. This is set by the framework during the regression test's initialization. | Read-only* | `System` | 2.2 |
| `descr`    | A detailed description of the test | `self.name` | string | 2.2 |
| `executable` | The name of the executable to be launched. | `./<name>` | string | 2.2 |
| `executable_opts` | List of options to be passed to the `executable`. | `[]` | list of strings | 2.2 |
| `job` | The job descriptor associated with this test. This is created during the setup phase of the test | Read-only* | `Job` | 2.2 |
| `keep_files` |  List of files to kept after the test finishes. By default, the framework saves the standard output and error and the (job) shell script generated. These files will be copied over to the framework's output directory during the cleanup phase. Directories are also accepted in this field. Non-absolute path names are resolved relative to the stage directory. | `[]` | list of strings | 2.2 |
| `local` | Always execute this test locally. | `False` | boolean | 2.2 |
| `maintainers` | List of people responsible for this test. When the test fails, this contact list will be printed out. | `[]` | list of strings | 2.2 |
| `modules` | List of modules to be loaded before running this test. These modules will be loaded during the test's setup phase. | `[]` | list of strings | 2.2 |
| `name`     | The name of the test | N/A | alphanumeric | 2.2 |
| `num_cpus_per_task` | Number of CPUs per task required by this test. Ignored if `None`. | `None` | integer | 2.2 |
| `num_gpus_per_node` | Number of GPUs per node required by this test. | `0` | integer | 2.2 |
| `num_tasks` | Number of tasks required by this test. | `1` | integer | 2.2 |
| `num_tasks_per_core` | Number of tasks per core required by this test. Ignored if `None`. | `None` | integer | 2.2 |
| `num_tasks_per_node` | Number of tasks per node required by this test. Ignored if `None`. | `None` | integer | 2.2 |
| `num_tasks_per_socket` | Number of tasks per socket required by this test. Ignored if `None`. | `None` | integer | 2.2 |
| `perf_patterns` | The set of performance patterns to look for for this test. Set to `None` for non-performance checks. | `None` | See [Output parsing](#output-parsing-and-performance-assessment) | 2.2 |
| `prefix` | The prefix directory of the test. | The directory of the test file | string | 2.2 |
| `postbuild_cmd` | List of shell commands to be executed after building. These commands are executed from inside the stage directory. | `[]` | list of strings | 2.2 |
| `prebuild_cmd` | List of shell commands to be executed before building. These commands are executed from inside the stage directory. | `[]` | list of strings | 2.2 |
| `readonly_files` | List of files or directories (relative to `sourcesdir`) that will be symlinked in the stage directory and not copied. You can use this variable to avoid copying to the stage directory very large files. | `[]` | list of strings | 2.3 |
| `reference` | The set of reference values for this test. | `{}` | scoped dictionary | 2.2 |
| `sanity_patterns` | The set of sanity patterns to look for for this test. | `None` | See [Output parsing](#output-parsing-and-performance-assessment) | 2.2 |
| `sourcepath` | The path to the source file or source directory of the test. If not absolute, it is resolved against the `sourcesdir` directory. If it refers to a regular file, this file will be compiled (its language will be automatically recognized) and the produced executable will be placed in the test's stage directory. If it refers to a directory, this will be copied to the test's stage directory and `make` will be invoked in that. | `''` | string | 2.2 |
| `sourcesdir` | The directory containing the test's resources. | `<test-prefix>/src/` | string | 2.2 |
| `stagedir` | The stage directory of the test. This is set during the test's setup phase. | Read-only* | string | 2.2 |
| `stderr` | The name of the file containing the standard error of the test. This is set during the test's setup phase. | Read-only* | string | 2.2 |
| `stdout` | The name of the file containing the standard output of the test. This is set during the test's setup phase. | Read-only* | string | 2.2 |
| `strict_check` | Specify whether this test allows a relaxed performance interpretation through the `--relax-performance-check` command-line option. If set to `False` and the aforementioned option is specified, the result of the performance checking phase will always be success. Note that the performance check is not skipped; it is just its result that is ignored. If set to `True`, the performance check for this test cannot be relaxed. |  `True` | boolean | 2.2 |
| `tags` | Set of tags to be associated with this regression test. This regression test can be selected from the frontend using any of these tags. | `{}` | set of strings | 2.2 |
| `time_limit` | Time limit for this test in the form of `(hh, mm, ss)`. | `(0, 10, 0)` | tuple of integers | 2.2 |
| `use_multithreading` | Specify whether this tests needs multithreading or not. | `False` | boolean | 2.2 |
| `valid_prog_environs` | Programming environmets supported by this test. | `[]` | list of strings | 2.2 |
| `valid_systems` | Systems supported by this test. The general syntax for systems is `<sysname>[:<partname]`. | `[]` | list of strings | 2.2 |
| `variables` | Environment variables to be set before running this test. These variables will be set during the test's setup phase. | `{}` | dictionary of string/string pairs | 2.2 |

# `System` class

| Field name | Description | Default value | Type | Version |
| ---------- | ----------- | ------------- | ---- | ------- |
| `descr`    | Detailed description of this system. | Read-only* | string | 2.2 |
| `name`     | Name of this system. | Read-only* | non-whitespace string | 2.2 |


# `SystemPartition` class

| Field name | Description | Default value | Type | Version |
| ---------- | ----------- | ------------- | ---- | ------- |
| `descr`    | Description of this partition. | Read-only* | string | 2.2 |
| `name`     | The name of this partition. | Read-only* | non-whitespace string | 2.2 |
| `scheduler` | Name of the scheduler backend associated with this partition. | Read-only* | non-whitespace string | 2.2 |



\* _Fields marked as "read-only" are not meant to be assigned by the user tests, although this policy is currently not enforced._
