# Requirements
* A recent version of Python 3 (>= 3.5).
  Python 2 is not supported.

* A functional modules software management environment with Python bindings.
  The framework currently supports only the traditional [modules](http://modules.sourceforge.net/) software management environment.
  The following need to be present or functional:
  * `MODULESHOME` variable must be set, otherwise the framework will not load at all.
  * `modulecmd python` must be supported, otherwise the framework will not load at all.

* A functional job submission management system.
  Currently only Slurm is supported with either a native Slurm job launcher or the Cray ALPS launcher.
  Please note that the Cray ALPS launcher is not thoroughly tested.
  Slurm accounting storage (`sacct` command) is required.

You are advised to run the [unit tests](framework#unit-tests) of the framework after installing it on a new system to make sure that everything works fine.

# Getting the framework

To get the latest development version of the framework, you should clone it directly from CSCS' internal git repository.
```bash
git clone https://github.com/eth-cscs/reframe.git
```

<!--Alternatively you can get a specific stable version of the framework by downloading it from [here](https://madra.cscs.ch/scs/PyRegression/tags).-->

A set of regression checks come bundled with the framework.
These tests can be found under the `checks/` directory organized in several subdirectories.

# Running the unit tests

As soon as you have installed and configured your systems, it is important to run the unit tests of the framework to make sure that everything is set up correctly.

```bash
./test_reframe.py -s -v -A 'not alps'
```

The output should look like the following:

```bash
test_compile_only_failure (unittests.test_checks.TestRegression) ... ok
test_compile_only_warning (unittests.test_checks.TestRegression) ... ok
test_environ_setup (unittests.test_checks.TestRegression) ... ok
test_hellocheck (unittests.test_checks.TestRegression) ... ok
test_hellocheck_local (unittests.test_checks.TestRegression) ... ok
test_hellocheck_make (unittests.test_checks.TestRegression) ... ok
...
test_copytree (unittests.test_utility.TestOSTools) ... ok
test_grep (unittests.test_utility.TestOSTools) ... ok
test_inpath (unittests.test_utility.TestOSTools) ... ok
test_subdirs (unittests.test_utility.TestOSTools) ... ok

----------------------------------------------------------------------
Ran 185 tests in 72.953s

OK
```

This will run all the unit tests not related to the ALPS backend.
If your system uses a submission mechanism that is not supported from the framework, you should skip all the unit tests that test job submission.
You can achieve this by doing the following:
```bash
./test_reframe.py -s -v -A 'not sumbit'
```

## Unit test tags

Some unit tests are associated with attributes that you could use for selecting them.
This is useful for development when you don't want to rerun the whole suite every time or when you just want to test a specific aspect of the framework.
The following attributes are now supported for the unit tests: `alps`, `nativeslurm`, `submit` and `slow`.
The first two are self-descriptive, the `submit` unit tests try to submit to a job submission queue, while `slow` are tests that my take some time to complete (e.g., job submission tests).
The `-A` option takes any valid python expression, so that for example you can run all the quick tests that do not use a submission queue as follows:

```bash
./test_reframe.py -s -v -A 'not sumbit and not slow'
```

# Where to go from here

ReFrame is available with only example regression checks.
You will need to read the sections [Configure your site](/configure), [Pipeline](/pipeline) and [Writing checks](/writing_checks) in order to write your own regression tests.

If you are just looking on how to invoked the regression test written with ReFrame, please look at [Running](/running) section.



