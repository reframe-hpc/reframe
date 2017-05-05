# Requirements
* A recent version of Python 3 (>= 3.5).
  Python 2 is not supported.
* A functional modules software management environment with Python bindings.
  The framework currently supports only the traditional [modules](http://modules.sourceforge.net/) software management environment.
  The following need to be present or functional:
  * `MODULESHOME` variable must be set, otherwise the framework will not load at all.
  * `modulecmd python` must be supported, otherwise the framework will not load at all.

## Optional

* The [nose](https://pypi.python.org/pypi/nose) Python module must be installed if you want to run the unit tests of the framework.
* If you want to use the framework for launching tests on a cluster, a functional job submission management system is required.
  Currently only Slurm is supported with either a native Slurm job launcher or the Cray ALPS launcher.
  * In the case of Slurm, job accounting storage (`sacct` command) is required to be functional.
  * Please note that the Cray ALPS launcher is not thoroughly tested.

You are advised to run the [unit tests](#running-the-unit-tests) of the framework after installing it on a new system to make sure that everything works fine.

# Getting the framework

To get the latest stable version of the framework, you can just clone it from the [github](https://github.com/eth-cscs/reframe) project page.
```bash
git clone https://github.com/eth-cscs/reframe.git
```

Alternatively, you can pick a previous stable version by downloading it from the previous [releases](https://github.com/eth-cscs/reframe/releases) section.

# Running the unit tests

After you have downloaded the framework, it is important to run the unit tests of to make sure that everything is set up correctly:

```bash
./test_reframe.py -v
```

The output should look like the following:

```bash
test_check_failure (unittests.test_cli.TestFrontend) ... ok
test_check_sanity_failure (unittests.test_cli.TestFrontend) ... ok
test_check_submit_success (unittests.test_cli.TestFrontend) ... SKIP: job submission not supported
test_check_success (unittests.test_cli.TestFrontend) ... ok
test_checkpath_recursion (unittests.test_cli.TestFrontend) ... ok
test_custom_performance_check_failure (unittests.test_cli.TestFrontend) ... ok
...
test_grep (unittests.test_utility.TestOSTools) ... ok
test_inpath (unittests.test_utility.TestOSTools) ... ok
test_subdirs (unittests.test_utility.TestOSTools) ... ok
test_always_true (unittests.test_utility.TestUtilityFunctions) ... ok
test_standard_threshold (unittests.test_utility.TestUtilityFunctions) ... ok

----------------------------------------------------------------------
Ran 190 tests in 13.254s

OK (SKIP=7)
```

You will notice in the output that all the job submission related tests have been skipped.
The test suite detects if the current system has a job submission system and is configured in ReFrame (see [Configure your site](/configure)) and it will skip all the unsupported unit tests.
As soon as you configure ReFrame for your system, you can rerun the test suite to check that job submission unit tests are passed, too.
Note here, that some unit tests may still be skipped depending on the configured job submission system.
For example, the Slurm+ALPS tests will also be skipped on a system configured with native SLURM.


# Where to go from here

The next step from here is to setup and configure your site, so that ReFrame can automatically recognize it and submit jobs to it.
Please refer to the [Configure your site](/configure) page on how to do that.

Before starting implementing a regression test, you should go through the [Regression pipeline](/pipeline) section to understand the mechanism that ReFrame uses to run the regression tests.

To learn how to invoke the ReFrame command-line interface for running your tests, please refer to the [Running ReFrame](/running) section.

Finally, ReFrame ships with a set of example regression tests that lie in the `examples/` directory.
These examples are slightly simplified examples of real regression tests we run at CSCS.
You can use them as a starting point for your tests. For a small subset of those examples, please refer to the [examples section](/examples).
