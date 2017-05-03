# ReFrame

ReFrame is a new framework for writing regression tests for HPC systems.
The goal of this framework is to abstract away the complexity of the interactions with the system, separating the logic of a regression test from the low-level details, which pertain to the system configuration and setup.
This allows users to write easily portable regression tests, focusing only on the functionality.

Regression tests in ReFrame are simple Python classes that specify the basic parameters of the test.
The framework will load the test and will send it down a well-defined pipeline that will take care of its execution.
The stages of this pipeline take care of all the system interaction details, such as programming environment switching, compilation, job submission, job status query, sanity checking and performance assessment.

Writing system regression tests in a high-level modern programming language, like Python, poses a great advantage in organizing and maintaining the tests.
Users can create their own test hierarchies, create test factories for generating multiple tests at the same time and also customize them in a simple and expressive way.

[Learn more](/about)

# Getting the framework

To get the latest development version of the framework, you should clone it directly from CSCS' internal git repository.
```bash
git clone https://github.com/eth-cscs/reframe.git
```

<!--Alternatively you can get a specific stable version of the framework by downloading it from [here](https://madra.cscs.ch/scs/PyRegression/tags).-->

A set of regression checks come bundled with the framework.
These tests can be found under the `checks/` directory organized in several subdirectories.

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

# Citing the Framework

If you use the framework please cite: [Reference of CUG paper](/)
