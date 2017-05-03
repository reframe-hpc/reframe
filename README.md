# PyRegression

# Documentation

* Official documentation will be maintaned on the [Wiki](https://madra.cscs.ch/scs/PyRegression/wikis/home)

# How to get the new framework

To get the new framework simply check out:

`git clone https://madra.cscs.ch/scs/PyRegression.git`


# Reframe

ReFrame is a python framework developed at [CSCS](http://www.cscs.ch) to facilitate the writing of regression tests that check the sanity of HPC systems.

Its main goal is to allow users to write their own regression tests without having to deal with all the details of setting up the environment for the test, quering the status of their job, managing the output of the job and looking for sanity and/or performance results.
Users should be concerned only about the logical requirements of their tests.
This allows users' regression checks to be maintained and adapted to new systems easily.

The user describes his test in a simple Python class and the framework takes care of all the details of the low-level interaction with the system.
Although the user still has to program in Python new checks, the framework is structured such that only a basic knowledge of Python is required to write a regression test, which will be able to run out-of-the-box on a variety of systems and programming environments.
In the future, we plan to allow users to describe their tests in a more abstract way using configuration files that wouldn't require any programming skills at all.
Of course, the Python interface will still be available for more advanced usage.

It can also be used as a continuous integration test

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
