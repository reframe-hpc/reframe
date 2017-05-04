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

# Run unittest


# Configure your site

## Define your system

## Define your programming environments

