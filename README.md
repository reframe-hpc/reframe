[![Build Status](https://travis-ci.org/eth-cscs/reframe.svg?branch=master)](https://travis-ci.org/eth-cscs/reframe) [![codecov.io](https://codecov.io/gh/eth-cscs/reframe/branch/master/graph/badge.svg)](https://codecov.io/github/eth-cscs/reframe)

# ReFrame

ReFrame is a new framework for writing regression tests for HPC systems.
The goal of this framework is to abstract away the complexity of the interactions with the system, separating the logic of a regression test from the low-level details, which pertain to the system configuration and setup.
This allows users to write easily portable regression tests, focusing only on the functionality.

Regression tests in ReFrame are simple Python classes that specify the basic parameters of the test.
The framework will load the test and will send it down a well-defined pipeline that will take care of its execution.
The stages of this pipeline take care of all the system interaction details, such as programming environment switching, compilation, job submission, job status query, sanity checking and performance assessment.

Writing system regression tests in a high-level modern programming language, like Python, poses a great advantage in organizing and maintaining the tests.
Users can create their own test hierarchies, create test factories for generating multiple tests at the same time and also customize them in a simple and expressive way.


## Documentation

The official documentation is maintained [here](https://eth-cscs.github.io/reframe/index.html).
It corresponds to the [latest](https://github.com/eth-cscs/reframe/releases/latest) stable release and not to the current status of the `master`.

### Building the documentation from master

You may build the documentation of the master either with Python 2 or Python 3 (<= 3.5).
Here is how to do it:

```
pip install -r docs/requirements.txt
make -C docs latest
```

For viewing it, you may do the following:

```
cd docs/html
python -m http.server # or python -m SimpleHTTPServer for Python 2
```

The documentation is now up on [localhost:8000](http://localhost:8000), where you can navigate with your browser.


## Examples of Regression Tests

In the `cscs-checks/` folder, you can find realistic regression tests used for the CSCS systems that you can reuse and adapt to your system.
Notice that these tests are published as examples and may not run as-is in your system.
However, they can serve as a very good starting point for implementing your system tests in ReFrame.
