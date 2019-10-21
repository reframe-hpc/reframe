[![Build Status](https://travis-ci.org/eth-cscs/reframe.svg?branch=master)](https://travis-ci.org/eth-cscs/reframe)
[![Documentation Status](https://readthedocs.org/projects/reframe-hpc/badge/?version=latest)](https://reframe-hpc.readthedocs.io/en/latest/?badge=latest)
[![codecov.io](https://codecov.io/gh/eth-cscs/reframe/branch/master/graph/badge.svg)](https://codecov.io/github/eth-cscs/reframe)
[![PyPI version](https://badge.fury.io/py/ReFrame-HPC.svg)](https://badge.fury.io/py/ReFrame-HPC)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Slack](https://reframe-slack.herokuapp.com/badge.svg)](https://reframe-slack.herokuapp.com/)

# ReFrame

ReFrame is a framework for writing regression tests for HPC systems.
The goal of this framework is to abstract away the complexity of the interactions with the system, separating the logic of a regression test from the low-level details, which pertain to the system configuration and setup.
This allows users to write easily portable regression tests, focusing only on the functionality.

Regression tests in ReFrame are simple Python classes that specify the basic parameters of the test.
The framework will load the test and will send it down a well-defined pipeline that will take care of its execution.
The stages of this pipeline take care of all the system interaction details, such as programming environment switching, compilation, job submission, job status query, sanity checking and performance assessment.

Writing system regression tests in a high-level modern programming language, like Python, poses a great advantage in organizing and maintaining the tests.
Users can create their own test hierarchies, create test factories for generating multiple tests at the same time and also customize them in a simple and expressive way.


## Getting ReFrame

You may install ReFrame directly from [PyPI](https://pypi.org/project/ReFrame-HPC/) through `pip`:

```bash
pip install reframe-hpc
```

ReFrame will be available in your PATH:

```bash
reframe -V
```

Alternatively, and especially if you want to contribute back to the framework, you may clone this repository:

```bash
git clone https://github.com/eth-cscs/reframe.git
cd reframe
./bin/reframe -V
```

Finally, you may access all previous versions of ReFrame [here](https://github.com/eth-cscs/reframe/releases).


## Documentation

You may find the official documentation of the latest release and the current master in the following links:

- [Latest release](https://reframe-hpc.readthedocs.io/en/stable)
- [Current master](https://reframe-hpc.readthedocs.io)


### Building the documentation locally

You may build the documentation of the master locally either with Python 2 or Python 3.
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


## Contact

You can get in contact with the ReFrame community in the following ways:

### Mailing list

For keeping up with the latest news about ReFrame, posting questions and, generally getting in touch with other users and the developers, you may follow the mailing list: [reframe@sympa.cscs.ch](mailto:reframe@sympa.cscs.ch).

Only subscribers may send messages to the list.
To subscribe, please send an empty message to [reframe-subscribe@sympa.cscs.ch](mailto:reframe-subscribe@sympa.cscs.ch).

For unsubscribing, you may send an empty message to [reframe-unsubscribe@sympa.cscs.ch](mailto:reframe-unsubscribe@sympa.cscs.ch).

### Slack

You may also reach the community through Slack [here](https://reframe-slack.herokuapp.com).
