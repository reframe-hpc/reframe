[![ReFrame Logo](https://raw.githubusercontent.com/eth-cscs/reframe/master/docs/_static/img/reframe_logo-width400p.png)](https://github.com/eth-cscs/reframe)<br/>
[![Build Status](https://github.com/eth-cscs/reframe/workflows/ReFrame%20CI/badge.svg)](https://github.com/eth-cscs/reframe/actions?query=workflow%3A%22ReFrame+CI%22)
[![Documentation Status](https://readthedocs.org/projects/reframe-hpc/badge/?version=latest)](https://reframe-hpc.readthedocs.io/en/latest/?badge=latest)
[![codecov.io](https://codecov.io/gh/eth-cscs/reframe/branch/master/graph/badge.svg)](https://codecov.io/github/eth-cscs/reframe)<br/>
![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/eth-cscs/reframe?include_prereleases)
![GitHub commits since latest release](https://img.shields.io/github/commits-since/eth-cscs/reframe/latest)
![GitHub contributors](https://img.shields.io/github/contributors-anon/eth-cscs/reframe)<br/>
[![PyPI version](https://badge.fury.io/py/ReFrame-HPC.svg)](https://badge.fury.io/py/ReFrame-HPC)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/reframe-hpc)<br/>
[![Slack](https://reframe-slack.herokuapp.com/badge.svg)](https://reframe-slack.herokuapp.com/)<br/>
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![DOI](https://zenodo.org/badge/89384186.svg)](https://zenodo.org/badge/latestdoi/89384186)<br/>
[![Twitter Follow](https://img.shields.io/twitter/follow/ReFrameHPC?style=social)](https://twitter.com/ReFrameHPC)

# ReFrame in a Nutshell

ReFrame is a framework for writing regression tests for HPC systems.
The goal of this framework is to abstract away the complexity of the interactions with the system, separating the logic of a regression test from the low-level details, which pertain to the system configuration and setup.
This allows users to write easily portable regression tests, focusing only on the functionality.

Regression tests in ReFrame are simple Python classes that specify the basic parameters of the test.
The framework will load the test and will send it down a well-defined pipeline that will take care of its execution.
The stages of this pipeline take care of all the system interaction details, such as programming environment switching, compilation, job submission, job status query, sanity checking and performance assessment.

Writing system regression tests in a high-level modern programming language, like Python, poses a great advantage in organizing and maintaining the tests.
Users can create their own test hierarchies, create test factories for generating multiple tests at the same time and also customize them in a simple and expressive way.

Please visit the project's documentation [page](https://reframe-hpc.readthedocs.io/) for all the details!


## Installation

ReFrame is fairly easy to install.
All you need is Python 3.6 or above and to run its bootstrap script:

```bash
git clone https://github.com/eth-cscs/reframe.git
cd reframe
./bootstrap.sh
./bin/reframe -V
```

If you want a specific release, please refer to the documentation [page](https://reframe-hpc.readthedocs.io/en/stable/started.html).


### Running the unit tests

You can optionally run the framework's unit tests with the following command:

```bash
./test_reframe.py -v
```

NOTE: Unit tests require a functional C compiler, available through the `cc` command, that is also able to recognize the ``-O2`` option.
The [GNU Make](https://www.gnu.org/software/make/) build tool is also needed.


### Building the documentation locally

You may build the documentation of the master manually as follows:

```
./bootstrap.sh +docs
```

For viewing it, you may do the following:

```
cd docs/html
python3 -m http.server
```

The documentation is now up on [localhost:8000](http://localhost:8000), where you can navigate with your browser.


## Examples of Regression Tests

In the `cscs-checks/` folder, you can find realistic regression tests used for the CSCS systems that you can reuse and adapt to your system.
Notice that these tests are published as examples and may not run as-is in your system.
However, they can serve as a very good starting point for implementing your system tests in ReFrame.


## Contact

You can get in contact with the ReFrame community in the following ways:

### Slack

Please join the community's [Slack channel](https://reframe-slack.herokuapp.com) for keeping up with the latest news about ReFrame, posting questions and, generally getting in touch with other users and the developers.

### Mailing list

You may also [subscribe](mailto:reframe-subscribe@sympa.cscs.ch) to the [mailing list](mailto:reframe@sympa.cscs.ch).
Only subscribers can send messages to the list.
For unsubscribing, you may send an empty message [here](mailto:reframe-unsubscribe@sympa.cscs.ch).


## Contributing back

ReFrame is an open-source project and we welcome and encourage contributions!
Check out our Contribution Guide [here](https://github.com/eth-cscs/reframe/wiki/contributing-to-reframe).
