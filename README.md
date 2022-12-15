<div align="left">
 <img src="docs/_static/img/reframe_logo-width400p.png#gh-light-mode-only" width="400px">
 <img src="docs/_static/img/reframe-logo-dark-bg.png#gh-dark-mode-only" width="400px">
</div>

[![Build Status](https://github.com/reframe-hpc/reframe/workflows/ReFrame%20CI/badge.svg)](https://github.com/reframe-hpc/reframe/actions?query=workflow%3A%22ReFrame+CI%22)
[![Documentation Status](https://readthedocs.org/projects/reframe-hpc/badge/?version=latest)](https://reframe-hpc.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/reframe-hpc/reframe/branch/master/graph/badge.svg)](https://codecov.io/gh/reframe-hpc/reframe)<br/>
![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/reframe-hpc/reframe?include_prereleases)
![GitHub commits since latest release](https://img.shields.io/github/commits-since/reframe-hpc/reframe/latest)
![GitHub contributors](https://img.shields.io/github/contributors-anon/reframe-hpc/reframe)<br/>
[![PyPI version](https://badge.fury.io/py/ReFrame-HPC.svg)](https://badge.fury.io/py/ReFrame-HPC)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/reframe-hpc)
[![Downloads](https://pepy.tech/badge/reframe-hpc)](https://pepy.tech/project/reframe-hpc)
[![Downloads](https://pepy.tech/badge/reframe-hpc/month)](https://pepy.tech/project/reframe-hpc)<br/>
[![Slack](https://reframe-slack.herokuapp.com/badge.svg)](https://reframe-slack.herokuapp.com/)<br/>
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![DOI](https://zenodo.org/badge/89384186.svg)](https://zenodo.org/badge/latestdoi/89384186)<br/>
[![Twitter Follow](https://img.shields.io/twitter/follow/ReFrameHPC?style=social)](https://twitter.com/ReFrameHPC)

# ReFrame in a Nutshell

ReFrame is a powerful framework for writing system regression tests and benchmarks, specifically targeted to HPC systems.
The goal of the framework is to abstract away the complexity of the interactions with the system, separating the logic of a test from the low-level details, which pertain to the system configuration and setup.
This allows users to write portable tests in a declarative way that describes only the test's functionality.

Tests in ReFrame are simple Python classes that specify the basic variables and parameters of the test.
ReFrame offers an intuitive and very powerful syntax that allows users to create test libraries, test factories, as well as complete test workflows using other tests as fixtures.
ReFrame will load the tests and send them down a well-defined pipeline that will execute them in parallel.
The stages of this pipeline take care of all the system interaction details, such as programming environment switching, compilation, job submission, job status query, sanity checking and performance assessment.

Please visit the project's documentation [page](https://reframe-hpc.readthedocs.io/) for all the details!


## Installation

ReFrame is fairly easy to install.
All you need is Python 3.6 or above and to run its bootstrap script:

```bash
git clone https://github.com/reframe-hpc/reframe.git
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

NOTE: Unit tests require a POSIX-compliant C compiler (available through the `cc` command), as well as the `make` utility.

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


## Test library

The framework comes with a library of tests that users can either run them from the command line directly or extend them and fine tune them for their systems. See [here](https://reframe-hpc.readthedocs.io/en/stable/hpctestlib.html) for more details.


## Test examples

You can find examples of real tests under the ReFrame HPC [community Github page](https://github.com/reframe-hpc).
The most complete suite of tests currently publicly available is that of [CSCS](https://cscs.ch/), which you can also find [here](https://github.com/eth-cscs/cscs-reframe-tests).
You can use those tests as a starting point for implementing your own tests.


## Contact

You can get in contact with the ReFrame community in the following ways:

### Slack

Please join the community's [Slack channel](https://reframe-slack.herokuapp.com) for keeping up with the latest news about ReFrame, posting questions and, generally getting in touch with other users and the developers.

## Contributing back

ReFrame is an open-source project and we welcome and encourage contributions!
Check out our Contribution Guide [here](https://github.com/reframe-hpc/reframe/wiki/contributing-to-reframe).
