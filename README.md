<a href="http://github.com/eth-cscs/reframe">
  <img src="docs/_static/img/reframe_logo-full.png" width=400>
</a>

[![Build Status](https://travis-ci.org/eth-cscs/reframe.svg?branch=master)](https://travis-ci.org/eth-cscs/reframe)
[![Documentation Status](https://readthedocs.org/projects/reframe-hpc/badge/?version=latest)](https://reframe-hpc.readthedocs.io/en/latest/?badge=latest)
[![codecov.io](https://codecov.io/gh/eth-cscs/reframe/branch/master/graph/badge.svg)](https://codecov.io/github/eth-cscs/reframe)<br/>
![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/eth-cscs/reframe?include_prereleases)
![GitHub commits since latest release](https://img.shields.io/github/commits-since/eth-cscs/reframe/latest)
![GitHub contributors](https://img.shields.io/github/contributors-anon/eth-cscs/reframe)<br/>
[![PyPI version](https://badge.fury.io/py/ReFrame-HPC.svg)](https://badge.fury.io/py/ReFrame-HPC)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/reframe-hpc)<br/>
[![Slack](https://reframe-slack.herokuapp.com/badge.svg)](https://reframe-slack.herokuapp.com/)<br/>
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

# ReFrame in a Nutshell

ReFrame is a framework for writing regression tests for HPC systems.
The goal of this framework is to abstract away the complexity of the interactions with the system, separating the logic of a regression test from the low-level details, which pertain to the system configuration and setup.
This allows users to write easily portable regression tests, focusing only on the functionality.

Regression tests in ReFrame are simple Python classes that specify the basic parameters of the test.
The framework will load the test and will send it down a well-defined pipeline that will take care of its execution.
The stages of this pipeline take care of all the system interaction details, such as programming environment switching, compilation, job submission, job status query, sanity checking and performance assessment.

Writing system regression tests in a high-level modern programming language, like Python, poses a great advantage in organizing and maintaining the tests.
Users can create their own test hierarchies, create test factories for generating multiple tests at the same time and also customize them in a simple and expressive way.


## Getting ReFrame

ReFrame is almost ready to run just after you clone it from Github.
All you need is Python 3.6 or above and to run its bootstrap script:

```bash
git clone https://github.com/eth-cscs/reframe.git
cd reframe
./bootstrap.sh
./bin/reframe -V
```

### Other installation ways

You can also install ReFrame through the following channels:

- Through [PyPI](https://pypi.org/project/ReFrame-HPC/):

  ```
  pip install reframe-hpc
  ```

- Through [Spack](https://spack.io/):

  ```
  spack install reframe
  ```

- Through [EasyBuild](https://easybuild.readthedocs.io/):

  ```
  eb easybuild/easyconfigs/r/ReFrame/ReFrame-VERSION.eb -r
  ```

Finally, you may access all previous versions of ReFrame [here](https://github.com/eth-cscs/reframe/releases).


## Documentation

You may find the official documentation of the latest release and the current master in the following links:

- [Latest release](https://reframe-hpc.readthedocs.io/en/stable)
- [Current master](https://reframe-hpc.readthedocs.io/en/latest)


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

### Mailing list

For keeping up with the latest news about ReFrame, posting questions and, generally getting in touch with other users and the developers, you may follow the mailing list: [reframe@sympa.cscs.ch](mailto:reframe@sympa.cscs.ch).

Only subscribers may send messages to the list.
To subscribe, please send an empty message to [reframe-subscribe@sympa.cscs.ch](mailto:reframe-subscribe@sympa.cscs.ch).

For unsubscribing, you may send an empty message to [reframe-unsubscribe@sympa.cscs.ch](mailto:reframe-unsubscribe@sympa.cscs.ch).

### Slack

You may also reach the community through Slack [here](https://reframe-slack.herokuapp.com).


## Contributing back

ReFrame is an open-source project and we welcome third-party contributions.
Check out our Contribution Guide [here](https://github.com/eth-cscs/reframe/wiki/contributing-to-reframe).
