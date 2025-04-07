[![ReFrame Logo](https://raw.githubusercontent.com/reframe-hpc/reframe/master/docs/_static/img/reframe_logo-width400p.png)](https://github.com/reframe-hpc/reframe)<br/>
[![Build Status](https://github.com/reframe-hpc/reframe/workflows/ReFrame%20CI/badge.svg)](https://github.com/reframe-hpc/reframe/actions?query=workflow%3A%22ReFrame+CI%22)
[![Documentation Status](https://readthedocs.org/projects/reframe-hpc/badge/?version=latest)](https://reframe-hpc.readthedocs.io/en/latest/?badge=latest)
[![codecov.io](https://codecov.io/gh/reframe-hpc/reframe/branch/master/graph/badge.svg)](https://codecov.io/github/reframe-hpc/reframe)<br/>
![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/reframe-hpc/reframe?include_prereleases)
![GitHub commits since latest release](https://img.shields.io/github/commits-since/reframe-hpc/reframe/latest)
![GitHub contributors](https://img.shields.io/github/contributors-anon/reframe-hpc/reframe)<br/>
[![PyPI version](https://badge.fury.io/py/ReFrame-HPC.svg)](https://badge.fury.io/py/ReFrame-HPC)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/reframe-hpc)
[![Downloads](https://pepy.tech/badge/reframe-hpc)](https://pepy.tech/project/reframe-hpc)
[![Downloads](https://pepy.tech/badge/reframe-hpc/month)](https://pepy.tech/project/reframe-hpc)<br/>
[![Slack](https://badgen.net/badge/icon/slack?icon=slack&label)](https://join.slack.com/t/reframetalk/shared_invite/zt-1tar8s71w-At0tolJ~~zxT2oG_2Ly9sw)
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

ReFrame also supports storing the test results in a database allowing for later inspection, basic analytics and performance comparisons.

Please visit the project's documentation [page](https://reframe-hpc.readthedocs.io/) and [GitHub repository](https://github.com/reframe-hpc/reframe) for all the details!

## Contact

You can get in contact with the ReFrame community in the following ways:

### Slack

Please join the community's [Slack channel](https://reframe-slack.herokuapp.com) for keeping up with the latest news about ReFrame, posting questions and, generally getting in touch with other users and the developers.

## Contributing back

ReFrame is an open-source project and we welcome and encourage contributions!
Check out our Contribution Guide [here](https://github.com/reframe-hpc/reframe/wiki/contributing-to-reframe).

## Citing ReFrame

You can cite ReFrame in publications as follows:

> Vasileios Karakasis et al. “Enabling Continuous Testing of HPC Systems Using ReFrame”. In: *Tools and Techniques for High Performance Computing. HUST - Annual Workshop on HPC User Support Tools* (Denver, Colorado, USA, Nov. 17–18, 2019). Ed. by Guido Juckeland and Sunita Chandrasekaran. Vol. 1190. Communications in Computer and Information Science. Cham, Switzerland: Springer International Publishing, Mar. 2020, pp. 49–68. isbn: 978-3-030-44728-1. doi: 10.1007/978-3-030-44728-1_3.

The corresponding BibTeX entry is the following:

```bibtex
@InProceedings{karakasis20a,
  author     = {Karakasis, Vasileios and Manitaras, Theofilos and Rusu, Victor Holanda and
                Sarmiento-P{\'e}rez, Rafael and Bignamini, Christopher and Kraushaar, Matthias and
                Jocksch, Andreas and Omlin, Samuel and Peretti-Pezzi, Guilherme and
                Augusto, Jo{\~a}o P. S. C. and Friesen, Brian and He, Yun and Gerhardt, Lisa and
                Cook, Brandon and You, Zhi-Qiang and Khuvis, Samuel and Tomko, Karen},
  title      = {Enabling Continuous Testing of {HPC} Systems Using {ReFrame}},
  booktitle  = {Tools and Techniques for High Performance Computing},
  editor     = {Juckeland, Guido and Chandrasekaran, Sunita},
  year       = {2020},
  month      = mar,
  series     = {Communications in Computer and Information Science},
  volume     = {1190},
  pages      = {49--68},
  address    = {Cham, Switzerland},
  publisher  = {Springer International Publishing},
  doi        = {10.1007/978-3-030-44728-1_3},
  venue      = {Denver, Colorado, USA},
  eventdate  = {2019-11-17/2019-11-18},
  eventtitle = {{HUST} - Annual Workshop on {HPC} User Support Tools},
  isbn       = {978-3-030-44728-1},
  issn       = {1865-0937},
}
```
