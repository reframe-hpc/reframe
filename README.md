# ReFrame

ReFrame is a new framework for writing regression tests for HPC systems.
The goal of this framework is to abstract away the complexity of the interactions with the system, separating the logic of a regression test from the low-level details, which pertain to the system configuration and setup.
This allows users to write easily portable regression tests, focusing only on the functionality.

Regression tests in ReFrame are simple Python classes that specify the basic parameters of the test.
The framework will load the test and will send it down a well-defined pipeline that will take care of its execution.
The stages of this pipeline take care of all the system interaction details, such as programming environment switching, compilation, job submission, job status query, sanity checking and performance assessment.

Writing system regression tests in a high-level modern programming language, like Python, poses a great advantage in organizing and maintaining the tests.
Users can create their own test hierarchies or test factories for generating multiple tests at the same time and they can also customize them in a simple and expressive way.

<!--# Citing the Framework

If you use the framework please cite:

*ReFrame* [Reference of CUG paper](/).-->

# Use cases

The ReFrame framework has been in production at [CSCS](http://www.cscs.ch) since the upgrade of the [Piz Daint](http://www.cscs.ch/computers/piz_daint/index.html) system in early December 2016.

[Read the full story](/usecases)...

# Latest release

Reframe is being actively developed at [CSCS](http://www.cscs.ch/). You can always find the latest release [here](https://github.com/eth-cscs/reframe/releases/latest).


# Publications
* _ReFrame: A regression framework for checking the health of large HPC systems_ [[slides](/files/reframe-cug17-slides.pdf)]
