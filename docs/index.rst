==================
Welcome to ReFrame
==================

ReFrame is a new framework for writing regression tests for HPC systems.
The goal of this framework is to abstract away the complexity of the interactions with the system, separating the logic of a regression test from the low-level details, which pertain to the system configuration and setup.
This allows users to write easily portable regression tests, focusing only on the functionality.

Regression tests in ReFrame are simple Python classes that specify the basic parameters of the test.
The framework will load the test and will send it down a well-defined pipeline that will take care of its execution.
The stages of this pipeline take care of all the system interaction details, such as programming environment switching, compilation, job submission, job status query, sanity checking and performance assessment.

ReFrame also offers a high-level and flexible abstraction for writing sanity and performance checks for your regression tests, without having to care about the details of parsing output files, searching for patterns and testing against reference values for different systems.

Writing system regression tests in a high-level modern programming language, like Python, poses a great advantage in organizing and maintaining the tests.
Users can create their own test hierarchies or test factories for generating multiple tests at the same time and they can also customize them in a simple and expressive way.

For versions 2.6.1 and older, please refer to `this documentation <_old/index.html>`__.

Use Cases
=========

The ReFrame framework has been in production at `CSCS <http://www.cscs.ch>`__ since the upgrade of the `Piz Daint <http://www.cscs.ch/computers/piz_daint/index.html>`__ system in early December 2016.

`Read the full story <usecases.html>`__...

Latest Release
==============

Reframe is being actively developed at `CSCS <http://www.cscs.ch/>`__.
You can always find the latest release `here <https://github.com/eth-cscs/reframe/releases/latest>`__.

Publications
============

* `Presentation <https://drive.google.com/file/d/1sIecW59E-AvhD-vl6c6QGXM14UKNzgo_/view?usp=sharing>`__ & `Demo <https://asciinema.org/a/6SQJTaRe2zrMInV92X0yb2gTh>`__ @ `SC18 <https://sc18.supercomputing.org/>`__
* Presentation [`pdf <https://github.com/eth-cscs/UserLabDay/blob/master/slides/ci_and_regression/ReFrame_CI.pdf>`__] [`pptx <https://github.com/eth-cscs/UserLabDay/blob/master/slides/ci_and_regression/ReFrame_CI.pptx>`__] @ `CSCS User Lab Day 2018 <https://github.com/eth-cscs/UserLabDay>`__
* `Presentation <_static/files/reframe-hpac18-slides.pdf>`__ & `Demo1 <https://asciinema.org/a/kAETsA1ojG6L7dkzow8opEGvr>`__,  `Demo2 <https://asciinema.org/a/LLcOToWYX4gRIfrcb1GpmvkuB>`__ @ `HPC Advisory Council 2018 <http://www.hpcadvisorycouncil.com/events/2018/swiss-workshop/>`__
* `Presentation <_static/files/reframe-bof-sc17-slides.pdf>`__ & `Demo <https://asciinema.org/a/kBZfdV0rmc0PCd84zxk6nAojG>`__ @ `SC17 <https://sc17.supercomputing.org/>`__
* `Presentation <_static/files/reframe-cug17-slides.pdf>`__ @ `CUG 2017 <https://cug.org/cug-2017/>`__



.. toctree::
   :caption: Table of Contents:
   :hidden:

   Getting Started <started>
   Configuring ReFrame For Your Site <configure>
   The Regression Test Pipeline <pipeline>
   ReFrame Tutorial <tutorial>
   Customizing Further A Regression Test <advanced>
   Understanding The Mechanism Of Sanity Functions <deferrables>
   Running ReFrame <running>
   Use cases <usecases>
   About ReFrame <about>
   Reference Guide <reference>
   Sanity Functions Reference <sanity_functions_reference>
