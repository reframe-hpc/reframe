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

Use Cases
=========

The ReFrame framework has been in production at `CSCS <http://www.cscs.ch>`__ since the upgrade of the `Piz Daint <http://www.cscs.ch/computers/piz_daint/index.html>`__ system in early December 2016.

`Read the full story <usecases.html>`__...

Latest Release
==============

ReFrame is being actively developed at `CSCS <http://www.cscs.ch/>`__.
You can always find the latest release `here <https://github.com/eth-cscs/reframe/releases/latest>`__.

Publications
============

* Slides [`pdf <https://drive.google.com/open?id=1W7R5lfRkXvBpVDSZ7dVBadk_d3K4dFrS>`__] @ `5th EasyBuild User Meeting 2020 <https://github.com/easybuilders/easybuild/wiki/5th-EasyBuild-User-Meeting>`__.
* Slides [`pdf <https://drive.google.com/open?id=1Z3faPh9OSSXvlLHL07co3MRRn443dYsY>`__] @ `HPC System Testing BoF <https://sc19.supercomputing.org/session/?sess=sess324>`__, SC'19.
* Slides [`pdf <https://drive.google.com/open?id=1JOFqY3ejbR1X5kTn_IZyp1GlCd2ZZS58>`__] @ `HUST 2019 <https://sc19.supercomputing.org/session/?sess=sess116>`__, SC'19.
* Slides [`pdf <https://drive.google.com/open?id=1iwg1I48LVaWhhZCZIYPJSi3hdFLRcuhi>`__] @ `HPC Knowledge Meeting '19 <https://hpckp.org/>`__.
* Slides [`pdf <https://fosdem.org/2019/schedule/event/reframe/attachments/slides/3226/export/events/attachments/reframe/slides/3226/FOSDEM_2019.pdf>`__] & `Talk <https://fosdem.org/2019/schedule/event/reframe/>`__ @ `FOSDEM'19 <https://fosdem.org/2019/>`__.
* Slides [`pdf <https://indico.cism.ucl.ac.be/event/4/contributions/24/attachments/30/62/ReFrame_EUM_2019.pdf>`__] @ `4th EasyBuild User Meeting <https://github.com/easybuilders/easybuild/wiki/4th-EasyBuild-User-Meeting>`__.
* Slides [`pdf <https://drive.google.com/open?id=1bSykDrl1e2gPflf4jFJ8kfe_SZAtrJ_Q>`__] @ `HUST 2018 <https://sc18.supercomputing.org/>`__, SC'18.
* Slides [`pdf <https://github.com/eth-cscs/UserLabDay/blob/master/slides/ci_and_regression/ReFrame_CI.pdf>`__] [`pptx <https://github.com/eth-cscs/UserLabDay/blob/master/slides/ci_and_regression/ReFrame_CI.pptx>`__] @ `CSCS User Lab Day 2018 <https://github.com/eth-cscs/UserLabDay>`__.
* Slides [`pdf <https://drive.google.com/open?id=1sZhibvUlGlT670aOHPdMlWFffWptYzLX>`__] @ `HPC Advisory Council 2018 <http://www.hpcadvisorycouncil.com/events/2018/swiss-workshop/>`__.
* Slides [`pdf <https://drive.google.com/open?id=1EyJ-siupkgLeVT54A4WlFpQtrJaU0xOy>`__] @ `SC17 <https://sc17.supercomputing.org/>`__.
* Slides [`pdf <https://drive.google.com/open?id=18VrCy0MTplGo67uxVbzYZicQChor9VSY>`__] @ `CUG 2017 <https://cug.org/cug-2017/>`__.



.. toctree::
   :caption: Table of Contents
   :hidden:

   Getting Started <started>
   Configuring ReFrame For Your Site <configure>
   The Regression Test Pipeline <pipeline>
   ReFrame Tutorial <tutorial>
   Customizing Further A Regression Test <advanced>
   How Test Dependencies Work in ReFrame <dependencies>
   Understanding The Mechanism Of Sanity Functions <deferrables>
   Running ReFrame <running>
   Use cases <usecases>
   Migrating to ReFrame 3 <migration_2_to_3>
   About ReFrame <about>
   Reference Guide <reference>
   Sanity Functions Reference <sanity_functions_reference>
   Configuration Reference <config_reference>
