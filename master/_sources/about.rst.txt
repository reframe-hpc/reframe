=============
About ReFrame
=============

What Is ReFrame?
----------------

ReFrame is a framework developed by CSCS to facilitate the writing of regression tests that check the sanity of HPC systems.
Its main goal is to allow users to write their own regression tests without having to deal with all the details of setting up the environment for the test, querying the status of their job, managing the output of the job and looking for sanity and/or performance results.
Users should be concerned only about the logical requirements of their tests.
This allows users' regression checks to be maintained and adapted to new systems easily.

The user describes his test in a simple Python class and the framework takes care of all the details of the low-level interaction with the system.
The framework is structured in such a way that with a basic knowledge of Python and minimal coding a user can write a regression test, which will be able to run out-of-the-box on a variety of systems and programming environments.

Writing regression tests in a high-level language, such as Python, allows users to take advantage of the language's higher expressiveness and bigger capabilities compared to classical shell scripting, which is the norm in HPC testing.
This could lead to a more manageable code base of regression tests with significantly reduced maintenance costs.

ReFrame's Goals
---------------

When designing the framework we have set three major goals:

Productivity
  The writer of a regression test should focus only on the logical structure and requirements of the test and should not need to deal with any of the low level details of interacting with the system, e.g., how the environment of the test is loaded, how the associated job is created and has its status checked, how the output parsing is performed etc.
Portability
  Configuring the framework to support new systems and system configurations should be easy and should not affect the existing tests.
  Also, adding support of a new system in a regression test should require minimal adjustments.
Robustness and ease of use
  The new framework must be stable enough and easy to use by non-advanced users.
  When the system needs to be returned to users outside normal working hours the personnel in charge should be able to run the regression suite and verify the sanity of the system with a minimal involvement.

Why ReFrame?
------------

HPC systems are highly complex systems in all levels of integration;
from the physical infrastructure up to the software stack provided to the users.
A small change in any of these levels could have an impact on the stability or the performance of the system perceived by the end users.
It is of crucial importance, therefore, not only to make sure that the system is in a sane condition after every maintenance before handing it off to users, but also to monitor its performance during production, so that possible problems are detected early enough and the quality of service is not compromised.

Regression testing can provide a reliable way to ensure the stability and the performance requirements of the system, provided that sufficient tests exist that cover a wide aspect of the system's operations from both the operators' and users' point of view.
However, given the complexity of HPC systems, writing and maintaining regression tests can be a very time consuming task.
A small change in system configuration or deployment may require adapting hundreds of regression tests at the same time.
Similarly, porting a test to a different system may require significant effort if the new system's configuration is substantially different than that of the system that it was originally written for.

ReFrame was designed to help HPC support teams to easily write tests that

* monitor the impact of changes to the system that would affect negatively the users,
* monitor system performance,
* monitor system stability and
* guarantee quality of service.

And also decrease the amount of time and resources required to

* write and maintain regression tests and
* port regression tests to other HPC systems.
