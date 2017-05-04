# What is ReFrame?

ReFrame is a framework developed by CSCS to facilitate the writing of regression tests that check the sanity of HPC systems.
Its main goal is to allow users to write their own regression tests without having to deal with all the details of setting up the environment for the test, quering the status of their job, managing the output of the job and looking for sanity and/or performance results.
Users should be concerned only about the logical requirements of their tests.
This allows users' regression checks to be maintained and adapted to new systems easily.

The user describes his test in a simple Python class and the framework takes care of all the details of the low-level interaction with the system.
Although the user still has to program in Python new checks, the framework is structured such that only a basic knowledge of Python is required to write a regression test, which will be able to run out-of-the-box on a variety of systems and programming environments.
In the future, we plan to allow users to describe their tests in a more abstract way using configuration files that wouldn't require any programming skills at all.
Of course, the Python interface will still be available for more advanced usage.

# General Goals

When designing the framework we have set three major goals:

* __Productivity__
  The writer of a regression test should focus only on the logical structure and requirements of the test and should not need to deal with any of the low level details of interacting with the system, e.g., how the environment of the test is loaded, how the associated job is created and has its status checked, how the output parsing is performed etc.
* __Portability__
  Configuring the framework to support new systems and system configurations should be easy and should not affect the existing tests.
  Also, adding support of a new system in a regression test should require minimal adjustments.
* __Robustness and ease of use__
  The new framework must be stable enough and easy to use by non-advanced users.
  When the system needs to be returned to users outside normal working hours the personnel in charge should be able to run the regression suite and verify the sanity of the system with a minimal involvement.


# Why ReFrame?

HPC systems are highly complex systems in all levels of integration;
from the physical infrastructure up to the software stack provided to the users.
A small change in any of these levels could have an impact on the stability or the performance of the system perceived by the end users.
It is of crucial importance, therefore, not only to make sure that the system is in a sane condition after every maintenance before handing it off to users, but also to monitor its performance during production, so that possible problems are detected early enough and the quality of service is not compromised.

Regression testing can provide a reliable way to ensure the stability and the performance requirements of the system, provided that sufficient tests exist that cover a wide aspect of the system's operations from both the operators' and users' point of view.
However, given the complexity of HPC systems, writing and maintaining regression tests can be a very time consuming task.
A small change in system configuration or deployment may require adapting hundreds of regression tests at the same time.
Similarly, porting a test to a different system may require significant effort if the new system's configuration is substantially different than that of the system that it was originally written for.

This way, ReFrame was designed to help HPC support teams to:
* monitor the impact of changes to the system that would affect negativelly the users
* monitor system performance
* monitor system stability
* guarantee quality of service

And also decrease the amount of time and resources required to:
* write and maintain regression tests
* port the regression test to other HPC systems


# What does it do?

The framework defines and implements a concrete pipeline that a regression test goes through during its lifetime and the user is given the opportunity to intervene between the different stages and customize their behavior if needed.
All the system interaction mechanisms are implemented as backends and are not exposed directly to the writer of the check.
For example, the exact same test could be run on a system using either native Slurm or Slurm+ALPS or PBS+mpirun.
Similarly, the same test can run *as-is* on system partitions configured differently.
The writer of a regression test need not also care about generating a job script, querying the status of the associated job or managing the files of the test.
All of these are taken care of by the framework without affecting the regression test.
This not only makes a regression test easier to write, but it increases its readability as well, since the intent of the test is made clear right from its high-level description.

# Development process
To meet the requirement of robustness we have employed a test-driven development process along with continuous integration right from the beginning of the framework's development, so as to make sure that it is tested thoroughly as it grows larger.
As a matter of fact, the amount of unit test code accompanying the framework almost matches the amount of the framework's code itself.
Regarding the ease of use we have tried to make the common case of invoking the regression suite as simple as possible by selecting reasonable defaults or by allowing to set default settings values per system configuration.



# Why Python?

We have written the framework entirely in Python and followed a layered design that abstracts away the system related details.
An API for writing regression tests is provided to the user at the highest level, allowing the description of the requirements of the test.
An advantage of writing regression tests in a high-level language, such as Python, is that one can take advantage of features not present in classical shell scripting.
For example, one can create groups of related tests that share common characteristics and/or functionality by implementing them in a base class, from which all the related concrete tests inherit.
This eliminates unnecessary code duplication and reduces significantly the maintenance cost.

