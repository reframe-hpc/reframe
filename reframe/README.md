# ReFrame

ReFrame is a framework developed by CSCS to facilitate the writing of regression tests that check the sanity of HPC systems.
Its main goal is to allow users to write their own regression tests without having to deal with all the details of setting up the environment for the test, quering the status of their job, managing the output of the job and looking for sanity and/or performance results.
Users should be concerned only about the logical requirements of their tests.
This allows users' regression checks to be maintained and adapted to new systems easily.

The user describes his test in a simple Python class and the framework takes care of all the details of the low-level interaction with the system.
Although the user still has to program in Python new checks, the framework is structured such that only a basic knowledge of Python is required to write a regression test, which will be able to run out-of-the-box on a variety of systems and programming environments.
In the future, we plan to allow users to describe their tests in a more abstract way using configuration files that wouldn't require any programming skills at all.
Of course, the Python interface will still be available for more advanced usage.

# Specific Goals

When designing the framework we have set three major goals:

## Productivity
  The writer of a regression test should focus only on the logical structure and requirements of the test and should not need to deal with any of the low level details of interacting with the system, e.g., how the environment of the test is loaded, how the associated job is created and has its status checked, how the output parsing is performed etc.
## Portability
  Configuring the framework to support new systems and system configurations should be easy and should not affect the existing tests.
  Also, adding support of a new system in a regression test should require minimal adjustments.
## Robustness and ease of use
  The new framework must be stable enough and easy to use by non-advanced users.
  When the system needs to be returned to users outside normal working hours the personnel in charge should be able to run the regression suite and verify the sanity of the system with a minimal involvement.


We have written the new framework entirely in Python and followed a layered design that abstracts away the system related details.
An API for writing regression tests is provided to the user at the highest level, allowing the description of the requirements of the test.
The framework defines and implements a concrete pipeline that a regression test goes through during its lifetime and the user is given the opportunity to intervene between the different stages and customize their behavior if needed.
All the system interaction mechanisms are implemented as backends and are not exposed directly to the writer of the check.
For example, the exact same test could be run on a system using either native Slurm or Slurm+ALPS or PBS+mpirun.
Similarly, the same test can run ``as-is'' on system partitions configured differently.
The writer of a regression test need not also care about generating a job script, querying the status of the associated job or managing the files of the test.
All of these are taken care of by the framework without affecting the regression test.
This not only makes a regression test easier to write, but it increases its readability as well, since the intent of the test is made clear right from its high-level description.

To meet the requirement of robustness we have employed a test-driven development process along with continuous integration right from the beginning of the framework's development, so as to make sure that it is tested thoroughly as it grows larger.
As a matter of fact, the amount of unit test code accompanying the framework almost matches the amount of the framework's code itself.
Regarding the ease of use we have tried to make the common case of invoking the regression suite as simple as possible by selecting reasonable defaults or by allowing to set default settings values per system configuration.

