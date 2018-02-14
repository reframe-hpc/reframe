============================
The Regression Test Pipeline
============================

The backbone of the ReFrame regression framework is the regression test pipeline.
This is a set of well defined phases that each regression test goes through during its lifetime.
The figure below depicts this pipeline in detail.

.. figure:: _static/img/pipeline.svg
  :align: center
  :alt: The regression test pipeline


  The regression test pipeline

A regression test starts its life after it has been instantiated by the framework.
This is where all the basic information of the test is set.
At this point, although it is initialized, the regression test is not yet *live*, meaning that it does not run yet.
The framework will then go over all the loaded and initialized checks (we will talk about the loading and selection phases later), it will pick the next partition of the current system and the next programming environment for testing and will try to run the test.
If the test supports the current system partition and the current programming environment, it will be run and it will go through all the following seven phases:

1. Setup
2. Compilation
3. Running
4. Sanity checking
5. Performance checking
6. Cleanup

A test may implement some of them as no-ops. As soon as the test is finished, its resources are cleaned up and the framework's environment is restored.
ReFrame will try to repeat the same procedure on the same regression test using the next programming environment and the next system partition until no further environments and partitions are left to be tested.
In the following we elaborate on each of the individual phases of the lifetime of a regression test.

0. The Initialization Phase
---------------------------

This phase is not part of the regression test pipeline as shown above, but it is quite important, since during this phase the test is loaded into memory and initialized.
As we shall see in the `"Tutorial" <tutorial.html>`__ and in the `"Customizing Further A ReFrame Regression Test" <advanced.html>`__ sections, this is the phase where the *specification* of a test is set.
At this point the current system is already known and the test may be set up accordingly.
If no further differentiation is needed depending on the system partition or the programming environment, the test could go through the whole pipeline performing all of its work without the need to override any of the other pipeline stages.
In fact, this is perhaps the most common case for most of the regression tests.

1. The Setup Phase
------------------

A regression test is instantiated once by the framework and it is then copied each time a new system partition or programming environment is tried.
This first phase of the regression pipeline serves the purpose of preparing the test to run on the specified partition and programming environment by performing a number of operations described below:

Set up and load the test's environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At this point the environment of the current partition, the current programming environment and any test's specific environment will be loaded.
For example, if the current partition requires ``slurm``, the current programming environment is ``PrgEnv-gnu`` and the test requires also ``cudatoolkit``, this phase will be equivalent to the following:

.. code:: bash

  module load slurm
  module unload PrgEnv-cray
  module load PrgEnv-gnu
  module load cudatoolkit

Note that the framework automatically detects conflicting modules and unloads them first.
So the user need not to care about the existing environment at all.
She only needs to specify what is needed by her test.

Setup the test's paths
^^^^^^^^^^^^^^^^^^^^^^

Each regression test is associated with a stage directory and an output directory.
The stage directory will be the working directory of the test and all of its resources will be copied there before running.
The output directory is the directory where some important output files of the test will be kept.
By default these are the generated job script file, the standard output and standard error.
The user can also specify additional files to be kept in the test's specification.
At this phase, all these directories are created.

Prepare a job for the test
^^^^^^^^^^^^^^^^^^^^^^^^^^

At this point a *job descriptor* will be created for the test.
A job descriptor in ReFrame is an abstraction of the job scheduler's functionality relevant to the regression framework.
It is responsible for submitting a job in a job queue and waiting for its completion.
ReFrame supports two job scheduler backends that can be combined with several different parallel program launchers.
For a complete list of the job scheduler/parallel launchers combinations, please refer to `"Partition Configuration" <configure.html#partition-configuration>`__.

2. The Compilation Phase
------------------------

At this phase the source code associated with test is compiled with the current programming environment.
Before compiling, all the resources of the test are copied to its stage directory and the compilation is performed from that directory.

3. The Run Phase
----------------

This phase comprises two subphases:

* **Job launch**: At this subphase a job script file for the regression test is generated and submitted to the job scheduler queue.
  If the job scheduler for the current partition is the **local** one, a simple wrapper shell script will be generated and will be launched as a local OS process.
* **Job wait**: At this subphase the job (or local process) launched in the previous subphase is waited for.
  This phase is pretty basic: it just checks that the launched job (or local process) has finished.
  No check is made of whether the job or process has finished successfully or not.
  This is the responsibility of the next pipeline stage.

ReFrame currently supports two execution policies:

* **serial**: In the serial execution policy, these two subphases are performed back-to-back and the framework blocks until the current regression test finishes.
* **asynchronous**: In the asynchronous execution policy, as soon as the job associated to the current test is launched, ReFrame continues its execution by executing and launching the subsequent test cases.

4. The Sanity Checking Phase
----------------------------

At this phase it is determined whether the check has finished successfully or not.
Although this decision is test-specific, ReFrame provides a very flexible and expressive way for specifying complex patterns and operations to be performed on the test's output in order to determine the outcome of the test.

5. The Performance Checking Phase
---------------------------------

At this phase the performance of the regression test is checked.
ReFrame uses the same mechanism for analyzing the output of the test as with sanity checking.
The only difference is that the user can now specify reference values per system or system partition, as well as acceptable performance thresholds

6. The Cleanup Phase
--------------------

This is the final stage of the regression test pipeline and it is responsible for cleaning up the resources of the test.
Three steps are performed in this phase:

1. The interesting files of the test (job script, standard output and standard error and any additional files specified by the user) are copied to its output directory for later inspection and bookkeeping,
2. the stage directory is removed and
3. the test's environment is revoked.

At this point the ReFrame's environment is clean and in its original state and the framework may continue by running more test cases.
