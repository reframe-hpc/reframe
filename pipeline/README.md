# Regression Check Pipeline

The backbone of the ReFrame regression framework is the pipeline of the regression check.
This is a set of well defined phases that each regression test goes through during its lifetime.
The figure below depicts this pipeline in detail.

![pipeline.png](img/pipeline.png)

A regression test starts its life after it has been instantiated by the framework.
This is where all the basic information of the test is set.
At this point, although it is initialized, the regression test is not yet *live*, meaning that it does not run yet.
The framework will then go over all the loaded and initialized checks (we will talk about the loading and selection phases later), it will pick the next partition of the current system and the next programming environment for testing and will try to run the test.
If the test supports the current system partition and the current programming environment, it will be run and will go through all the following seven phases:
1. Initialization
2. Setup
3. Compilation
4. Running
5. Sanity checking
6. Performance checking
7. Cleanup


<!--1. *Setup*
   * During this phase the check is set up for the current partition and the current programming environment.
     The check's stage and output directories as well as its job descriptor are set up.
     The job descriptor contains all the necessary information needed to launch the regression check.
2. *Compilation*
   * Here the source code of the check, if any, is compiled. Some tests may not need to compile anything, in which case the status of this phase is always success.
3. *Job submission*
   * At this phase the regression check is launched.
     How the check will be launched depends on the job scheduler that serves the current system partition.
     A system partition (e.g., the login nodes of the system) may only accept local jobs (see [Site configuration](#site-configuration) for more information), in which case a local OS process will be launched for running the check.
     You can also force the regression to run all checks locally using the `--force-local` option.

4. *Job wait*
   * During this phase the previously launched job or process is waited for until it finishes and the job ID or the process ID are reported respectively.
     No check is performed whether the job or process finished gracefully.
     It is responsibility for the check to judge this.
     In practice, this means that this phase should always pass, unless something catastrophic has happened (bug in the framework or malfunctioning job scheduler).

5. *Sanity checking*
   * At this phase the regression check verifies whether it has finished successfully or not.

6. *Performance verification*
   * This phase is only relevant for performance regression checks, in which case the check verifies whether it has met its performance requirements.
     For simple regression checks, this phase is always a success.

7. *Clean up*
   * This phase is responsible for cleaning up the resources of the regression check.
     This includes copying some important files of the check to the output directory (e.g., generated job scripts, standard output/error etc.), removing its temporary stage directory and unloading its environment.-->


A test could implement some of them as no-ops.
As soon as the test is finished, its resources are cleaned up and the regression's environment is restored.
The regression will try to repeat the same procedure on the same regression test using the next programming environment until no further environments are left to be tested.
In the following we elaborate on each of the individual phases of the lifetime of a regression test.

# 1. The initialization phase

Although this phase is not part of the regression check pipeline as shown in the Figure above, it is quite important, since it sets up the definition of the regression test.
It serves as the *specification* of the check, where all the needed information to run the test is set.
A test could go through the whole pipeline performing all of its work without the need to override any of the pipeline stages.
In fact, this is the case for the majority of tests we have implemented for CSCS production systems.

# 2. The setup phase

A regression test is instantiated once by the framework and it is then reused several times for each of the system's partitions and their corresponding programming environments.
This first phase of the regression pipeline serves the purpose of preparing the test to run on the specified partition and programming environment by performing a number of operations described below:


## Setup and load the test's environment
At this point the environment of the current partition, the current programming environment and any test's specific environment will be loaded.
For example, if the current partition requires `slurm`, the current programming environment is `PrgEnv-gnu` and the test requires also `cudatoolkit`, this phase will be equivalent to the following:

```bash
module load slurm
module unload PrgEnv-cray
module load PrgEnv-gnu
module load cudatoolkit
```

Note that the framework automatically detects conflicting modules and unloads them first.
So the user need not to care about the existing environment at all.
He/She only needs to specify what is needed by his test.

## Setup the test's paths
Each regression test is associated with a stage directory and an output directory.
The stage directory will be the working directory of the test and all of its resources will be copied there before running.
The output directory is the directory where some important output files of the test will be kept.
By default these are the generated job script file, the standard output and standard error
The user can also specify additional files to be kept in the test's specification.
At this phase, all these directories are created.

## Prepare a job for the test
At this point a job descriptor will be created for the test, that wraps all the necessary information for generating a job script for it.
However, no job script is generated yet.
The job descriptor is an abstraction of the job scheduler's functionality relevant to the regression framework.
It is responsible for submitting a job in a job queue and waiting for its completion.
Currently, the ReFrame framework supports three job scheduler backends:
* __local__, which is basically a *pseudo-scheduler* that just spawns local OS processes,
* __nativeslurm__, which is the native [Slurm](https://slurm.schedmd.com) [[Yoo 2003](http://dx.doi.org/10.1007/10968987_3)] job scheduler and
* __slurm+alps__, which uses [Slurm](https://slurm.schedmd.com) for job submission, but uses [Cray's ALPS](http://docs.cray.com/books/S-2529-116//S-2529-116.pdf) for launching MPI processes on the compute nodes.

# 3. The compilation phase

At this phase the source code associated with test is compiled with the current programming environment.
Before compiling, all the resources of the test are copied to its stage directory and the framework changes into it.
After finishing the compilation, the framework returns to its original working directory.

# 4. The run phase

This phase comprises two subphases:
* __Job launch__: At this subphase a job script file for the regression test is generated and submitted to the job scheduler queue.
  If the job scheduler for the current partition is the __local__ one, a simple wrapper shell script will be generated and will be launched as a local OS process.
* __Job wait__: At this subphase the job (or local process) launched in the previous subphase is waited for.
  This phase is pretty basic: it just checks that the launched job (or local process) has finished.
  No check is made of whether the job or process has finished successfully or not.
  This is the responsibility of the next pipeline stage.

Currently, these two subphases are performed back-to-back making the ReFrame framework effectively serial, but in the future we plan to support asynchronous execution of regression tests.

# 5. The sanity checking phase

At this phase it is determined whether the check has finished successfully or not.
Although this decision is test-specific, the ReFrame framework provides the tests with an easy way for specifying complex patterns to check in the output files.
Multiple output files can be checked at the same time for determining the final sanity result.
Stateful parsing (e.g., aggregate operations such as average, min, max, etc.) is also supported and implemented transparently to the user.
We will present in detail the output parsing mechanism of the framework in [Output Parsing and Performance Assesment](/writing_checks/#output-parsing-and-performance-assessment).


# 6. The performance checking phase

At this phase the performance of the regression test is checked.
The framework uses the same mechanism for analyzing the output of the tests as in the sanity checking phase.
The only difference is that the user can now specify reference values per system or system partition, as well as threshold values for the performance.
The framework will take care of the output parsing and the matching of the correct reference values. For further information see [Output Parsing and Performance Assesment](/writing_checks/#output-parsing-and-performance-assessment).


# 7. The cleanup phase

This is the final stage of the regression pipeline and cleans up the resources of the environment.
Three steps are performed in this phase:
* The interesting files of the test (job script, standard output and standard error and any additional files specified by the user) are copied to its output directory for later inspection and bookkeeping,
* the stage directory is removed and
* the test's environment is revoked.

At this point the regression's environment is clean and in its original state and the regression can continue by either running the same test with a different programming environment or moving to another test.
