==========================
How ReFrame Executes Tests
==========================

A ReFrame test will be normally tried for different programming environments and different partitions within the same ReFrame run.
These are defined in the test's :func:`__init__` method, but it is not this original test object that is scheduled for execution.
The following figure explains in more detail the process:

.. figure:: _static/img/reframe-test-cases.svg
  :align: center
  :alt: How ReFrame loads and schedules tests for execution.

  :sub:`How ReFrame loads and schedules tests for execution.`

When ReFrame loads a test from the disk it unconditionally constructs it executing its :func:`__init__` method.
The practical implication of this is that your test will be instantiated even if it will not run on the current system.
After all the tests are loaded, they are filtered based on the current system and any other criteria (such as programming environment, test attributes etc.) specified by the user (see `Test Filtering <manpage.html#test-filtering>`__ for more details).
After the tests are filtered, ReFrame creates the actual `test cases` to be run. A test case is essentially a tuple consisting of the test, the system partition and the programming environment to try.
The test that goes into a test case is essentially a `clone` of the original test that was instantiated upon loading.
This ensures that the test case's state is not shared and may not be reused in any case.
Finally, the generated test cases are passed to a `runner` that is responsible for scheduling them for execution based on the selected execution policy.


The Regression Test Pipeline
----------------------------

Each ReFrame test case goes through a pipeline with clearly defined stages.
ReFrame tests can customize their operation as they execute by attaching hooks to the pipeline stages.
The following figure shows the different pipeline stages.

.. figure:: _static/img/pipeline.svg
  :align: center
  :alt: The regression test pipeline

  :sub:`The regression test pipeline.`


All tests will go through every stage one after the other.
However, some types of tests implement some stages as no-ops, whereas the sanity or performance check phases may be skipped on demand (see :option:`--skip-sanity-check` and :option:`--skip-performance-check` options).
In the following we describe in more detail what happens in every stage.

---------------
The Setup Phase
---------------

During this phase the test will be set up for the currently selected system partition and programming environment.
The :attr:`current_partition` and :attr:`current_environ` test attributes will be set and the paths associated to this test case (stage, output and performance log directories) will be created.
A `job descriptor <regression_test_api.html#reframe.core.pipeline.RegressionTest.job>`__ will also be created for the test case containing information about the job to be submitted later in the pipeline.


---------------
The Build Phase
---------------

During this phase the source code associated with the test is compiled using the current programming environment.
If the test is `"run-only," <regression_test_api.html#reframe.core.pipeline.RunOnlyRegressionTest>`__ this phase is a no-op.

Before building the test, all the `resources <regression_test_api.html#reframe.core.pipeline.RegressionTest.sourcesdir>`__ associated with it are copied to the test case's stage directory.
ReFrame then temporarily switches to that directory and builds the test.

-------------
The Run Phase
-------------

During this phase a job script associated with the test case will be created and it will be submitted for execution.
If the test is `"run-only," <regression_test_api.html#reframe.core.pipeline.RunOnlyRegressionTest>`__ its `resources <regression_test_api.html#reframe.core.pipeline.RegressionTest.sourcesdir>`__ will be first copied to the test case's stage directory.
ReFrame will temporarily switch to that directory and spawn the test's job from there.
This phase is executed asynchronously (either a batch job is spawned or a local process is started) and it is up to the selected `execution policy <#execution-policies>`__ to block or not until the associated job finishes.


----------------
The Sanity Phase
----------------

During this phase, the sanity of the test's output is checked.
ReFrame makes no assumption as of what a successful test is; it does not even look into its exit code.
This is entirely up to the test to define.
ReFrame provides a flexible and expressive way for specifying complex patterns and operations to be performed on the test's output in order to determine the outcome of the test.

---------------------
The Performance Phase
---------------------

During this phase, the performance metrics reported by the test (if it is performance test) are collected, logged and compared to their reference values.
The mechanism for extracting performance metrics from the test's output is the same used by the sanity checking phase for extracting patterns from the test's output.

-----------------
The Cleanup Phase
-----------------

During this final stage of the pipeline, the test's resources are cleaned up.
More specifically, if the test has finished successfully, all interesting test files (build/job scripts, build/job script output and any user-specified files) are copied to ReFrame's output directory and the stage directory of the test is deleted.


Execution Policies
------------------

All regression tests in ReFrame will execute the pipeline stages described above.
However, how exactly this pipeline will be executed is responsibility of the test execution policy.
There are two execution policies in ReFrame: the serial and the asynchronous one.

In the serial execution policy, a new test gets into the pipeline after the previous one has exited.
As the figure below shows, this can lead to long idling times in the run phase, since the execution blocks until the associated test job finishes.


.. figure:: _static/img/serial-exec-policy.svg
  :align: center
  :alt: The serial execution policy.

  :sub:`The serial execution policy.`


In the asynchronous execution policy, multiple tests can be simultaneously on-the-fly.
When a test enters the run phase, ReFrame does not block, but continues by picking the next test case to run.
This continues until no more test cases are left for execution or until a maximum concurrency limit is reached.
At the end, ReFrame enters a busy-wait loop monitoring the spawned test cases.
As soon as test case finishes, it resumes its pipeline and runs it to completion.
The following figure shows how the asynchronous execution policy works.


.. figure:: _static/img/async-exec-policy.svg
  :align: center
  :alt: The asynchronous execution policy.

  :sub:`The asynchronous execution policy.`


ReFrame tries to keep concurrency high by maintaining as many test cases as possible simultaneously active.
When the `concurrency limit <config_reference.html#.systems[].partitions[].max_jobs>`__ is reached, ReFrame will first try to free up execution slots by checking if any of the spawned jobs have finished, and it will fill that slots first before throttling execution.

ReFrame uses polling to check the status of the spawned jobs, but it does so in a dynamic way, in order to ensure both responsiveness and avoid overloading the system job scheduler with excessive polling.

Time Profiling of the Pipeline
------------------------------

Since version 3.0, ReFrame keeps track of the time a test spends in each phases of the pipeline, but it has some limitations.
The time that is reported for the run phase is not obtained from reliable sources, like the accounting information of the scheduler, but instead from the framework itself.
This means that the end of the run phase is considered to be when ReFrame realizes the test has finished running and it can vary significantly because of the polling rate.

The second limitation comes from the dependencies and the fact that a test's resources might not be cleaned up as soon as it finishes.
ReFrame will wait until all the dependencies of a test finish successfully their execution before cleaning up its resources and if one of them fails it will ignore this phase completely.
The final status of the test and its time profiling information are reported before the cleanup phase.
