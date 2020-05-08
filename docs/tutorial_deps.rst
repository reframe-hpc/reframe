===============================================
Tutorial 3: Using Dependencies in ReFrame Tests
===============================================

.. versionadded:: 2.21


A ReFrame test may define dependencies to other tests.
An example scenario is to test different runtime configurations of a benchmark that you need to compile, or run a scaling analysis of a code.
In such cases, you don't want to rebuild your test for each runtime configuration.
You could have a build test, which all runtime tests would depend on.
This is the approach we take with the following example, that fetches, builds and runs several `OSU benchmarks <http://mvapich.cse.ohio-state.edu/benchmarks/>`__.
We first create a basic compile-only test, that fetches the benchmarks and builds them for the different programming environments:

.. literalinclude:: ../tutorial/advanced/osu/osu_benchmarks.py
   :lines: 92-106

There is nothing particular to that test, except perhaps that you can set :attr:`sourcesdir <reframe.core.pipeline.RegressionTest.sourcesdir>` to ``None`` even for a test that needs to compile something.
In such a case, you should at least provide the commands that fetch the code inside the :attr:`prebuild_cmd <reframe.core.pipeline.RegressionTest.prebuild_cmd>` attribute.

For the next test we need to use the OSU benchmark binaries that we just built, so as to run the MPI ping-pong benchmark.
Here is the relevant part:

.. literalinclude:: ../tutorial/advanced/osu/osu_benchmarks.py
   :lines: 12-44

First, since we will have multiple similar benchmarks, we move all the common functionality to the :class:`OSUBenchmarkTestBase` base class.
Again nothing new here; we are going to use two nodes for the benchmark and we set :attr:`sourcesdir <reframe.core.pipeline.RegressionTest.sourcesdir>` to ``None``, since none of the benchmark tests will use any additional resources.
The new part comes in with the :class:`OSULatencyTest` test in the following line:


.. literalinclude:: ../tutorial/advanced/osu/osu_benchmarks.py
   :lines: 32

Here we tell ReFrame that this test depends on a test named ``OSUBuildTest``.
This test may or may not be defined in the same test file; all ReFrame needs is the test name.
By default, the :func:`depends_on() <reframe.core.pipeline.RegressionTest.depends_on>` function will create dependencies between the individual test cases of the :class:`OSULatencyTest` and the :class:`OSUBuildTest`, such that the :class:`OSULatencyTest` using ``PrgEnv-gnu`` will depend on the outcome of the :class:`OSUBuildTest` using ``PrgEnv-gnu``, but not on the outcome of the :class:`OSUBuildTest` using ``PrgEnv-intel``.
This behaviour can be changed, but it is covered in detail in :doc:`dependencies`.
You can create arbitrary test dependency graphs, but they need to be acyclic.
If ReFrame detects cyclic dependencies, it will refuse to execute the set of tests and will issue an error pointing out the cycle.

A ReFrame test with dependencies will execute, i.e., enter its `setup` stage, only after `all` of its dependencies have succeeded.
If any of its dependencies fails, the current test will be marked as failure as well.

The next step for the :class:`OSULatencyTest` is to set its executable to point to the binary produced by the :class:`OSUBuildTest`.
This is achieved with the following specially decorated function:

.. literalinclude:: ../tutorial/advanced/osu/osu_benchmarks.py
   :lines: 37-43

The :func:`@require_deps <reframe.core.decorators.require_deps>` decorator will bind the arguments passed to the decorated function to the result of the dependency that each argument names.
In this case, it binds the ``OSUBuildTest`` function argument to the result of a dependency named ``OSUBuildTest``.
In order for the binding to work correctly the function arguments must be named after the target dependencies.
However, referring to a dependency only by the test's name is not enough, since a test might be associated with multiple programming environments.
For this reason, a dependency argument is actually bound to a function that accepts as argument the name of a target programming environment.
If no arguments are passed to that function, as in this example, the current programming environment is implied, such that ``OSUBuildTest()`` is equivalent to ``OSUBuildTest(self.current_environ.name)``.
This call returns the actual test case of the dependency that has been executed.
This allows you to access any attribute from the target test, as we do in this example by accessing the target test's stage directory, which we use to construct the path of the executable.
This concludes the presentation of the :class:`OSULatencyTest` test. The :class:`OSUBandwidthTest` is completely analogous.

The :class:`OSUAllreduceTest` shown below is similar to the other two, except that it is parameterized.
It is essentially a scalability test that is running the ``osu_allreduce`` executable created by the :class:`OSUBuildTest` for 2, 4, 8 and 16 nodes.

.. literalinclude:: ../tutorial/advanced/osu/osu_benchmarks.py
   :lines: 69-89

The full set of OSU example tests is shown below:

.. literalinclude:: ../tutorial/advanced/osu/osu_benchmarks.py

Notice that the order dependencies are defined in a test file is irrelevant.
In this case, we define :class:`OSUBuildTest` at the end.
ReFrame will make sure to properly sort the tests and execute them.

Here is the output when running the OSU tests with the asynchronous execution policy:

.. code-block:: none

  [==========] Running 7 check(s)
  [==========] Started on Wed Mar 25 13:51:06 2020

  [----------] started processing OSUBuildTest (OSU benchmarks build test)
  [ RUN      ] OSUBuildTest on daint:gpu using PrgEnv-gnu
  [ RUN      ] OSUBuildTest on daint:gpu using PrgEnv-intel
  [ RUN      ] OSUBuildTest on daint:gpu using PrgEnv-pgi
  [----------] finished processing OSUBuildTest (OSU benchmarks build test)

  [----------] started processing OSULatencyTest (OSU latency test)
  [ RUN      ] OSULatencyTest on daint:gpu using PrgEnv-gnu
  [      DEP ] OSULatencyTest on daint:gpu using PrgEnv-gnu
  [ RUN      ] OSULatencyTest on daint:gpu using PrgEnv-intel
  [      DEP ] OSULatencyTest on daint:gpu using PrgEnv-intel
  [ RUN      ] OSULatencyTest on daint:gpu using PrgEnv-pgi
  [      DEP ] OSULatencyTest on daint:gpu using PrgEnv-pgi
  [----------] finished processing OSULatencyTest (OSU latency test)

  [----------] started processing OSUBandwidthTest (OSU bandwidth test)
  [ RUN      ] OSUBandwidthTest on daint:gpu using PrgEnv-gnu
  [      DEP ] OSUBandwidthTest on daint:gpu using PrgEnv-gnu
  [ RUN      ] OSUBandwidthTest on daint:gpu using PrgEnv-intel
  [      DEP ] OSUBandwidthTest on daint:gpu using PrgEnv-intel
  [ RUN      ] OSUBandwidthTest on daint:gpu using PrgEnv-pgi
  [      DEP ] OSUBandwidthTest on daint:gpu using PrgEnv-pgi
  [----------] finished processing OSUBandwidthTest (OSU bandwidth test)

  [----------] started processing OSUAllreduceTest_2 (OSU Allreduce test)
  [ RUN      ] OSUAllreduceTest_2 on daint:gpu using PrgEnv-gnu
  [      DEP ] OSUAllreduceTest_2 on daint:gpu using PrgEnv-gnu
  [ RUN      ] OSUAllreduceTest_2 on daint:gpu using PrgEnv-intel
  [      DEP ] OSUAllreduceTest_2 on daint:gpu using PrgEnv-intel
  [ RUN      ] OSUAllreduceTest_2 on daint:gpu using PrgEnv-pgi
  [      DEP ] OSUAllreduceTest_2 on daint:gpu using PrgEnv-pgi
  [----------] finished processing OSUAllreduceTest_2 (OSU Allreduce test)

  [----------] started processing OSUAllreduceTest_4 (OSU Allreduce test)
  [ RUN      ] OSUAllreduceTest_4 on daint:gpu using PrgEnv-gnu
  [      DEP ] OSUAllreduceTest_4 on daint:gpu using PrgEnv-gnu
  [ RUN      ] OSUAllreduceTest_4 on daint:gpu using PrgEnv-intel
  [      DEP ] OSUAllreduceTest_4 on daint:gpu using PrgEnv-intel
  [ RUN      ] OSUAllreduceTest_4 on daint:gpu using PrgEnv-pgi
  [      DEP ] OSUAllreduceTest_4 on daint:gpu using PrgEnv-pgi
  [----------] finished processing OSUAllreduceTest_4 (OSU Allreduce test)

  [----------] started processing OSUAllreduceTest_8 (OSU Allreduce test)
  [ RUN      ] OSUAllreduceTest_8 on daint:gpu using PrgEnv-gnu
  [      DEP ] OSUAllreduceTest_8 on daint:gpu using PrgEnv-gnu
  [ RUN      ] OSUAllreduceTest_8 on daint:gpu using PrgEnv-intel
  [      DEP ] OSUAllreduceTest_8 on daint:gpu using PrgEnv-intel
  [ RUN      ] OSUAllreduceTest_8 on daint:gpu using PrgEnv-pgi
  [      DEP ] OSUAllreduceTest_8 on daint:gpu using PrgEnv-pgi
  [----------] finished processing OSUAllreduceTest_8 (OSU Allreduce test)

  [----------] started processing OSUAllreduceTest_16 (OSU Allreduce test)
  [ RUN      ] OSUAllreduceTest_16 on daint:gpu using PrgEnv-gnu
  [      DEP ] OSUAllreduceTest_16 on daint:gpu using PrgEnv-gnu
  [ RUN      ] OSUAllreduceTest_16 on daint:gpu using PrgEnv-intel
  [      DEP ] OSUAllreduceTest_16 on daint:gpu using PrgEnv-intel
  [ RUN      ] OSUAllreduceTest_16 on daint:gpu using PrgEnv-pgi
  [      DEP ] OSUAllreduceTest_16 on daint:gpu using PrgEnv-pgi
  [----------] finished processing OSUAllreduceTest_16 (OSU Allreduce test)

  [----------] waiting for spawned checks to finish
  [       OK ] ( 1/21) OSUBuildTest on daint:gpu using PrgEnv-pgi
  [       OK ] ( 2/21) OSUBuildTest on daint:gpu using PrgEnv-gnu
  [       OK ] ( 3/21) OSUBuildTest on daint:gpu using PrgEnv-intel
  [       OK ] ( 4/21) OSUAllreduceTest_2 on daint:gpu using PrgEnv-pgi
  [       OK ] ( 5/21) OSUAllreduceTest_4 on daint:gpu using PrgEnv-pgi
  [       OK ] ( 6/21) OSUAllreduceTest_8 on daint:gpu using PrgEnv-pgi
  [       OK ] ( 7/21) OSUAllreduceTest_16 on daint:gpu using PrgEnv-pgi
  [       OK ] ( 8/21) OSUAllreduceTest_4 on daint:gpu using PrgEnv-gnu
  [       OK ] ( 9/21) OSUAllreduceTest_16 on daint:gpu using PrgEnv-gnu
  [       OK ] (10/21) OSUAllreduceTest_8 on daint:gpu using PrgEnv-gnu
  [       OK ] (11/21) OSUAllreduceTest_16 on daint:gpu using PrgEnv-intel
  [       OK ] (12/21) OSULatencyTest on daint:gpu using PrgEnv-pgi
  [       OK ] (13/21) OSUAllreduceTest_2 on daint:gpu using PrgEnv-gnu
  [       OK ] (14/21) OSULatencyTest on daint:gpu using PrgEnv-gnu
  [       OK ] (15/21) OSUBandwidthTest on daint:gpu using PrgEnv-pgi
  [       OK ] (16/21) OSUBandwidthTest on daint:gpu using PrgEnv-gnu
  [       OK ] (17/21) OSUAllreduceTest_8 on daint:gpu using PrgEnv-intel
  [       OK ] (18/21) OSUAllreduceTest_4 on daint:gpu using PrgEnv-intel
  [       OK ] (19/21) OSULatencyTest on daint:gpu using PrgEnv-intel
  [       OK ] (20/21) OSUAllreduceTest_2 on daint:gpu using PrgEnv-intel
  [       OK ] (21/21) OSUBandwidthTest on daint:gpu using PrgEnv-intel
  [----------] all spawned checks have finished

  [  PASSED  ] Ran 21 test case(s) from 7 check(s) (0 failure(s))
  [==========] Finished on Wed Mar 25 14:37:53 2020

Before starting running the tests, ReFrame topologically sorts them based on their dependencies and schedules them for running using the selected execution policy.
With the serial execution policy, ReFrame simply executes the tests to completion as they "arrive", since the tests are already topologically sorted.
In the asynchronous execution policy, tests are spawned and not waited for.
If a test's dependencies have not yet completed, it will not start its execution and a ``DEP`` message will be printed to denote this.

Finally, ReFrame's runtime takes care of properly cleaning up the resources of the tests respecting dependencies.
Normally when an individual test finishes successfully, its stage directory is cleaned up.
However, if other tests are depending on this one, this would be catastrophic, since most probably the dependent tests would need the outcome of this test.
ReFrame fixes that by not cleaning up the stage directory of a test until all its dependent tests have finished successfully.
