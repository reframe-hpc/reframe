===============================================
Tutorial 3: Using Dependencies in ReFrame Tests
===============================================

.. versionadded:: 2.21


A ReFrame test may define dependencies to other tests.
An example scenario is to test different runtime configurations of a benchmark that you need to compile, or run a scaling analysis of a code.
In such cases, you don't want to download and rebuild your test for each runtime configuration.
You could have a test where only the sources are fetched, and which all build tests would depend on.
And, similarly, all the runtime tests would depend on their corresponding build test.
This is the approach we take with the following example, that fetches, builds and runs several `OSU benchmarks <http://mvapich.cse.ohio-state.edu/benchmarks/>`__.
We first create a basic run-only test, that fetches the benchmarks:

.. code-block:: console

   cat tutorials/deps/osu_benchmarks.py


.. literalinclude:: ../tutorials/deps/osu_benchmarks.py
   :lines: 110-122

This test doesn't need any specific programming environment, so we simply pick the ``builtin`` environment in the ``login`` partition.
The build tests would then copy the benchmark code and build it for the different programming environments:

.. literalinclude:: ../tutorials/deps/osu_benchmarks.py
   :lines: 91-107

The only new thing that comes in with the :class:`OSUBuildTest` test is the following line:

.. literalinclude:: ../tutorials/deps/osu_benchmarks.py
   :lines: 97

Here we tell ReFrame that this test depends on a test named :class:`OSUDownloadTest`.
This test may or may not be defined in the same test file; all ReFrame needs is the test name.
The :func:`depends_on() <reframe.core.pipeline.RegressionTest.depends_on>` function will create dependencies between the individual test cases of the :class:`OSUBuildTest` and the :class:`OSUDownloadTest`, such that all the test case of :class:`OSUBuildTest` will depend on the outcome of the :class:`OSUDownloadTest`.
This behaviour can be changed, but it is covered in detail in :doc:`dependencies`.
You can create arbitrary test dependency graphs, but they need to be acyclic.
If ReFrame detects cyclic dependencies, it will refuse to execute the set of tests and will issue an error pointing out the cycle.

A ReFrame test with dependencies will execute, i.e., enter its "setup" stage, only after *all* of its dependencies have succeeded.
If any of its dependencies fails, the current test will be marked as failure as well.

The next step for the :class:`OSUBuildTest` is to set its :attr:`sourcesdir` to point to the source code that was fetched by the :class:`OSUDownloadTest`.
This is achieved with the following specially decorated function:

.. literalinclude:: ../tutorials/deps/osu_benchmarks.py
   :lines: 102-107

The :func:`@require_deps <reframe.core.decorators.require_deps>` decorator binds each argument of the decorated function to the corresponding target dependency.
In order for the binding to work correctly the function arguments must be named after the target dependencies.
Referring to a dependency only by the test's name is not enough, since a test might be associated with multiple programming environments.
For this reason, each dependency argument is actually bound to a function that accepts as argument the name of the target partition and target programming environment.
If no arguments are passed, the current programming environment is implied, such that ``OSUDownloadTest()`` is equivalent to ``OSUDownloadTest(self.current_environ.name, self.current_partition.name)``.
In this case, since both the partition and environment of the target dependency do not much those of the current test, we need to specify both.

This call returns the actual test case of the dependency that has been executed.
This allows you to access any attribute from the target test, as we do in this example by accessing the target test's stage directory, which we use to construct the sourcesdir of the test.

For the next test we need to use the OSU benchmark binaries that we just built, so as to run the MPI ping-pong benchmark.
Here is the relevant part:

.. literalinclude:: ../tutorials/deps/osu_benchmarks.py
   :lines: 13-44

First, since we will have multiple similar benchmarks, we move all the common functionality to the :class:`OSUBenchmarkTestBase` base class.
Again nothing new here; we are going to use two nodes for the benchmark and we set :attr:`sourcesdir <reframe.core.pipeline.RegressionTest.sourcesdir>` to ``None``, since none of the benchmark tests will use any additional resources.
As done previously, we define the dependencies with the the following line:

.. literalinclude:: ../tutorials/deps/osu_benchmarks.py
   :lines: 23

Here we tell ReFrame that this test depends on a test named :class:`OSUBuildTest` "by environment."
This means that the test cases of this test will only depend on the test cases of the :class:`OSUBuildTest` that use the same environment;
partitions may be different.

The next step for the :class:`OSULatencyTest` is to set its executable to point to the binary produced by the :class:`OSUBuildTest`.
This is achieved with the following specially decorated function:

.. literalinclude:: ../tutorials/deps/osu_benchmarks.py
   :lines: 38-44

This concludes the presentation of the :class:`OSULatencyTest` test. The :class:`OSUBandwidthTest` is completely analogous.

The :class:`OSUAllreduceTest` shown below is similar to the other two, except that it is parameterized.
It is essentially a scalability test that is running the ``osu_allreduce`` executable created by the :class:`OSUBuildTest` for 2, 4, 8 and 16 nodes.

.. literalinclude:: ../tutorials/deps/osu_benchmarks.py
   :lines: 69-88

The full set of OSU example tests is shown below:

.. literalinclude:: ../tutorials/deps/osu_benchmarks.py

Notice that the order in which dependencies are defined in a test file is irrelevant.
In this case, we define :class:`OSUBuildTest` at the end.
ReFrame will make sure to properly sort the tests and execute them.

Here is the output when running the OSU tests with the asynchronous execution policy:

.. code-block:: none

   [ReFrame Setup]
     version:           3.4-dev2 (rev: 56c6c237)
     command:           './bin/reframe --system=daint -C tutorials/config/settings.py -c tutorials/deps/osu_benchmarks.py -r'
     launched by:       user@daint101
     working directory: '/users/user/Devel/reframe'
     settings file:     'tutorials/config/settings.py'
     check search path: '/users/user/Devel/reframe/tutorials/deps/osu_benchmarks.py'
     stage directory:   '/users/user/Devel/reframe/stage'
     output directory:  '/users/user/Devel/reframe/output'

   [==========] Running 8 check(s)
   [==========] Started on Mon Jan 25 19:34:09 2021

   [----------] started processing OSUDownloadTest (OSU benchmarks download sources)
   [ RUN      ] OSUDownloadTest on daint:login using builtin
   [----------] finished processing OSUDownloadTest (OSU benchmarks download sources)

   [----------] started processing OSUBuildTest (OSU benchmarks build test)
   [ RUN      ] OSUBuildTest on daint:gpu using gnu
   [      DEP ] OSUBuildTest on daint:gpu using gnu
   [ RUN      ] OSUBuildTest on daint:gpu using intel
   [      DEP ] OSUBuildTest on daint:gpu using intel
   [ RUN      ] OSUBuildTest on daint:gpu using pgi
   [      DEP ] OSUBuildTest on daint:gpu using pgi
   [----------] finished processing OSUBuildTest (OSU benchmarks build test)

   [----------] started processing OSULatencyTest (OSU latency test)
   [ RUN      ] OSULatencyTest on daint:gpu using gnu
   [      DEP ] OSULatencyTest on daint:gpu using gnu
   [ RUN      ] OSULatencyTest on daint:gpu using intel
   [      DEP ] OSULatencyTest on daint:gpu using intel
   [ RUN      ] OSULatencyTest on daint:gpu using pgi
   [      DEP ] OSULatencyTest on daint:gpu using pgi
   [----------] finished processing OSULatencyTest (OSU latency test)

   [----------] started processing OSUBandwidthTest (OSU bandwidth test)
   [ RUN      ] OSUBandwidthTest on daint:gpu using gnu
   [      DEP ] OSUBandwidthTest on daint:gpu using gnu
   [ RUN      ] OSUBandwidthTest on daint:gpu using intel
   [      DEP ] OSUBandwidthTest on daint:gpu using intel
   [ RUN      ] OSUBandwidthTest on daint:gpu using pgi
   [      DEP ] OSUBandwidthTest on daint:gpu using pgi
   [----------] finished processing OSUBandwidthTest (OSU bandwidth test)

   [----------] started processing OSUAllreduceTest_2 (OSU Allreduce test)
   [ RUN      ] OSUAllreduceTest_2 on daint:gpu using gnu
   [      DEP ] OSUAllreduceTest_2 on daint:gpu using gnu
   [ RUN      ] OSUAllreduceTest_2 on daint:gpu using intel
   [      DEP ] OSUAllreduceTest_2 on daint:gpu using intel
   [ RUN      ] OSUAllreduceTest_2 on daint:gpu using pgi
   [      DEP ] OSUAllreduceTest_2 on daint:gpu using pgi
   [----------] finished processing OSUAllreduceTest_2 (OSU Allreduce test)

   [----------] started processing OSUAllreduceTest_4 (OSU Allreduce test)
   [ RUN      ] OSUAllreduceTest_4 on daint:gpu using gnu
   [      DEP ] OSUAllreduceTest_4 on daint:gpu using gnu
   [ RUN      ] OSUAllreduceTest_4 on daint:gpu using intel
   [      DEP ] OSUAllreduceTest_4 on daint:gpu using intel
   [ RUN      ] OSUAllreduceTest_4 on daint:gpu using pgi
   [      DEP ] OSUAllreduceTest_4 on daint:gpu using pgi
   [----------] finished processing OSUAllreduceTest_4 (OSU Allreduce test)

   [----------] started processing OSUAllreduceTest_8 (OSU Allreduce test)
   [ RUN      ] OSUAllreduceTest_8 on daint:gpu using gnu
   [      DEP ] OSUAllreduceTest_8 on daint:gpu using gnu
   [ RUN      ] OSUAllreduceTest_8 on daint:gpu using intel
   [      DEP ] OSUAllreduceTest_8 on daint:gpu using intel
   [ RUN      ] OSUAllreduceTest_8 on daint:gpu using pgi
   [      DEP ] OSUAllreduceTest_8 on daint:gpu using pgi
   [----------] finished processing OSUAllreduceTest_8 (OSU Allreduce test)

   [----------] started processing OSUAllreduceTest_16 (OSU Allreduce test)
   [ RUN      ] OSUAllreduceTest_16 on daint:gpu using gnu
   [      DEP ] OSUAllreduceTest_16 on daint:gpu using gnu
   [ RUN      ] OSUAllreduceTest_16 on daint:gpu using intel
   [      DEP ] OSUAllreduceTest_16 on daint:gpu using intel
   [ RUN      ] OSUAllreduceTest_16 on daint:gpu using pgi
   [      DEP ] OSUAllreduceTest_16 on daint:gpu using pgi
   [----------] finished processing OSUAllreduceTest_16 (OSU Allreduce test)

   [----------] waiting for spawned checks to finish
   [       OK ] ( 1/22) OSUDownloadTest on daint:login using builtin [compile: 0.006s run: 1.272s total: 1.349s]
   [       OK ] ( 2/22) OSUBuildTest on daint:gpu using gnu [compile: 21.474s run: 0.043s total: 86.844s]
   [       OK ] ( 3/22) OSUBuildTest on daint:gpu using pgi [compile: 27.948s run: 58.876s total: 86.842s]
   [       OK ] ( 4/22) OSUAllreduceTest_2 on daint:gpu using pgi [compile: 0.007s run: 20.752s total: 36.777s]
   [       OK ] ( 5/22) OSUAllreduceTest_8 on daint:gpu using gnu [compile: 0.007s run: 28.699s total: 36.779s]
   [       OK ] ( 6/22) OSUAllreduceTest_16 on daint:gpu using gnu [compile: 0.006s run: 34.055s total: 36.785s]
   [       OK ] ( 7/22) OSUBuildTest on daint:gpu using intel [compile: 37.314s run: 58.469s total: 123.772s]
   [       OK ] ( 8/22) OSULatencyTest on daint:gpu using pgi [compile: 0.009s run: 29.095s total: 56.517s]
   [       OK ] ( 9/22) OSUAllreduceTest_2 on daint:gpu using gnu [compile: 0.006s run: 37.876s total: 56.534s]
   [       OK ] (10/22) OSUAllreduceTest_4 on daint:gpu using pgi [compile: 0.007s run: 45.804s total: 56.563s]
   [       OK ] (11/22) OSUAllreduceTest_16 on daint:gpu using pgi [compile: 0.007s run: 56.553s total: 56.580s]
   [       OK ] (12/22) OSULatencyTest on daint:gpu using gnu [compile: 0.009s run: 27.131s total: 57.330s]
   [       OK ] (13/22) OSUAllreduceTest_8 on daint:gpu using pgi [compile: 0.007s run: 51.868s total: 57.292s]
   [       OK ] (14/22) OSUAllreduceTest_4 on daint:gpu using gnu [compile: 0.007s run: 44.443s total: 57.803s]
   [       OK ] (15/22) OSUBandwidthTest on daint:gpu using pgi [compile: 0.015s run: 75.905s total: 97.177s]
   [       OK ] (16/22) OSUBandwidthTest on daint:gpu using gnu [compile: 0.019s run: 82.091s total: 106.348s]
   [       OK ] (17/22) OSUAllreduceTest_16 on daint:gpu using intel [compile: 0.006s run: 89.678s total: 89.699s]
   [       OK ] (18/22) OSUAllreduceTest_4 on daint:gpu using intel [compile: 0.006s run: 113.071s total: 121.153s]
   [       OK ] (19/22) OSUAllreduceTest_2 on daint:gpu using intel [compile: 0.006s run: 110.686s total: 121.408s]
   [       OK ] (20/22) OSUAllreduceTest_8 on daint:gpu using intel [compile: 0.006s run: 119.416s total: 122.079s]
   [       OK ] (21/22) OSULatencyTest on daint:gpu using intel [compile: 0.008s run: 133.892s total: 149.776s]
   [       OK ] (22/22) OSUBandwidthTest on daint:gpu using intel [compile: 0.006s run: 170.584s total: 183.903s]
   [----------] all spawned checks have finished

   [  PASSED  ] Ran 22 test case(s) from 8 check(s) (0 failure(s))
   [==========] Finished on Mon Jan 25 19:39:18 2021
   Log file(s) saved in: '/tmp/rfm-g1a6axrf.log'

Before starting running the tests, ReFrame topologically sorts them based on their dependencies and schedules them for running using the selected execution policy.
With the serial execution policy, ReFrame simply executes the tests to completion as they "arrive," since the tests are already topologically sorted.
In the asynchronous execution policy, tests are spawned and not waited for.
If a test's dependencies have not yet completed, it will not start its execution and a ``DEP`` message will be printed to denote this.

ReFrame's runtime takes care of properly cleaning up the resources of the tests respecting dependencies.
Normally when an individual test finishes successfully, its stage directory is cleaned up.
However, if other tests are depending on this one, this would be catastrophic, since most probably the dependent tests would need the outcome of this test.
ReFrame fixes that by not cleaning up the stage directory of a test until all its dependent tests have finished successfully.

When selecting tests using the test filtering options, such as the :option:`-t`, :option:`-n` etc., ReFrame will automatically select any dependencies of these tests as well.
For example, if we select only the :class:`OSULatencyTest` for running, ReFrame will also select the :class:`OSUBuildTest` and the :class:`OSUDownloadTest`:

.. code-block:: console

   ./bin/reframe -c tutorials/deps/osu_benchmarks.py -n OSULatencyTest -l


.. code-block:: none

   $ ./bin/reframe -C -c tutorials/deps/osu_benchmarks.py -n OSULatencyTest -l
   [ReFrame Setup]
     version:           3.3-dev2 (rev: 8ded20cd)
     command:           './bin/reframe -C tutorials/config/settings.py -c tutorials/deps/osu_benchmarks.py -n OSULatencyTest -l'
     launched by:       user@daint101
     working directory: '/users/user/Devel/reframe'
     settings file:     'tutorials/config/settings.py'
     check search path: '/users/user/Devel/reframe/tutorials/deps/osu_benchmarks.py'
     stage directory:   '/users/user/Devel/reframe/stage'
     output directory:  '/users/user/Devel/reframe/output'

   [List of matched checks]
   - OSUDownloadTest (found in '/users/user/Devel/reframe/tutorials/deps/osu_benchmarks.py')
   - OSUBuildTest (found in '/users/user/Devel/reframe/tutorials/deps/osu_benchmarks.py')
   - OSULatencyTest (found in '/users/user/Devel/reframe/tutorials/deps/osu_benchmarks.py')
   Found 3 check(s)
   Log file(s) saved in: '/tmp/rfm-4c15g820.log'


Finally, when ReFrame cannot resolve a dependency of a test, it will issue a warning and skip completely all the test cases that recursively depend on this one.
In the following example, we restrict the run of the :class:`OSULatencyTest` to the ``daint:gpu`` partition.
This is problematic, since its dependencies cannot run on this partition and, particularly, the :class:`OSUDownloadTest`.
As a result, its immediate dependency :class:`OSUBuildTest` will be skipped, which will eventually cause all combinations of the :class:`OSULatencyTest` to be skipped.

.. code-block:: console

   ./bin/reframe -c tutorials/deps/osu_benchmarks.py -n OSULatencyTest -l

.. code-block:: none

   [ReFrame Setup]
     version:           3.5-dev0 (rev: 93948510)
     command:           './bin/reframe -C tutorials/config/settings.py --system=daint:gpu -c tutorials/deps/osu_benchmarks.py -l'
     launched by:       user@daint101
     working directory: '/users/user/Devel/reframe'
     settings file:     'tutorials/config/settings.py'
     check search path: '/users/user/Devel/reframe/tutorials/deps/osu_benchmarks.py'
     stage directory:   '/users/user/Devel/reframe/stage'
     output directory:  '/users/user/Devel/reframe/output'

   ./bin/reframe: could not resolve dependency: ('OSUBuildTest', 'daint:gpu', 'gnu') -> 'OSUDownloadTest'
   ./bin/reframe: could not resolve dependency: ('OSUBuildTest', 'daint:gpu', 'intel') -> 'OSUDownloadTest'
   ./bin/reframe: could not resolve dependency: ('OSUBuildTest', 'daint:gpu', 'pgi') -> 'OSUDownloadTest'
   ./bin/reframe: skipping all dependent test cases
     - ('OSUBuildTest', 'daint:gpu', 'intel')
     - ('OSUBuildTest', 'daint:gpu', 'pgi')
     - ('OSUAllreduceTest_4', 'daint:gpu', 'intel')
     - ('OSUAllreduceTest_2', 'daint:gpu', 'intel')
     - ('OSULatencyTest', 'daint:gpu', 'pgi')
     - ('OSUBandwidthTest', 'daint:gpu', 'pgi')
     - ('OSUAllreduceTest_8', 'daint:gpu', 'intel')
     - ('OSUAllreduceTest_16', 'daint:gpu', 'pgi')
     - ('OSUBuildTest', 'daint:gpu', 'gnu')
     - ('OSUBandwidthTest', 'daint:gpu', 'intel')
     - ('OSULatencyTest', 'daint:gpu', 'intel')
     - ('OSUAllreduceTest_16', 'daint:gpu', 'intel')
     - ('OSUAllreduceTest_8', 'daint:gpu', 'pgi')
     - ('OSULatencyTest', 'daint:gpu', 'gnu')
     - ('OSUBandwidthTest', 'daint:gpu', 'gnu')
     - ('OSUAllreduceTest_2', 'daint:gpu', 'pgi')
     - ('OSUAllreduceTest_4', 'daint:gpu', 'pgi')
     - ('OSUAllreduceTest_8', 'daint:gpu', 'gnu')
     - ('OSUAllreduceTest_4', 'daint:gpu', 'gnu')
     - ('OSUAllreduceTest_2', 'daint:gpu', 'gnu')
     - ('OSUAllreduceTest_16', 'daint:gpu', 'gnu')

   [List of matched checks]

   Found 0 check(s)
   Log file(s) saved in: '/tmp/rfm-hjit66h2.log'


Listing Dependencies
--------------------

You can view the dependencies of a test by using the :option:`-L` option:


.. code-block:: console

   ./bin/reframe -c tutorials/deps/osu_benchmarks.py -n OSULatencyTest -L


.. code-block:: none

   < ... omitted ... >

   - OSULatencyTest:
       Description:
         OSU latency test

       Environment modules:
         <none>

       Location:
         /users/user/Devel/reframe/tutorials/deps/osu_benchmarks.py

       Maintainers:
         <none>

       Node allocation:
         standard (2 task(s))

       Pipeline hooks:
         - post_setup: set_executable

       Tags:
         <none>

       Valid environments:
         gnu, pgi, intel

       Valid systems:
         daint:gpu

       Dependencies (conceptual):
         OSUBuildTest

       Dependencies (actual):
         - ('OSULatencyTest', 'daint:gpu', 'gnu') -> ('OSUBuildTest', 'daint:login', 'gnu')
         - ('OSULatencyTest', 'daint:gpu', 'intel') -> ('OSUBuildTest', 'daint:login', 'intel')
         - ('OSULatencyTest', 'daint:gpu', 'pgi') -> ('OSUBuildTest', 'daint:login', 'pgi')

   < ... omitted ... >


Dependencies are not only listed conceptually, e.g., "test A depends on test B," but also in a way that shows how they are actually interpreted between the different test cases of the tests.
The test dependencies do not change conceptually, but their actual interpretation might change from system to system or from programming environment to programming environment.
The following listing shows how the actual test cases dependencies are formed when we select only the ``gnu`` and ``builtin`` programming environment for running:

.. note::

   If we do not select the ``builtin`` environment, we will end up with a dangling dependency as in the example above and ReFrame will skip all the dependent test cases.


.. code-block:: console

   ./bin/reframe -c tutorials/deps/osu_benchmarks.py -n OSULatencyTest -L -p builtin -p gnu


.. code-block:: none
   :emphasize-lines: 35

   < ... omitted ... >

   - OSULatencyTest:
       Description:
         OSU latency test

       Environment modules:
         <none>

       Location:
         /users/user/Devel/reframe/tutorials/deps/osu_benchmarks.py

       Maintainers:
         <none>

       Node allocation:
         standard (2 task(s))

       Pipeline hooks:
         - post_setup: set_executable

       Tags:
         <none>

       Valid environments:
         gnu, pgi, intel

       Valid systems:
         daint:gpu

       Dependencies (conceptual):
         OSUBuildTest

       Dependencies (actual):
         - ('OSULatencyTest', 'daint:gpu', 'gnu') -> ('OSUBuildTest', 'daint:login', 'gnu')

   < ... omitted ... >

For more information on test dependencies, you can have a look at :doc:`dependencies`.
