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
   :pyobject: OSUDownloadTest

This test doesn't need any specific programming environment, so we simply pick the ``builtin`` environment in the ``login`` partition.
The build tests would then copy the benchmark code and build it for the different programming environments:

.. literalinclude:: ../tutorials/deps/osu_benchmarks.py
   :pyobject: OSUBuildTest

The only new thing that comes in with the :class:`OSUBuildTest` test is the following:

.. literalinclude:: ../tutorials/deps/osu_benchmarks.py
   :pyobject: OSUBuildTest.inject_dependencies

Here we tell ReFrame that this test depends on a test named :class:`OSUDownloadTest`.
This test may or may not be defined in the same test file; all ReFrame needs is the test name.
The :func:`depends_on() <reframe.core.pipeline.RegressionTest.depends_on>` function will create dependencies between the individual test cases of the :class:`OSUBuildTest` and the :class:`OSUDownloadTest`, such that all the test cases of :class:`OSUBuildTest` will depend on the outcome of the :class:`OSUDownloadTest`.
This behaviour can be changed, but it is covered in detail in :doc:`dependencies`.
You can create arbitrary test dependency graphs, but they need to be acyclic.
If ReFrame detects cyclic dependencies, it will refuse to execute the set of tests and will issue an error pointing out the cycle.

A ReFrame test with dependencies will execute, i.e., enter its "setup" stage, only after *all* of its dependencies have succeeded.
If any of its dependencies fails, the current test will be marked as failure as well.

The next step for the :class:`OSUBuildTest` is to set its :attr:`sourcesdir` to point to the source code that was fetched by the :class:`OSUDownloadTest`.
This is achieved with the following specially decorated function:

.. literalinclude:: ../tutorials/deps/osu_benchmarks.py
   :pyobject: OSUBuildTest.set_sourcedir

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
   :start-after: # rfmdocstart: osupingpong
   :end-before: # rfmdocend: osupingpong

First, since we will have multiple similar benchmarks, we move all the common functionality to the :class:`OSUBenchmarkTestBase` base class.
Again nothing new here; we are going to use two nodes for the benchmark and we set :attr:`sourcesdir <reframe.core.pipeline.RegressionTest.sourcesdir>` to ``None``, since none of the benchmark tests will use any additional resources.
As done previously, we define the dependencies with the following:

.. literalinclude:: ../tutorials/deps/osu_benchmarks.py
   :pyobject: OSUBenchmarkTestBase.set_dependencies

Here we tell ReFrame that this test depends on a test named :class:`OSUBuildTest` "by environment."
This means that the test cases of this test will only depend on the test cases of the :class:`OSUBuildTest` that use the same environment;
partitions may be different.

The next step for the :class:`OSULatencyTest` is to set its executable to point to the binary produced by the :class:`OSUBuildTest`.
This is achieved with the following specially decorated function:

.. literalinclude:: ../tutorials/deps/osu_benchmarks.py
   :pyobject: OSULatencyTest.set_executable

This concludes the presentation of the :class:`OSULatencyTest` test. The :class:`OSUBandwidthTest` is completely analogous.

The :class:`OSUAllreduceTest` shown below is similar to the other two, except that it is parameterized.
It is essentially a scalability test that is running the ``osu_allreduce`` executable created by the :class:`OSUBuildTest` for 2, 4, 8 and 16 nodes.

.. literalinclude:: ../tutorials/deps/osu_benchmarks.py
   :pyobject: OSUAllreduceTest

The full set of OSU example tests is shown below:

.. literalinclude:: ../tutorials/deps/osu_benchmarks.py

Notice that the order in which dependencies are defined in a test file is irrelevant.
In this case, we define :class:`OSUBuildTest` at the end.
ReFrame will make sure to properly sort the tests and execute them.

Here is the output when running the OSU tests with the asynchronous execution policy:

.. code-block:: console

   ./bin/reframe -c tutorials/deps/osu_benchmarks.py -r

.. literalinclude:: listings/osu_bench_deps.txt
   :language: console

Before starting running the tests, ReFrame topologically sorts them based on their dependencies and schedules them for running using the selected execution policy.
With the serial execution policy, ReFrame simply executes the tests to completion as they "arrive," since the tests are already topologically sorted.
In the asynchronous execution policy, tests are spawned and not waited for.
If a test's dependencies have not yet completed, it will not start its execution immediately.

ReFrame's runtime takes care of properly cleaning up the resources of the tests respecting dependencies.
Normally when an individual test finishes successfully, its stage directory is cleaned up.
However, if other tests are depending on this one, this would be catastrophic, since most probably the dependent tests would need the outcome of this test.
ReFrame fixes that by not cleaning up the stage directory of a test until all its dependent tests have finished successfully.

When selecting tests using the test filtering options, such as the :option:`-t`, :option:`-n` etc., ReFrame will automatically select any dependencies of these tests as well.
For example, if we select only the :class:`OSULatencyTest` for running, ReFrame will also select the :class:`OSUBuildTest` and the :class:`OSUDownloadTest`:

.. code-block:: console

   ./bin/reframe -c tutorials/deps/osu_benchmarks.py -n OSULatencyTest -l

.. literalinclude:: listings/osu_latency_list.txt
   :language: console

Finally, when ReFrame cannot resolve a dependency of a test, it will issue a warning and skip completely all the test cases that recursively depend on this one.
In the following example, we restrict the run of the :class:`OSULatencyTest` to the ``daint:gpu`` partition.
This is problematic, since its dependencies cannot run on this partition and, particularly, the :class:`OSUDownloadTest`.
As a result, its immediate dependency :class:`OSUBuildTest` will be skipped, which will eventually cause all combinations of the :class:`OSULatencyTest` to be skipped.

.. code-block:: console

   ./bin/reframe -c tutorials/deps/osu_benchmarks.py --system=daint:gpu -n OSULatencyTest -l

.. literalinclude:: listings/osu_latency_unresolved_deps.txt
   :language: console


Listing Dependencies
--------------------

As shown in the listing of :class:`OSULatencyTest` before, the full dependency chain of the test is listed along with the test.
Each target dependency is printed in a new line prefixed by the ``^`` character and indented proportionally to its level.
If a target dependency appears in multiple paths, it will only be listed once.

The default test listing will list the dependencies at the test level or the *conceptual* dependencies.
ReFrame generates multiple test cases from each test depending on the target system configuration.
We have seen in the :doc:`tutorial_basics` already how the STREAM benchmark generated many more test cases when it was run in a HPC system with multiple partitions and programming environments.
These are the *actual* depedencies and form the actual test case graph that will be executed by the runtime.
The mapping of a test to its concrete test cases that will be executed on a system is called *test concretization*.
You can view the exact concretization of the selected tests with ``--list=concretized`` or simply ``-lC``.
Here is how the OSU benchmarks of this tutorial are concretized on the system ``daint``:


.. code-block:: console

   ./bin/reframe -c tutorials/deps/osu_benchmarks.py -lC

.. literalinclude:: listings/osu_bench_list_concretized.txt
   :language: console

Notice how the various test cases of the run benchmarks depend on the corresponding test cases of the build tests.

The concretization of test cases changes if a specifc partition or programming environment is passed from the command line or, of course, if the test is run on a different system.
If we scope our programming environments to ``gnu`` and ``builtin`` only, ReFrame will generate 8 test cases only instead of 22:

.. note::

   If we do not select the ``builtin`` environment, we will end up with a dangling dependency as in the example above and ReFrame will skip all the dependent test cases.


.. code-block:: console

   ./bin/reframe -c tutorials/deps/osu_benchmarks.py -n OSULatencyTest -L -p builtin -p gnu

.. literalinclude:: listings/osu_bench_list_concretized_gnu.txt
   :language: console


To gain a deeper understanding on how test dependencies work in Reframe, please refer to :doc:`dependencies`.


.. _param_deps:

Depending on Parameterized Tests
--------------------------------

As shown earlier in this section, tests define their dependencies by referencing the target tests by their unique name.
This is straightforward when referring to regular tests, where their name matches the class name, but it becomes cumbersome trying to refer to a parameterized tests, since no safe assumption should be made as of the variant number of the test or how the parameters are encoded in the name.
In order to safely and reliably refer to a parameterized test, you should use the :func:`~reframe.core.pipeline.RegressionMixin.get_variant_nums` and :func:`~reframe.core.pipeline.RegressionMixin.variant_name` class methods as shown in the following example:

.. literalinclude:: ../tutorials/deps/parameterized.py
   :emphasize-lines: 37-

In this example, :class:`TestB` depends only on selected variants of :class:`TestA`.
The :func:`get_variant_nums` method accepts a set of key-value pairs representing the target test parameters and selector functions and returns the list of the variant numbers that correspond to these variants.
Using the :func:`variant_name` subsequently, we can get the actual name of the variant.


.. code-block:: console

   ./bin/reframe -c tutorials/deps/parameterized.py -l

.. literalinclude:: listings/param_deps_list.txt
   :language: console
