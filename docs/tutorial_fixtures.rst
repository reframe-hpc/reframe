Tutorial 4: Using Test Fixtures
===============================

.. versionadded:: 3.9.0

A fixture in ReFrame is a test that manages a resource of another test.
Fixtures can be chained to create essentially a graph of dependencies.
Similarly to test dependencies, the test that uses the fixture will not execute until its fixture has executed.
In this tutorial, we will rewrite the OSU benchmarks example presented in :doc:`tutorial_deps` using fixtures.
We will cover only the basic concepts of fixtures that will allow you to start using them in your tests.
For the full documentation of the test fixtures, you should refer to the :doc:`regression_test_api` documentation.

The full example of the OSU benchmarks using test fixtures is shown below with the relevant parts highlighted:

.. literalinclude:: ../tutorials/fixtures/osu_benchmarks.py
   :lines: 6-
   :emphasize-lines: 6-43,53,67-68,85-86,106-107

Let's start from the leaf tests, i.e. the tests that execute the benchmarks (:class:`osu_latency_test`, :class:`osu_bandwidth_test` and :class:`osu_allreduce_test`).
As in the dependencies example, all these tests derive from the :class:`OSUBenchmarkTestBase`, where we define a fixture that will take care of generating the binaries of the tests:

.. literalinclude:: ../tutorials/fixtures/osu_benchmarks.py
   :lines: 58

A test defines a fixture using the :func:`~reframe.core.pipeline.RegressionMixin.fixture` builtin and assigns it a name by assigning the return value of the builtin to a test variable, here ``osu_binaries``.
This name will be used later to access the resource managed by the fixture.

As stated previously, a fixture is another full-fledged ReFrame test, here the :class:`build_osu_benchmarks` which will take care of building the OSU benchmarks.
Each fixture is associated with a scope.
This practically indicates at which level a fixture is shared with other tests.
There are four fixture scopes, which are listed below in decreasing order of generality:

- ``session``: A fixture with this scope will be executed once per a ReFrame run session and will be shared across the whole run.
- ``partition``: A fixture with this scope will be executed once per partition and will be shared across all tests that run in that partition.
- ``environment``: A fixture with this scope will be executed once per partition and environment combination and will be shared across all tests that run with this partition and environment combination.
- ``test``: A fixture with this scope is private to the test and will be executed for each test case.

In this example, we need to build once the OSU benchmarks for each partition and environment combination, so we use the ``environment`` scope.

Accessing the fixture is very straightforward.
The fixture's result is accessible after the *setup* pipeline stage through the corresponding variable in the test that is defining it.
Since a fixture is a standard ReFrame test, you can access any information of the test.
The individual benchmarks do exactly that:

.. literalinclude:: ../tutorials/fixtures/osu_benchmarks.py
   :lines: 69-76
   :emphasize-lines: 4-5

Here we construct the final executable path by accessing the standard :attr:`~reframe.core.pipeline.RegressionTest.stagedir` attribute of the test as well as the custom-defined :attr:`build_prefix` variable of the :class:`build_osu_benchmarks` fixture.

Let's inspect now the :class:`build_osu_benchmarks` fixture:

.. literalinclude:: ../tutorials/fixtures/osu_benchmarks.py
   :lines: 25-49
   :emphasize-lines: 5,9,12

It is obvious that it is a normal ReFrame test except that it does not need to be decorated with the :func:`@simple_test <reframe.core.decorators.simple_test>` decorator.
This means that the test will only be executed if it is a fixture of another test.
If it was decorated, it would be executed both as a standalone test and as a fixture of another test.
Another detail is that this test does dot define the :attr:`~reframe.core.pipeline.RegressionTest.valid_systems` and :attr:`~reframe.core.pipeline.RegressionTest.valid_prog_environs` variables.
Fixtures inherit those variables from the test that owns them depending on the scope.

Similarly to :class:`OSUBenchmarkTestBase`, this test uses a fixture that fetches the OSU benchmarks sources.
We could fetch the OSU benchmarks in this test, but we choose to separate the two primarily for demonstration purposes, but it would also make sense in cases that the data fetch is too slow.

The ``osu_benchmarks`` fixture is defined at session scope, since we only need to download the benchmarks once for the whole session:

.. literalinclude:: ../tutorials/fixtures/osu_benchmarks.py
   :lines: 58

The rest of the test is very straightforward.

Let's inspect the last fixture, the :class:`fetch_osu_benchmarks`:

.. literalinclude:: ../tutorials/fixtures/osu_benchmarks.py
   :lines: 11-22
   :emphasize-lines: 8

There is nothing special to this test -- it is just an ordinary test -- except that we force it to execute locally by setting its :attr:`~reframe.core.pipeline.RegressionTest.local` variable.
The reason for that is that a fixture at session scope can execute with any partition/environment combination, so ReFrame could have to spawn a job in case it has chosen a remote partition to launch this fixture on.
For this reason, we simply force it execute locally regardless of the chosen partition.

It is now time to run the new tests, but let us first list them:

.. code-block:: bash

   export RFM_CONFIG_FILE=$(pwd)/tutorials/config/settings.py
   reframe -c tutorials/fixtures/osu_benchmarks.py -l

.. code-block:: console

   [ReFrame Setup]
     version:           3.9.0
     command:           'reframe -c tutorials/fixtures/osu_benchmarks.py -l'
     launched by:       user@daint106
     working directory: '/users/user/Devel/reframe'
     settings file:     '/users/user/Devel/reframe/tutorials/config/settings.py'
     check search path: '/users/user/Devel/reframe/tutorials/fixtures/osu_benchmarks.py'
     stage directory:   '/users/user/Devel/reframe/stage'
     output directory:  '/users/user/Devel/reframe/output'

   [List of matched checks]
   - osu_latency_test (found in '/users/user/Devel/reframe/tutorials/fixtures/osu_benchmarks.py')
   - osu_allreduce_test_8 (found in '/users/user/Devel/reframe/tutorials/fixtures/osu_benchmarks.py')
   - osu_allreduce_test_2 (found in '/users/user/Devel/reframe/tutorials/fixtures/osu_benchmarks.py')
   - osu_allreduce_test_4 (found in '/users/user/Devel/reframe/tutorials/fixtures/osu_benchmarks.py')
   - osu_bandwidth_test (found in '/users/user/Devel/reframe/tutorials/fixtures/osu_benchmarks.py')
   - osu_allreduce_test_16 (found in '/users/user/Devel/reframe/tutorials/fixtures/osu_benchmarks.py')
   Found 6 check(s)

   Log file(s) saved in '/tmp/rfm-dlkc1vb_.log'

Notice that only the leaf tests are listed and not their fixtures.
Listing the tests in detailed mode, however, using the :option:`-L` option, you will see all the generated fixtures:


.. code-block:: bash

   reframe -c tutorials/fixtures/osu_benchmarks.py -n osu_bandwidth_test -L

ReFrame will generate 4 fixtures for this test based on the partition and environment configurations for the current system.
The following figure shows the generated fixtures as well as their conceptual dependencies.

.. figure:: _static/img/fixtures-conceptual-deps.svg
  :align: center

  :sub:`Expanded fixtures and dependencies for the OSU benchmarks example.`

Notice how the :class:`build_osu_benchmarks` fixture is populated three times, for each partition and environment combination, and the :class:`fetch_osu_benchmarks` is generated only once.
Tests in a single ReFrame session must have unique names, so the fixture class name is mangled by the framework to generate a unique name in the test dependency DAG.
A *scope* part is added to the base name of the fixture, which in this figure is indicated with red color.

Under the hood, fixtures use the test dependency mechanism which is described in :doc:`dependencies`.
The dependencies shown in the previous figure are conceptual.
A single test in ReFrame generates a series of test cases for all the combinations of valid systems and valid programming environments and the actual dependencies are expressed in this more fine-grained layer, which is also the layer at which the execution of tests is scheduled.

The following figure shows how the above graph translates into the actual DAG of test cases.

.. figure:: _static/img/fixtures-actual-deps.svg
  :align: center

  :sub:`The actual dependencies for the OSU benchmarks example using fixtures.`


The first thing to notice here is how the individual test cases of :class:`osu_bandwidth_test` depend only the specific fixtures for their scope:
when :class:`osu_bandwidth_test` runs on the ``daint:gpu`` partition using the ``gnu`` compiler it will only depend on the :class:`build_osu_benchmarks~daint:gpu+gnu` fixture.
The second thing to notice is where the :class:`fetch_osu_benchmarks~daint` fixture will run.
Since this is a *session* fixture, ReFrame has arbitrarily chosen to run it on ``daint:gpu`` using the ``gnu`` environment.
A session fixture can run on any combination of valid partitions and environments.
The following figure shows how the test dependency DAG is concretized when we scope the valid programming environments from the command line using ``-p pgi``.


.. figure:: _static/img/fixtures-actual-deps-scoped.svg
  :align: center

  :sub:`The dependency graph concretized for the 'pgi' environment only.`


Notice how the :class:`fetch_osu_benchmarks~daint` fixture is selected to run in the only valid partition/environment combination.

The following listing shows the output of running the tutorial examples.

.. code-block:: console

   [==========] Running 10 check(s)
   [==========] Started on Sun Oct 31 22:00:28 2021

   [----------] started processing fetch_osu_benchmarks~daint (Fetch OSU benchmarks)
   [ RUN      ] fetch_osu_benchmarks~daint on daint:gpu using gnu
   [----------] finished processing fetch_osu_benchmarks~daint (Fetch OSU benchmarks)

   [----------] started processing build_osu_benchmarks~daint:gpu+gnu (Build OSU benchmarks)
   [ RUN      ] build_osu_benchmarks~daint:gpu+gnu on daint:gpu using gnu
   [      DEP ] build_osu_benchmarks~daint:gpu+gnu on daint:gpu using gnu
   [----------] finished processing build_osu_benchmarks~daint:gpu+gnu (Build OSU benchmarks)

   [----------] started processing build_osu_benchmarks~daint:gpu+intel (Build OSU benchmarks)
   [ RUN      ] build_osu_benchmarks~daint:gpu+intel on daint:gpu using intel
   [      DEP ] build_osu_benchmarks~daint:gpu+intel on daint:gpu using intel
   [----------] finished processing build_osu_benchmarks~daint:gpu+intel (Build OSU benchmarks)

   [----------] started processing build_osu_benchmarks~daint:gpu+pgi (Build OSU benchmarks)
   [ RUN      ] build_osu_benchmarks~daint:gpu+pgi on daint:gpu using pgi
   [      DEP ] build_osu_benchmarks~daint:gpu+pgi on daint:gpu using pgi
   [----------] finished processing build_osu_benchmarks~daint:gpu+pgi (Build OSU benchmarks)

   [----------] started processing osu_allreduce_test_16 (OSU Allreduce test)
   [ RUN      ] osu_allreduce_test_16 on daint:gpu using gnu
   [      DEP ] osu_allreduce_test_16 on daint:gpu using gnu
   [ RUN      ] osu_allreduce_test_16 on daint:gpu using intel
   [      DEP ] osu_allreduce_test_16 on daint:gpu using intel
   [ RUN      ] osu_allreduce_test_16 on daint:gpu using pgi
   [      DEP ] osu_allreduce_test_16 on daint:gpu using pgi
   [----------] finished processing osu_allreduce_test_16 (OSU Allreduce test)

   [----------] started processing osu_allreduce_test_8 (OSU Allreduce test)
   [ RUN      ] osu_allreduce_test_8 on daint:gpu using gnu
   [      DEP ] osu_allreduce_test_8 on daint:gpu using gnu
   [ RUN      ] osu_allreduce_test_8 on daint:gpu using intel
   [      DEP ] osu_allreduce_test_8 on daint:gpu using intel
   [ RUN      ] osu_allreduce_test_8 on daint:gpu using pgi
   [      DEP ] osu_allreduce_test_8 on daint:gpu using pgi
   [----------] finished processing osu_allreduce_test_8 (OSU Allreduce test)

   [----------] started processing osu_allreduce_test_4 (OSU Allreduce test)
   [ RUN      ] osu_allreduce_test_4 on daint:gpu using gnu
   [      DEP ] osu_allreduce_test_4 on daint:gpu using gnu
   [ RUN      ] osu_allreduce_test_4 on daint:gpu using intel
   [      DEP ] osu_allreduce_test_4 on daint:gpu using intel
   [ RUN      ] osu_allreduce_test_4 on daint:gpu using pgi
   [      DEP ] osu_allreduce_test_4 on daint:gpu using pgi
   [----------] finished processing osu_allreduce_test_4 (OSU Allreduce test)

   [----------] started processing osu_allreduce_test_2 (OSU Allreduce test)
   [ RUN      ] osu_allreduce_test_2 on daint:gpu using gnu
   [      DEP ] osu_allreduce_test_2 on daint:gpu using gnu
   [ RUN      ] osu_allreduce_test_2 on daint:gpu using intel
   [      DEP ] osu_allreduce_test_2 on daint:gpu using intel
   [ RUN      ] osu_allreduce_test_2 on daint:gpu using pgi
   [      DEP ] osu_allreduce_test_2 on daint:gpu using pgi
   [----------] finished processing osu_allreduce_test_2 (OSU Allreduce test)

   [----------] started processing osu_bandwidth_test (OSU bandwidth test)
   [ RUN      ] osu_bandwidth_test on daint:gpu using gnu
   [      DEP ] osu_bandwidth_test on daint:gpu using gnu
   [ RUN      ] osu_bandwidth_test on daint:gpu using intel
   [      DEP ] osu_bandwidth_test on daint:gpu using intel
   [ RUN      ] osu_bandwidth_test on daint:gpu using pgi
   [      DEP ] osu_bandwidth_test on daint:gpu using pgi
   [----------] finished processing osu_bandwidth_test (OSU bandwidth test)

   [----------] started processing osu_latency_test (OSU latency test)
   [ RUN      ] osu_latency_test on daint:gpu using gnu
   [      DEP ] osu_latency_test on daint:gpu using gnu
   [ RUN      ] osu_latency_test on daint:gpu using intel
   [      DEP ] osu_latency_test on daint:gpu using intel
   [ RUN      ] osu_latency_test on daint:gpu using pgi
   [      DEP ] osu_latency_test on daint:gpu using pgi
   [----------] finished processing osu_latency_test (OSU latency test)

   [----------] waiting for spawned checks to finish
   [       OK ] ( 1/22) fetch_osu_benchmarks~daint on daint:gpu using gnu [compile: 0.009s run: 2.761s total: 2.802s]
   [       OK ] ( 2/22) build_osu_benchmarks~daint:gpu+gnu on daint:gpu using gnu [compile: 25.758s run: 0.056s total: 104.626s]
   [       OK ] ( 3/22) build_osu_benchmarks~daint:gpu+pgi on daint:gpu using pgi [compile: 33.936s run: 70.452s total: 104.473s]
   [       OK ] ( 4/22) build_osu_benchmarks~daint:gpu+intel on daint:gpu using intel [compile: 44.565s run: 65.010s total: 143.664s]
   [       OK ] ( 5/22) osu_allreduce_test_4 on daint:gpu using gnu [compile: 0.011s run: 78.717s total: 101.428s]
   [       OK ] ( 6/22) osu_allreduce_test_2 on daint:gpu using pgi [compile: 0.014s run: 88.060s total: 101.409s]
   [       OK ] ( 7/22) osu_latency_test on daint:gpu using pgi [compile: 0.009s run: 101.325s total: 101.375s]
   [       OK ] ( 8/22) osu_allreduce_test_8 on daint:gpu using pgi [compile: 0.013s run: 76.031s total: 102.005s]
   [       OK ] ( 9/22) osu_allreduce_test_2 on daint:gpu using gnu [compile: 0.011s run: 85.525s total: 101.974s]
   [       OK ] (10/22) osu_allreduce_test_4 on daint:gpu using pgi [compile: 0.011s run: 82.847s total: 102.407s]
   [       OK ] (11/22) osu_allreduce_test_8 on daint:gpu using gnu [compile: 0.010s run: 77.818s total: 106.993s]
   [       OK ] (12/22) osu_latency_test on daint:gpu using gnu [compile: 0.012s run: 103.641s total: 106.858s]
   [       OK ] (13/22) osu_bandwidth_test on daint:gpu using pgi [compile: 0.011s run: 157.129s total: 164.087s]
   [       OK ] (14/22) osu_bandwidth_test on daint:gpu using gnu [compile: 0.010s run: 154.343s total: 164.540s]
   [       OK ] (15/22) osu_allreduce_test_8 on daint:gpu using intel [compile: 0.010s run: 194.643s total: 207.980s]
   [       OK ] (16/22) osu_allreduce_test_2 on daint:gpu using intel [compile: 0.013s run: 201.145s total: 207.983s]
   [       OK ] (17/22) osu_allreduce_test_4 on daint:gpu using intel [compile: 0.016s run: 198.143s total: 208.335s]
   [       OK ] (18/22) osu_latency_test on daint:gpu using intel [compile: 0.010s run: 208.271s total: 208.312s]
   [       OK ] (19/22) osu_allreduce_test_16 on daint:gpu using pgi [compile: 0.013s run: 215.854s total: 248.101s]
   [       OK ] (20/22) osu_allreduce_test_16 on daint:gpu using gnu [compile: 0.010s run: 213.190s total: 248.731s]
   [       OK ] (21/22) osu_allreduce_test_16 on daint:gpu using intel [compile: 0.010s run: 194.339s total: 210.962s]
   [       OK ] (22/22) osu_bandwidth_test on daint:gpu using intel [compile: 0.022s run: 267.171s total: 270.475s]
   [----------] all spawned checks have finished

   [  PASSED  ] Ran 22/22 test case(s) from 10 check(s) (0 failure(s), 0 skipped)
   [==========] Finished on Sun Oct 31 22:07:25 2021
   Run report saved in '/users/user/.reframe/reports/run-report.json'
   Log file(s) saved in '/tmp/rfm-qst7lvou.log'


.. tip::
   A reasonable question that arises is when should I use fixtures and when dependencies?

   The rule of thumb is use fixtures if your test needs to use any resource of the target test and use dependencies if you simply want to impose an order of execution for your tests.
