=========
Use Cases
=========

ReFrame has been publicly released on May 2017, but it has been used in production at the Swiss National Supercomputing Centre since December 2016.
Since then it has gained visibility across computing centers, some of which have already integrated in their production testing workflows and others are considering to fully adopt it.
To our knowledge, private companies in the HPC sector are using it as well.
Here we will briefly present the use cases of ReFrame at the Swiss National Supercomputing Centre (`CSCS <https://www.cscs.ch/>`__) in Switzerland, at the National Energy Research Scientific Computing Center (`NERSC <https://www.nersc.gov/>`__) and at the Ohio Supercomputer Center (`OSC <https://www.osc.edu/>`__) in the United States.


ReFrame at CSCS
---------------

CSCS uses ReFrame for both functionality and performance tests for all of its production and test development systems, among which are the `Piz Daint <https://www.cscs.ch/computers/piz-daint/>`__ supercomputer (Cray XC40/XC50 hybrid system), the `Kesch/Escha <https://www.cscs.ch/computers/kesch-escha-meteoswiss/>`__ twin systems (Cray CS-Storm used by MeteoSwiss for weather predition).
The same ReFrame tests are reused as much as possible across systems with minor adaptations.
The test suite of CSCS (publicly `available <https://github.com/eth-cscs/reframe/tree/master/cscs-checks>`__ inside ReFrame's repository) comprises tests for full scientific applications, scientific libraries, programming environments, compilation and linking, profiling and debugger tools, basic CUDA operations, performance microbenchmarks and I/O libraries.
Using tags we have split the tests in three broad overlapping categories:

1. Production tests -- This category comprises a large variety of tests and is run daily overnight using Jenkins.
2. Maintenance tests -- This suite is essentially a small subset of the production tests, comprising mostly application sanity and performance tests, as well as sanity tests for the programming environment and the scheduler.
   It is run before and after maintenance of the systems.
3. Benchmarking tests -- These tests are used to measure the performance of different computing and networking components and are run manually before major upgrades or when a performance problem needs to be investigated.

We are currently working on a fourth category of tests that are intended to run frequently (e.g., every 10 minutes).
The purpose of these tests is to measure the system behavior and performance as perceived by the users.
Example tests are the time it takes to run basic Slurm commands and/or performance basic filesystem operations.
Such glitches might affect the performance of running applications and cause users to open support tickets.
Collecting periodically such performance data will help us correlate system events with user application performance.
Finally, there is an ongoing effort to expand our ReFrame test suite to virtual clusters based on OpenStack.
The new tests will measure the responsiveness of our OpenStack installation to deploy compute instances, volumes, and perform snapshots.
We plan to make them publicly available in the near future.

Our regression test suite consists of 278 tests in total, from which 204 are marked as production tests.
A test can be valid for one or more systems and system partitions and can be tried with multiple programming environments.
Specifically on Piz Daint, the production suite runs 640 test cases from 193 tests.

ReFrame really focuses on abstracting away all the gory details from the regression test description, hence letting the user to concentrate solely on the logic of his test.
This effect can be seen in the following Table where the total amount of lines of code (loc) of the regression tests written with the previous shell script-based solution is shown in comparison to ReFrame.

============================= ====================== ====================== =======================
    Maintenance Burden           Shell-Script Based     ReFrame (May 2017)     ReFrame (Apr. 2020)
============================= ====================== ====================== =======================
    Total tests                  179                    122                    278
    Total size of tests          14635 loc              2985 loc               8421 loc
    Avg. test file size          179 loc                93 loc                 102 loc
    Avg. effective test size     179 loc                25 loc                 30 loc
============================= ====================== ====================== =======================

The difference in the total amount of regression test code is dramatic.
From the 15K lines of code of the old shell script based regression testing suite, ReFrame tests used only 3K lines of code (first public release, May 2017) achieving a higher coverage.

Each regression test file in ReFrame is approximately 100 loc on average.
However, each regression test file may contain or generate more than one related tests, thus leading to the effective decrease of the line count per test to only 30 loc.
If we also account for the test cases generated per test, this number decreases further.

Separating the logical description of a regression test from all the unnecessary implementation details contributes significantly to the ease of writing and maintaining new regression tests with ReFrame.

.. note:: The higher test count of the older suite refers to test cases, i.e., running the same test for different programming environments, whereas for ReFrame the counts do not account for this.

.. note:: CSCS maintains a separate repository for tests related to HPC debugging and performance tools, which you can find `here <https://github.com/eth-cscs/hpctools>`__. These tests were not accounted in this analysis.


ReFrame at NERSC
----------------

ReFrame at `NERSC <https://www.nersc.gov/>`__ covers functionality and performance of its current HPC system `Cori <https://www.nersc.gov/systems/cori/>`__, a Cray XC40 with Intel "Haswell" and "Knights Landing" compute nodes; as well as its smaller Cray CS-Storm cluster featuring Intel "Skylake" CPUs and NVIDIA "Volta" GPUs.
The performance tests include several general-purpose benchmarks designed to stress different components of the system, including `HPGMG <https://hpgmg.org/>`__ (both finite-element and finite-volume tests), `HPCG <https://www.hpcg-benchmark.org/>`__, `Graph500 <https://graph500.org/>`__, `IOR <https://ior.readthedocs.io/en/latest/>`__, and others.
Additionally, the tests include several benchmark codes used during NERSC system procurements, as well as several extracted benchmarks from full applications which participate in the NERSC Exascale Science Application Program (`NESAP <https://www.nersc.gov/research-and-development/nesap/>`__).
Including NESAP applications ensures that representative components of the NERSC workload are included in the performance tests.

The functionality tests evaluate several different components of the system; for example, there are several tests for the Cray `DataWarp <https://www.cray.com/products/storage/datawarp>`__ software which enables users to interact with the Cori burst buffer.
There are also several Slurm tests which verify that partitions and QoSs are correctly configured for jobs of varying sizes.
The Cray programming environments, including compiler wrappers, MPI and OpenMP capability, and Shifter, are also included in these tests, and are especially impactful following changes in defaults to the programming environments.

The test battery at NERSC can be invoked both manually and automatically, depending on the need.
Specifically, the full battery is typically executed manually following a significant change to the Cori system, e.g., after a major system software change, or a Cray Linux OS upgrade, before the system is released back to users.
Under most other circumstances, however, only a subset of tests are typically run, and in most causes they are executed automatically.
NERSC uses ReFrame's tagging capabilities to categorize the various subsets of tests, such that groups of tests which evaluate a particular component of the system can be invoked easily.
For example, some performance tests are tagged as "daily", others as "weekly", "reboot", "slurm", "aries", etc., such that it is clear from the test's Python code when and how frequently a particular test is run.

ReFrame has also been integrated into NERSC's centralized data collection service used for facility and system monitoring, called the "Data Collect."
The Data Collect stores data in an Elasticsearch instance, uses `Logstash <https://www.elastic.co/logstash>`__ to ingest log information about the Cori system, and provides a web-based GUI to display results via `Kibana <https://www.elastic.co/kibana>`__.
Cray, in turn, provides the `Cray Lightweight Log Manager <https://pubs.cray.com/content/S-2393/CLE%206.0.UP05/xctm-series-system-administration-guide/cray-lightweight-log-management-llm-system>`__ on XC systems such as Cori, which provides a syslog interface.
ReFrame's support for Syslog, and the Python standard `logging <https://docs.python.org/3.8/library/logging.html>`__ library, enabled simple integration with NERSC's Data Collect
The result of this integration with ReFrame to the Data Collect is that the results from each ReFrame test executed on Cori are visible via a Kibana query within a few seconds of the test completing.
One can then configure Elasticsearch to alert a system administrator if a particular system functionality stops working, or if the performance of certain benchmarks suddenly declines.

Finally, ReFrame has been automated at NERSC via the continuous integration (CI) capabilities provided by an internal GitLab instance.
More specifically, GitLab was enhanced due to efforts from the US Department of Energy `Exascale Computing Project (ECP) <https://www.exascaleproject.org/>`__ in order to allow CI "runners" to submit jobs to queues on HPC systems such as Cori automatically via schedulable "pipelines."
Automation via GitLab runners is a significant improvement over test executed automated by cron, because the runners exist outside of the Cori system, and therefore are unaffected by system shutdowns, reboots, and other disruptions.
The pipelines are configured to run tests with particular tags at particular times, e.g., tests tagged with "daily" are invoked each day at the same time, tests tagged "weekly" are invoked once per week, etc.


ReFrame at OSC
--------------

At OSC, we use ReFrame to build the testing system for the software environment.
As a change is made to an application, e.g., upgrade, module change or new installation, ReFrame tests are performed by a user-privilege account and the OSC staff members who receive the test summary can easily check the result to decide if the change should be approved.

ReFrame is configured and installed on three production systems (`Pitzer <https://www.osc.edu/resources/technical_support/supercomputers/pitzer>`__, `Owens <https://www.osc.edu/resources/technical_support/supercomputers/owens>`__ and `Ruby <https://www.osc.edu/resources/technical_support/supercomputers/ruby>`__).
For each application we prepare the following classes of ReFrame tests:

1. default version -- checks if a new installation overwrites the default module file
2. broken executable or library -- i.e. run a binary with the ``--version`` flag and compare the result with the module version,
3. functionality -- i.e. numerical tests,
4. performance -- extensive functionality checking and benchmarking,

where we currently have functionality and performance tests for a limited subset of our deployed software.

All checks are designed to be general and version independent.
The correct module file is loaded at runtime, reducing the number of Python classes to be maintained.
In addition, all application-based ReFrame tests are performed as regression testing of software environment when the system has critical update or rolling reboot.

ReFrame is also used for performance monitoring.
We run weekly MPI tests and monthly HPCG tests. The performance data is logged directly to an internal `Splunk <https://www.splunk.com/>`__ server via Syslog protocol.
The job summary is sent to the responsible OSC staff member who can watch the performance dashboards.
