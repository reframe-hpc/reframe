# ReFrame Development and Usage at CSCS

The ReFrame framework has been in production at [CSCS](http://www.cscs.ch) since the upgrade of the [Piz Daint](http://www.cscs.ch/computers/piz_daint/index.html) system in early December 2016.
There are two large sets of regression tests:
* production tests and
* maintenance tests.

[Tags](/writing_checks/#check-tagging) are used to mark these categories and a regression test may belong to both of them.
Production tests are run daily to monitor the sanity of the system and its performance.
All performance tests log their performance values. The performance over time of certain applications are monitored graphically using [Grafana](https://grafana.com/).

The total set of regression tests comprises 122 individual tests, from which 104 are marked as production.
Some of them are eligible to run on both the multicore and hybrid partitions, whereas others are meant to run only on the login nodes.
Depending on the test, multiple programming environments might be tried.
In total, 437 test cases are run from 157 regression tests on all the system partitions.
The following Table summarizes the production regression tests.

Type        | Partition | # Tests | # Test cases | Total
------------|-----------|---------|--------------|------
Production  | Login     | 15      | 24           |
Production  | Multicore | 61      | 190          |
Production  | Hybrid    | 81      | 223          | 437
Maintenance | Login     | 2       | 2            |
Maintenance | Multicore | 7       | 7            |
Maintenance | Hybrid    | 19      | 19           | 28

The set of maintenance regression tests is much more limited to decrease the downtime of the system.
The regression suite runs at the beginning of the maintenance session and just before returning the machine to the users, so that we can ensure that the user experience is at least at the level before the system was taken down.
The maintenance set of tests comprises application performance tests, some GPU library performance checks, Slurm checks and some POSIX filesystem checks.
The total runtime of the maintenance regression suite for all the partitions is approximately 20 minutes.

The porting of the regression suite to the [MeteoSwiss](http://www.meteosvizzera.admin.ch/home.html?tab=overview) production system [Piz Kesch](http://www.cscs.ch/computers/kesch_escha_meteoswiss/index.html), using ReFrame was almost trivial. The new system entry was added in the framework's configuration file describing the different partitions together with a new redefined `PrgEnv-gnu` environment to use different compiler wrappers.
Porting the regression tests of interest was also a straightforward process.
In most of the cases, adding just the corresponding system partitions to the `valid_systems` variables and adjusting accordingly the `valid_prog_environs` was enough.


The ReFrame framework really focuses on abstracting away all the gory details from the regression test description, hence letting the user to concentrate solely on the logic of his test. A bit of this effect can be seen in the following Table where the total amount of lines of code (loc) of different components of the previous regression framework used at CSCS and ReFrame is summarized.

Component                 | Previous Framework | ReFrame
--------------------------|--------------------|---------
Core                      | N/A                | 3660 loc
Front-end                 | 1038 loc           | 958 loc
Regression tests          | 14635 loc          | 2985 loc
Avg. regression file size | 179 loc            | 93 loc
Avg. regression test size | 179 loc            | 25 loc

The benefits observed in the maintenance and development of new regression tests with ReFrame have exceed the initial costs related to the development of the framework. In the previous regression there was no core components to be maintained or developed. And the maintenance burden associated to the core operations were transmitted to the regression tests, which increased the number of loc to be implemented per regression test. This number in the previous framework was about 15K, where the number is around 3K in ReFrame (a dramatic 80% decrease in loc).

There are 32 files implementing the regression tests written with ReFrame, which gives an average of 93 loc per regression test file. These 32 files genereate a total of 122 tests, which translates to an effective number of 25 loc per test.

Separating the logical description of a regression test from all the unnecessary implementation contributes significantly in the ease of writing and maintaining new regression tests.
