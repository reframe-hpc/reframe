# Use Cases

## ReFrame Usage at CSCS

The ReFrame framework has been in production at [CSCS](http://www.cscs.ch) since December 2016.
We use it to test not only [Piz Daint](http://www.cscs.ch/computers/piz_daint/index.html), but almost all our systems that we provide to users.

We have two large sets of regression tests:

* production tests and
* maintenance tests.

Tags are used to mark these categories and a regression test may belong to both of them.
Production tests are run daily to monitor the sanity of the system and its performance.
All performance tests log their performance values.
The performance over time of certain applications are monitored graphically using [Grafana](https://grafana.com/).

The total set of our regression tests comprises 172 individual tests, from which 153 are marked as production tests.
Some of them are eligible to run on both the multicore and hybrid partitions of the system, whereas others are meant to run only on the login nodes.
Depending on the test, multiple programming environments might be tried.
In total, 448 test cases are run from the 153 regression tests on all the system partitions.
The following Table summarizes the production regression tests.

The set of maintenance regression tests is much more limited to decrease the downtime of the system.
The regression suite runs at the beginning of the maintenance session and just before returning the machine to the users, so that we can ensure that the user experience is at least at the level before the system was taken down.
The maintenance set of tests comprises application performance tests, some GPU library performance checks, Slurm checks and some POSIX filesystem checks.

The porting of the regression suite to the [MeteoSwiss](http://www.meteosvizzera.admin.ch/home.html?tab=overview) production system [Piz Kesch](http://www.cscs.ch/computers/kesch_escha_meteoswiss/index.html), using ReFrame was almost trivial.
The new system entry was added in the framework's configuration file describing the different partitions together with a new redefined `PrgEnv-gnu` environment to use different compiler wrappers.
Porting the regression tests of interest was also a straightforward process.
In most of the cases, adding just the corresponding system partitions to the `valid_systems` variables and adjusting accordingly the `valid_prog_environs` was enough.

ReFrame really focuses on abstracting away all the gory details from the regression test description, hence letting the user to concentrate solely on the logic of his test.
A bit of this effect can be seen in the following Table where the total amount of lines of code (loc) of the regression tests written in the previous shell script-based solution and ReFrame is shown.
We also present a snapshot of the first public release of ReFrame ([v2.2](https://github.com/eth-cscs/reframe/releases/tag/v2.2)).


Maintenance Burden        | Shell-Script Based | ReFrame (v2.2) | ReFrame (v2.7) |
--------------------------|--------------------|----------------|----------------|
Total tests               |   179              |  122           |  172           |
Total size of tests       | 14635 loc          | 2985 loc       | 4493 loc       |
Avg. test file size       |   179 loc          |   93 loc       |   87 loc       |
Avg. effective test size  |   179 loc          |   25 loc       |   25 loc       |

The difference in the total amount of regression test code is dramatic.
From the 15K lines of code of the old shell script based regression testing suite, ReFrame tests use only 3K lines of code (first release) achieving a higher coverage.

> NOTE: The higher test count of the older suite refers to test cases, i.e., running the same test for different programming environments, whereas for ReFrame the counts does not account for this.

Each regression test file in ReFrame is 80&ndash;90 loc on average.
However, eash regression test file may contain or generate more than one related tests, thus leading to the effective decrease of the line count per test to only 25 loc.

Separating the logical description of a regression test from all the unnecessary implementation details contributes significantly in the ease of writing and maintaining new regression tests with ReFrame.
