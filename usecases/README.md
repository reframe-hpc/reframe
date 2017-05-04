# ReFrame at CSCS

The ReFrame framework has been in production at [CSCS](http://www.cscs.ch) since the upgrade of the [Piz Daint](http://www.cscs.ch/computers/piz_daint/index.html) system in early December 2016.
There are two large sets of regression tests:
* production tests and
* maintenance tests.

Tags (see [Check Tagging](/writing_checks/#check-tagging)) are used to mark these categories and a regression test may belong to both of them.
Production tests are run daily to monitor the sanity of the system and its performance.
All performance tests log their performance values. The performance over time of certain applications are monitored graphically using [Grafana](https://grafana.com/).

The set of production regression tests comprises 104 individual tests.
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

The set of maintenance regression tests is much more limited, since we want to decrease the downtime of the system.
The regression suite runs at the beginning of the maintenance session and just before returning the machine to the users, so that we can ensure that the user experience is at least at the level before the system was taken down.
The maintenance set of tests comprises application performance tests, some GPU library performance checks, Slurm checks and some POSIX filesystem checks.
The total runtime of the maintenance regression suite for all the partitions is approximately 20 minutes.

We are now porting the regression suite to the [MeteoSwiss](http://www.meteosvizzera.admin.ch/home.html?tab=overview) production system [Piz Kesch](http://www.cscs.ch/computers/kesch_escha_meteoswiss/index.html).
Configuring this system for ReFrame was trivial: we have just added a new system entry in the framework's configuration file describing the different partitions and redefined the  `PrgEnv-gnu` environment to use different compiler wrappers.
Porting the regression tests of interest is also a straightforward process.
In most of the cases, adding just the corresponding system partitions to the `valid_systems` variables and adjusting accordingly the `valid_prog_environs` is enough.

