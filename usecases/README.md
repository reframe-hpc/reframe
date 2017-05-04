# Use Cases

The ReFrame framework has been put into production with the upgrade of the Piz Daint system in early December 2016.
We have two large sets of regression tests:
\begin{inparaenum}[(a)]
\item production tests and
\item maintenance tests.
\end{inparaenum}
We use tags (see Section~\ref{sec:check-tags}) to mark these categories and a regression test may belong to both of them.
Production tests are run daily to monitor the sanity of the system and its performance.
All performance tests log their performance values and we use Grafana~\cite{grafana} to graphically monitor the performance of certain applications and benchmarks over time.

The set of production regression tests comprises 104 individual tests.
Some of them are eligible to run on both the multicore and hybrid partitions, whereas others are meant to run only on the login nodes.
Depending on the test, multiple programming environments might be tried.
In total, we run 437 test cases from 157 regression tests on all the system partitions.
Table~\ref{tab:regression-suite-daint} summarizes the production regression tests.
