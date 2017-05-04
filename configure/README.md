\RegressionName provides an easy and flexible way to configure new systems and new programming environments.
It is shipped by default with the Cray Swan system configured.
As soon as you have configured a new system with its programming environments, adapting an existing regression test could be as easy as just adding the system's name in the `valid_systems` list and its associated programming environments in the `valid_prog_environs` list.
...

# New Systems

From the regression's point of view each system consists of a set of logical partitions.
These partitions need not necessarily correspond to real scheduler partitions.
For example, Daint comprises three logical partitions: the login nodes (named *login*), the hybrid nodes (named *gpu*) and the multicore nodes (named *mc*), but these do not correspond to actual Slurm partitions.
Logical partitions may even use different job schedulers.
An obvious example is the *login* partition that uses the *local* scheduler, since regression tests for login nodes are meant to run locally.
A logical partition may also be associated with a job scheduler option that enables access to it.
For example, on Piz Daint the hybrid and multicore nodes are obtained through Slurm constraints using the `--constraint` option.
On other systems the logical partitions may be mapped 1--1 to real scheduler partitions, in which case the `--partition` option of Slurm would be used.
You can associate also modules and environment variables with logical partitions.
These modules will always be loaded and environment variables will be set before a regression test runs on that partition.
For example, on Piz Daint, you have to load a specific module on each partition, which makes available an optimized software stack for the nodes of the partition.
Finally, a partition is associated with a list of (programming) environments to test, e.g., `PrgEnv-cray`, `PrgEnv-gnu` etc.
These are defined inside a scoped dictionary (see Section~\ref{sec:tag-resolution}) keyed on the system or system partition.
This allows you to define programming environments for a specific system only or override environment definitions.
For example, on one of our systems we needed to override the default definition of `PrgEnv-gnu` to use `mpicc`, `mpicxx` and `mpif90` as the compiler wrappers.
The nice trait with this is that the regression tests supporting `PrgEnv-gnu` do not need to change, even if the compiler wrappers change.

# New Programming Environments