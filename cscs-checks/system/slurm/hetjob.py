# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class HeterogenousSlurmJobTest(rfm.RegressionTest):
    descr = 'Heterogenous Slurm job test'
    sourcepath = 'heterogeneous.c'
    valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc']
    valid_prog_environs = ['PrgEnv-cray']
    num_tasks = 12
    num_tasks_per_node = 3
    num_cpus_per_task = 4
    num_tasks_het = 1
    num_tasks_per_node_het = 1
    num_cpus_per_task_het = 6
    build_system = 'SingleSource'
    maintainers = ['TM', 'VH']
    tags = {'slurm'}

    @run_before('compile')
    def set_cflags(self):
        self.build_system.cflags = ['-fopenmp']

    @run_before('run')
    def set_threads_per_core(self):
        self.job.options = ['--threads-per-core=1']

    @run_before('run')
    def setup_heterogeneous_job(self):
        self.job.options += [
            f'#SBATCH hetjob', f'--ntasks={self.num_tasks_het}',
            f'--ntasks-per-node={self.num_tasks_per_node_het}',
            f'--cpus-per-task={self.num_cpus_per_task}',
            f'--threads-per-core=1',
            # The second constraint has to be passed using the #SBATCH prefix
            # verbatim, so that ReFrame does not combine the constraints
            f'#SBATCH --constraint={self.current_partition.name}'
        ]
        # Ensure that the two heterogeneous jobs share the MPI_COMM_WORLD
        # communicator
        self.job.launcher.options = ['--het-group=0,1']

    @sanity_function
    def validate(self):
        return sn.all([
            *[sn.assert_found(f'Hello from rank {rank} running omp thread '
                              f'{thread}/{self.num_cpus_per_task}',
                              self.stdout)
              for rank in range(self.num_tasks)
              for thread in range(self.num_cpus_per_task)],
            # The mpi ranks of the second job are assigned the remaining ids
            # of the total ranks of the MPI_COMM_WORLD communicator
            *[sn.assert_found(f'Hello from rank {rank + self.num_tasks} '
                              f'running omp thread '
                              f'{thread}/{self.num_cpus_per_task_het}',
                              self.stdout)
              for rank in range(self.num_tasks_het)
              for thread in range(self.num_cpus_per_task_het)],
        ])
