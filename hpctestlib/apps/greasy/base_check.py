# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from datetime import datetime

import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher


def to_seconds(str):
    return (datetime.strptime(str, '%H:%M:%S') -
            datetime.strptime('00:00:00', '%H:%M:%S')).total_seconds()


class GREASY_BaseCheck(rfm.RegressionTest, pin_prefix = True):
    '''Base class for the GREASY Test.

    CSCS provides the GREASY meta scheduler to manage high
    throughput simulations on Piz Daint: GREASY was developed
    by the Barcelona SuperComputing Center to simplify the
    execution of embarrassingly parallel simulations in any
    environment.

    It is primarily designed to run serial applications, but
    there is a custom version on Piz Daint that is able to handle
    serial, MPI, MPI+OpenMP and OpenMP only applications. Please,
    note that this is a modified version of GREASY and contains
    features that are not present in the original. Although every
    functionality of the original version exists as well in this one,
    there are differences in the way the greasy command is run. For
    instance, on Piz Daint, there is an additional environment variable
    (GREASY_NWORKERS_PER_NODE) to control the number of workers per node,
    which also controls the number of MPI ranks per worker
    (see user.cscs.ch/tools/high_throughput/).

    This test checks GREASY for serial, MPI, MPI+OpenMP and OpenMP.
    It checks if the correct number of tasks is specified for each
    programming interface and if the result is correct (the 'Hello World!'
    message is printed). The default assumption is that GREASY is already
    installed on the device under test.
    '''

    sourcepath = 'tasks_mpi_openmp.c'
    executable = 'tasks_mpi_openmp.x'
    tasks_file = 'tasks.txt'
    executable_opts = [tasks_file]
    greasy_logfile = 'greasy.log'

    @run_after('init')
    def set_keep_files(self):
        self.keep_files = [self.tasks_file, self.greasy_logfile]


    @run_after('init')
    def set_sanity_pattern(self):
        self.sanity_patterns = self.eval_sanity()

    @run_before('setup')
    def set_perf_variables(self):
        """On SLURM there is no need to set OMP_NUM_THREADS if one defines
        num_cpus_per_task, but adding for completeness and portability"""
        self.variables = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'GREASY_NWORKERS_PER_NODE': str(self.nworkes_per_node),
            'GREASY_LOGFILE': self.greasy_logfile
        }

    @run_before('run')
    def generate_tasks_file(self):
        with open(os.path.join(self.stagedir, self.tasks_file), 'w') as fp:
            for i in range(self.num_greasy_tasks):
                fp.write(f'./{self.executable} output-{i}\n')

    @run_before('run')
    def change_executable_name(self):
        """After compiling the code we can change the executable to be
        the greasy one"""

        self.executable = 'greasy'

    @run_before('run')
    def set_launcher(self):
        """The job launcher has to be changed to local since greasy
        make calls to srun"""

        self.job.launcher = getlauncher('local')()

    @deferrable
    def eval_sanity(self):
        """Function for sanity pattern definition."""

        output_files = []
        output_files = [file for file in os.listdir(self.stagedir)
                        if file.startswith('output-')]
        num_greasy_tasks = len(output_files)
        failure_msg = (f'Requested {self.num_greasy_tasks} task(s), but '
                       f'executed only {num_greasy_tasks} tasks(s)')
        sn.evaluate(sn.assert_eq(num_greasy_tasks, self.num_greasy_tasks,
                                 msg=failure_msg))
        num_tasks = sn.getattr(self, 'nranks_per_worker')
        num_cpus_per_task = sn.getattr(self, 'num_cpus_per_task')

        def tid(match):
            return int(match.group(1))

        def num_threads(match):
            return int(match.group(2))

        def rank(match):
            return int(match.group(3))

        def num_ranks(match):
            return int(match.group(4))

        for output_file in output_files:
            result = sn.findall(r'Hello, World from thread \s*(\d+) out '
                                r'of \s*(\d+) from process \s*(\d+) out of '
                                r'\s*(\d+)', output_file)

            failure_msg = (f'Found {sn.count(result)} Hello, World... '
                           f'pattern(s) but expected '
                           f'{num_tasks * num_cpus_per_task} pattern(s) '
                           f'inside the output file {output_file}')
            sn.evaluate(sn.assert_eq(sn.count(result),
                                     num_tasks * num_cpus_per_task,
                                     msg=failure_msg))

            sn.evaluate(sn.all(
                sn.chain(
                    sn.map(
                        lambda x: sn.assert_lt(
                            tid(x), num_threads(x),
                            msg=(f'Found {tid(x)} threads rather than '
                                 f'{num_threads(x)}')
                        ), result
                    ),
                    sn.map(
                        lambda x: sn.assert_lt(
                            rank(x), num_ranks(x),
                            msg=(f'Rank id {rank(x)} is not lower than the '
                                 f'number of ranks {self.nranks_per_worker} '
                                 f'in output file')
                        ), result
                    ),
                    sn.map(
                        lambda x: sn.assert_lt(
                            tid(x), self.num_cpus_per_task,
                            msg=(f'Rank id {tid(x)} is not lower than the '
                                 f'number of cpus per task '
                                 f'{self.num_cpus_per_task} in output '
                                 f'file {output_file}')
                        ), result
                    ),
                    sn.map(
                        lambda x: sn.assert_eq(
                            num_threads(x), num_cpus_per_task,
                            msg=(f'Found {num_threads(x)} threads rather than '
                                 f'{self.num_cpus_per_task} in output file '
                                 f'{output_file}')
                        ), result
                    ),
                    sn.map(
                        lambda x: sn.assert_lt(
                            rank(x), num_tasks,
                            msg=(f'Found {rank(x)} threads rather than '
                                 f'{self.num_cpus_per_task} in output file '
                                 f'{output_file}')
                        ), result
                    ),
                    sn.map(
                        lambda x: sn.assert_eq(
                            num_ranks(x), num_tasks,
                            msg=(f'Number of ranks {num_ranks(x)} is not '
                                 f'equal to {self.nranks_per_worker} in '
                                 f'output file {output_file}')
                        ), result
                    )
                )
            ))
        sn.evaluate(sn.assert_found(r'Finished greasing', self.greasy_logfile))
        sn.evaluate(sn.assert_found(
            (f'INFO: Summary of {self.num_greasy_tasks} '
             f'tasks: '
             f'{self.num_greasy_tasks} OK, '
             f'0 FAILED, '
             f'0 CANCELLED, '
             fr'0 INVALID\.'), self.greasy_logfile
        ))

        return True

    @performance_function('s', perf_key='time')
    def set_perf_patterns(self):
        return sn.extractsingle(r'Total time: (?P<perf>\S+)',
                                self.greasy_logfile,
                                'perf', to_seconds)
