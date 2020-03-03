# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import itertools
import os
import fnmatch

from datetime import datetime

import reframe as rfm
import reframe.utility.sanity as sn

from reframe.core.launchers.registry import getlauncher


def toSeconds(str):
    return (datetime.strptime(str, '%H:%M:%S') -
            datetime.strptime('00:00:00', '%H:%M:%S')).total_seconds()


@rfm.required_version('>=2.19')
@rfm.parameterized_test(
    ['serial',     'daint:gpu', 24, 12, 1, 1],
    ['serial',     'daint:mc',  72, 36, 1, 1],
    ['openmp',     'daint:gpu', 24,  3, 1, 4],
    ['openmp',     'daint:mc',  72,  9, 1, 4],
    ['mpi',        'daint:gpu', 24,  4, 3, 1],
    ['mpi',        'daint:mc',  72, 12, 3, 1],
    ['mpi+openmp', 'daint:gpu', 24,  3, 2, 2],
    ['mpi+openmp', 'daint:mc',  72,  6, 3, 2])
class GREASYCheck(rfm.RegressionTest):
    def __init__(self, variant, system, num_greasy_tasks, nworkes_per_node,
                 nranks_per_worker, ncpus_per_worker):

        self.valid_systems = [system]
        if system.startswith('daint'):
            self.valid_systems += [system.replace('daint', 'dom')]

        self.valid_prog_environs = ['PrgEnv-gnu']
        self.sourcepath = 'tasks_mpi_openmp.c'
        self.build_system = 'SingleSource'

        # sleep enough time to distinguish if the files are running in parallel
        # or not
        self.sleep_time = 60
        self.build_system.cflags = ['-DSLEEP_TIME=%d' % self.sleep_time]

        if variant in ['openmp']:
            self.build_system.cflags += ['-fopenmp']
        elif variant in ['mpi']:
            self.build_system.cflags += ['-D_MPI']
        elif variant in ['mpi+openmp']:
            self.build_system.cflags += ['-fopenmp', '-D_MPI']

        self.executable = 'tasks_mpi_openmp.x'
        self.tasks_file = 'tasks.txt'
        self.executable_opts = [self.tasks_file]
        self.greasy_logfile = 'greasy.log'
        self.keep_files = [self.tasks_file, self.greasy_logfile]

        self.sanity_patterns = self.eval_sanity()

        nnodes = 2
        self.num_greasy_tasks = num_greasy_tasks
        self.nworkes_per_node = nworkes_per_node
        self.nranks_per_worker = nranks_per_worker
        self.num_tasks_per_node = nranks_per_worker * nworkes_per_node
        self.num_tasks = self.num_tasks_per_node * nnodes
        self.num_cpus_per_task = ncpus_per_worker

        # Reference value is system agnostic and depnes
        refperf = self.sleep_time * num_greasy_tasks / nworkes_per_node / nnodes
        self.reference = {
            '*': {
                'time': (refperf, None, 0.3, 's')
            }
        }
        self.perf_patterns = {
            'time': sn.extractsingle(r'Total time: (?P<perf>\S+)',
                                     self.greasy_logfile,
                                     'perf', toSeconds)
        }

        # On SLURM there is no need to set OMP_NUM_THREADS if one defines
        # num_cpus_per_task, but adding for completeness and portability
        self.variables = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'GREASY_NWORKERS_PER_NODE': str(nworkes_per_node),
            'GREASY_LOGFILE': self.greasy_logfile
        }

        self.modules = ['GREASY']
        self.maintainers = ['VH', 'SK']
        self.use_multithreading = False

        self.tags = {'production'}

    @rfm.run_before('run')
    def generate_tasks_file(self):
        with open(os.path.join(self.stagedir, self.tasks_file), 'w') as outfile:
            for i in range(self.num_greasy_tasks):
                outfile.write("./%s output-%d\n" % (self.executable, i))

    @rfm.run_before('run')
    def daint_dom_gpu_specific_workaround(self):
        if self.current_partition.fullname in ['daint:gpu', 'dom:gpu']:
            self.variables['CRAY_CUDA_MPS'] = "1"
            self.variables['CUDA_VISIBLE_DEVICES'] = "0"
            self.variables['GPU_DEVICE_ORDINAL'] = "0"

            self.extra_resources = {
                'gres': {
                    'gres': 'gpu:0,craynetwork:4'
                }
            }

    @rfm.run_before('run')
    def change_executable_name(self):
        # After compiling the code we can change the executable to be
        # the greasy one
        self.executable = 'greasy'

    @rfm.run_before('run')
    def set_launcher(self):
        # The job launcher has to be changed to local since greasy
        # make calls to srun
        self.job.launcher = getlauncher('local')()

    @sn.sanity_function
    def eval_sanity(self):
        output_files = []
        for file in os.listdir(self.stagedir):
            if file.startswith("output-"):
                output_files.append(file)

        num_greasy_tasks = len(output_files)
        failure_msg = ('Requested %s task(s), but only executed %s tasks(s)' %
                       (self.num_greasy_tasks, num_greasy_tasks))
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

            failure_msg = ('Found %s Hello, World... pattern(s), but expected '
                           '%s pattern(s) inside the output file %s' % (
                                              sn.count(result),
                                              num_tasks * num_cpus_per_task,
                                              output_file))
            sn.evaluate(sn.assert_eq(sn.count(result),
                                     num_tasks * num_cpus_per_task,
                                     msg=failure_msg))

            sn.evaluate(sn.assert_true(sn.all(
                sn.chain(
                    sn.map(lambda x: sn.assert_lt(tid(x), num_threads(x)),
                            result),
                    sn.map(lambda x: sn.assert_lt(rank(x), num_ranks(x)),
                            result),
                    sn.map(
                        lambda x: sn.assert_lt(tid(x), num_cpus_per_task),
                        result),
                    sn.map(
                        lambda x: sn.assert_eq(num_threads(x),
                                                num_cpus_per_task),
                        result),
                    sn.map(lambda x: sn.assert_lt(rank(x), num_tasks),
                            result),
                    sn.map(lambda x: sn.assert_eq(num_ranks(x), num_tasks),
                            result),
                )
            )))

        sn.evaluate(sn.assert_eq(sn.count(sn.findall(r'Finished greasing',
                                         self.greasy_logfile)), 1))

        result = sn.findall(r'INFO: Summary of (\d+) tasks: '
                            r'(\d+) OK, '
                            r'(\d+) FAILED, '
                            r'(\d+) CANCELLED, '
                            r'(\d+) INVALID\.', output_file)
        sn.evaluate(sn.assert_true(sn.all(
                    sn.chain(
                        sn.map(lambda x: sn.assert_eq(int(x.group(1)),
                                                      self.num_greasy_tasks),
                               result),
                        sn.map(lambda x: sn.assert_eq(int(x.group(2)),
                                                      self.num_greasy_tasks),
                               result),
                        sn.map(lambda x: sn.assert_eq(int(x.group(3)), 0),
                               result),
                        sn.map(lambda x: sn.assert_eq(int(x.group(4)), 0),
                               result),
                        sn.map(lambda x: sn.assert_eq(int(x.group(5)), 0),
                               result),
                    )
                )))

        return True
