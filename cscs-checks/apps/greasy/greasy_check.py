# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
from hpctestlib.apps.greasy.base_check import GREASY_BaseCheck

@rfm.simple_test
class GREASYCheckCSCS(GREASY_BaseCheck):
    modules = ['GREASY']
    maintainers = ['VH', 'SK']
    tags = {'production'}
    valid_prog_environs = ['PrgEnv-gnu']
    build_system = 'SingleSource'
    use_multithreading = False
    nnodes = 2
    mode = parameter(
        [['serial',     'gpu', 24, 12, 1, 1],
         ['serial',     'mc',  72, 36, 1, 1],
         ['openmp',     'gpu', 24,  3, 1, 4],
         ['openmp',     'mc',  72,  9, 1, 4],
         ['mpi',        'gpu', 24,  4, 3, 1],
         ['mpi',        'mc',  72, 12, 3, 1],
         ['mpi+openmp', 'gpu', 24,  3, 2, 2],
         ['mpi+openmp', 'mc',  72,  6, 3, 2],
         ])

    @run_after('init')
    def parameters_unpacking(self):
        (self.variant, self.partition, self.num_greasy_tasks,
         self.nworkes_per_node, self.nranks_per_worker,
         self.ncpus_per_worker) = self.mode

    @run_after('init')
    def add_valid_systems(self):
        self.valid_systems = ['daint:' + self.partition,
                              'dom:' + self.partition]

    @run_after('init')
    def set_num_tasks(self):
        self.num_tasks_per_node = (self.nranks_per_worker *
                                   self.nworkes_per_node)
        self.num_tasks = self.num_tasks_per_node * self.nnodes
        self.num_cpus_per_task = self.ncpus_per_worker

    @run_before('setup')
    def add_cflags(self):
        # sleep enough time to distinguish if the files are running in parallel
        # or not
        self.sleep_time = 60
        self.build_system.cflags = [f'-DSLEEP_TIME={self.sleep_time:d}']
        if self.variant == 'openmp':
            self.build_system.cflags += ['-fopenmp']
        elif self.variant == 'mpi':
            self.build_system.cflags += ['-D_MPI']
        elif self.variant == 'mpi+openmp':
            self.build_system.cflags += ['-fopenmp', '-D_MPI']

    @run_before('run')
    def daint_dom_gpu_specific_workaround(self):
        if self.current_partition.fullname in ['daint:gpu', 'dom:gpu']:
            self.variables['CRAY_CUDA_MPS'] = '1'
            self.variables['CUDA_VISIBLE_DEVICES'] = '0'
            self.variables['GPU_DEVICE_ORDINAL'] = '0'
            self.extra_resources = {
                'gres': {
                    'gres': 'gpu:0,craynetwork:4'
                }
            }
        elif self.current_partition.fullname in ['dom:mc']:
            self.extra_resources = {
                'gres': {
                    'gres': 'craynetwork:72'
                }
            }
        elif self.current_partition.fullname in ['daint:mc']:
            if self.variant != 'serial':
                self.extra_resources = {
                    'gres': {
                        'gres': 'craynetwork:72'
                    }
                }

    @run_before('performance')
    def set_reference(self):
        # Reference value is system agnostic
        # Adding 10 secs of slowdown per greasy tasks
        # this is to compensate for whenever the systems are full and srun gets
        # slightly slower
        refperf = (
            (self.sleep_time+10)*self.num_greasy_tasks /
            self.nworkes_per_node / self.nnodes
        )
        self.reference = {
            '*': {
                'time': (refperf, None, 0.5, 's')
            }
        }
