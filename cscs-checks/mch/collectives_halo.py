# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


class CollectivesBaseTest(rfm.RegressionTest):
    def __init__(self, variant, bench_reference):
        self.valid_systems = ['dom:gpu', 'daint:gpu', 'kesch:cn', 'tiger:gpu',
                              'arolla:cn', 'tsa:cn']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.variables = {'G2G': '1'}
        self.executable = 'build/src/comm_overlap_benchmark'
        if variant != 'default':
            self.executable_opts = ['--' + variant]

        self.build_system = 'CMake'
        self.build_system.builddir = 'build'
        self.build_system.config_opts = ['-DCMAKE_BUILD_TYPE=Release',
                                         '-DENABLE_MPI_TIMER=ON']

        if self.current_system.name == 'kesch':
            self.exclusive_access = True
            self.num_tasks = 144
            self.num_gpus_per_node = 16
            self.num_tasks_per_node = 16
            self.num_tasks_per_socket = 8
            self.modules = ['cmake']
            self.variables['MV2_USE_CUDA'] = '1'
            self.build_system.config_opts += [
                '-DMPI_VENDOR=mvapich2',
                '-DCUDA_COMPUTE_CAPABILITY="sm_37"'
            ]
            self.build_system.max_concurrency = 1
        elif self.current_system.name in ['arolla', 'tsa']:
            self.exclusive_access = True
            self.num_tasks = 32
            self.num_gpus_per_node = 8
            self.num_tasks_per_node = 16
            self.num_tasks_per_socket = 8
            self.modules = ['cmake']
            self.build_system.config_opts += [
                '-DMPI_VENDOR=openmpi',
                '-DCUDA_COMPUTE_CAPABILITY="sm_70"'
            ]
            self.build_system.max_concurrency = 1
        elif self.current_system.name in {'daint', 'dom', 'tiger'}:
            self.num_tasks = 4
            self.num_gpus_per_node = 1
            self.num_tasks_per_node = 1
            if self.current_system.name in {'tiger'}:
                self.modules = ['craype-accel-nvidia60']
            else:
                self.modules = ['craype-accel-nvidia60', 'CMake']

            self.variables['MPICH_RDMA_ENABLED_CUDA'] = '1'
            self.build_system.config_opts += [
                '-DCUDA_COMPUTE_CAPABILITY="sm_60"'
            ]
            self.build_system.max_concurrency = 8
        else:
            self.num_tasks = 4
            self.num_gpus_per_node = 1
            self.num_tasks_per_node = 1
            self.build_system.max_concurrency = 1

        self.sanity_patterns = sn.assert_found(r'ELAPSED TIME:', self.stdout)
        self.perf_patterns = {
            'elapsed_time': sn.extractsingle(r'ELAPSED TIME:\s+(\S+)',
                                             self.stdout, 1, float, -1)
        }
        ref_values = {
            'kesch': {
                'nocomm':  5.7878,
                'nocomp':  5.62155,
                'default': 5.53777
            },
            'daint': {
                'nocomm':  0.0171947,
                'nocomp':  0.0137893,
                'default': 0.0138493
            }
        }

        if self.current_system.name == 'dom':
            sysname = 'daint'
        else:
            sysname = self.current_system.name

        try:
            ref = bench_reference[sysname][variant]
        except KeyError:
            ref = 0.0

        self.reference = {
            'kesch:cn': {
                'elapsed_time': (ref, None, 0.15, 's')
            },
            'daint': {
                'elapsed_time': (ref, None, 0.15, 's')
            },
            'dom': {
                'elapsed_time': (ref, None, 0.15, 's')
            },
        }

        self.maintainers = ['AJ', 'MKr']
        if self.current_system.name == 'tsa':
            self.tags = {'mch'}
        else:
            self.tags = {'production', 'mch', 'craype'}

    @rfm.run_before('run')
    def set_launcher_options(self):
        if self.current_system.name in ['arolla', 'kesch', 'tsa']:
            self.job.launcher.options = ['--distribution=block:block',
                                         '--cpu_bind=q']


@rfm.parameterized_test(['default'], ['nocomm'], ['nocomp'])
class AlltoallvTest(CollectivesBaseTest):
    def __init__(self, variant):
        super().__init__(variant,
                         {
                             'kesch': {
                                 'nocomm':  6.89819,
                                 'nocomp':  6.98276,
                                 'default': 6.85289
                             },
                             'daint': {
                                 'nocomm':  0.0171947,
                                 'nocomp':  0.0137893,
                                 'default': 0.0138493
                             },
                         })
        self.strict_check = False
        self.sourcesdir = 'https://github.com/eth-cscs/comm_overlap_bench.git'
        self.prebuild_cmds = ['git checkout alltoallv']


@rfm.parameterized_test(['default'], ['nocomm'], ['nocomp'])
class HaloExchangeTest(CollectivesBaseTest):
    def __init__(self, variant):
        super().__init__(variant,
                         {
                             'kesch': {
                                 'nocomm':  5.7878,
                                 'nocomp':  54.2012,
                                 'default': 55.142
                             },
                             'daint': {
                                 'nocomm':  0.978306,
                                 'nocomp':  1.36716,
                                 'default': 2.53509
                             },
                         })
        self.sourcesdir = 'https://github.com/eth-cscs/comm_overlap_bench.git'
        self.prebuild_cmds = ['git checkout barebones']
