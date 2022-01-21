# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


class CollectivesBaseTest(rfm.RegressionTest):
    variant = parameter(['default', 'nocomm', 'nocomp'])

    def __init__(self, bench_reference):
        self.valid_systems = ['dom:gpu', 'daint:gpu', 'arolla:cn', 'tsa:cn']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.variables = {'G2G': '1'}
        self.executable = 'build/src/comm_overlap_benchmark'
        if self.variant != 'default':
            self.executable_opts = [f'--{self.variant}']

        self.build_system = 'CMake'
        self.build_system.builddir = 'build'
        self.build_system.config_opts = ['-DCMAKE_BUILD_TYPE=Release',
                                         '-DENABLE_MPI_TIMER=ON']

        if self.current_system.name in ['arolla', 'tsa']:
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
        elif self.current_system.name in {'daint', 'dom'}:
            self.num_tasks = 4
            self.num_gpus_per_node = 1
            self.num_tasks_per_node = 1
            self.modules = ['craype-accel-nvidia60', 'CMake', 'cdt-cuda']
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
            ref = bench_reference[sysname][self.variant]
        except KeyError:
            ref = 0.0

        self.reference = {
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

    @run_before('run')
    def set_launcher_options(self):
        if self.current_system.name in ['arolla', 'tsa']:
            self.job.launcher.options = ['--distribution=block:block',
                                         '--cpu_bind=q']


@rfm.simple_test
class AlltoallvTest(CollectivesBaseTest):
    def __init__(self):
        super().__init__(
            {
                'daint': {
                    'nocomm':  0.0171947,
                    'nocomp':  0.0137893,
                    'default': 0.0138493
                },
            }
        )
        self.strict_check = False
        self.sourcesdir = 'https://github.com/eth-cscs/comm_overlap_bench.git'
        self.prebuild_cmds = ['git checkout alltoallv']


@rfm.simple_test
class HaloExchangeTest(CollectivesBaseTest):
    def __init__(self):
        super().__init__(
            {
                'daint': {
                    'nocomm':  0.978306,
                    'nocomp':  1.36716,
                    'default': 2.53509
                },
            }
        )
        self.sourcesdir = 'https://github.com/eth-cscs/comm_overlap_bench.git'
        self.prebuild_cmds = ['git checkout barebones']
