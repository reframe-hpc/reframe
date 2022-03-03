# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn

from hpctestlib.microbenchmarks.mpi.osu import (osu_latency,
                                                osu_bandwidth,
                                                build_osu_benchmarks)


class build_osu_benchmarks_gpu(build_osu_benchmarks):
    @run_after('setup')
    def set_modules(self):
        if self.current_system.name in ['daint', 'dom']:
            self.modules = ['cudatoolkit/21.3_11.2']
            if self.current_environ.name == 'PrgEnv-cray':
                self.prebuild_cmds += ['module sw cce cce/10.0.2']

            if self.current_environ.name == 'PrgEnv-gnu':
                self.prebuild_cmds += ['module sw gcc gcc/10.3.0']

            if self.current_environ.name == 'PrgEnv-intel':
                self.prebuild_cmds += ['module sw intel intel/19.1.1.217']

        elif self.current_system.name in ['arolla', 'tsa']:
            self.modules = ['cuda/10.1.243']


cpu_build_variant = build_osu_benchmarks.get_variant_nums(
    build_type='cpu'
)
cuda_build_variant = build_osu_benchmarks_gpu.get_variant_nums(
    build_type='cuda'
)


@rfm.simple_test
class allreduce_check(osu_latency):
    variant = parameter(['small'], ['large'])
    ctrl_msg_size = 8
    perf_msg_size = 8
    executable = 'osu_allreduce'
    num_tasks_per_node = 1
    num_gpus_per_node  = 1
    osu_binaries = fixture(build_osu_benchmarks, scope='environment',
                           variants=cpu_build_variant)
    strict_check = False
    valid_systems = ['daint:gpu', 'daint:mc']
    valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-nvidia']
    maintainers = ['RS', 'AJ']
    tags = {'production', 'benchmark', 'craype'}
    extra_resources = {
        'switches': {
            'num_switches': 1
        }
    }

    @run_after('init')
    def add_valid_systems(self):
        if self.variant == 'small':
            self.valid_systems += ['dom:gpu', 'dom:mc']

    @run_before('run')
    def set_num_tasks(self):
        self.num_tasks = 6 if self.variant == 'small' else 16

    @run_before('performance')
    def set_performance_patterns(self):
        if self.variant == 'small':
            self.reference = {
                'dom:gpu': {
                    'latency': (5.67, None, 0.05, 'us')
                },
                'daint:gpu': {
                    'latency': (8.66, None, 0.85, 'us')
                },
                'daint:mc': {
                    'latency': (10.90, None, 1.90, 'us')
                }
            }
        else:
            self.reference = {
                'daint:gpu': {
                    'latency': (13.62, None, 1.16, 'us')
                },
                'daint:mc': {
                    'latency': (19.07, None, 1.64, 'us')
                }
            }
        self.perf_patterns = {
            'latency': sn.extractsingle(r'^8\s+(?P<latency>\S+)',
                                        self.stdout, 'latency', float)
        }


@rfm.simple_test
class alltoall_check(osu_latency):
    ctrl_msg_size = 8
    perf_msg_size = 8
    executable = 'osu_alltoall'
    osu_binaries = fixture(build_osu_benchmarks, scope='environment',
                           variants=cpu_build_variant)
    valid_systems = ['daint:gpu', 'dom:gpu']
    valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                           'PrgEnv-intel', 'PrgEnv-nvidia']
    strict_check = False
    reference = {
        'dom:gpu': {
            'latency': (8.23, None, 0.1, 'us')
        },
        'daint:gpu': {
            'latency': (20.73, None, 2.0, 'us')
        }
    }
    num_tasks_per_node = 1
    num_gpus_per_node  = 1
    extra_resources = {
        'switches': {
            'num_switches': 1
        }
    }
    tags = {'production', 'benchmark', 'craype'}
    maintainers = ['RS', 'AJ']

    @run_before('run')
    def set_num_tasks(self):
        if self.current_system.name == 'daint':
            self.num_tasks = 16
        else:
            self.num_tasks = 6


@rfm.simple_test
class alltoall_flex_check(osu_latency):
    descr = 'Flexible Alltoall OSU test'
    executable = 'osu_alltoall'
    ctrl_msg_size = 1048576
    num_tasks_per_node = 1
    num_tasks = 0
    osu_binaries = fixture(build_osu_benchmarks, scope='environment',
                           variants=cpu_build_variant)
    valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
                     'arolla:cn', 'arolla:pn', 'tsa:cn', 'tsa:pn']
    valid_prog_environs = ['PrgEnv-cray']
    tags = {'diagnostic', 'ops', 'benchmark', 'craype'}
    maintainers = ['RS', 'AJ']

    @run_after('init')
    def add_prog_environ(self):
        if self.current_system.name in ['arolla', 'tsa']:
            self.exclusive_access = True
            self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-pgi']


class p2p_config_cscs(rfm.RegressionMixin):
    @run_after('init')
    def cscs_config(self):
        self.num_warmup_iters = 100
        self.num_iters = 1000
        self.num_tasks = 2
        self.num_tasks_per_node = 1
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'arolla:cn', 'tsa:cn']
        if self.current_system.name in ['arolla', 'tsa']:
            self.exclusive_access = True
            self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-pgi']
        else:
            self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                        'PrgEnv-intel', 'PrgEnv-nvidia']

        if not self.device:
            self.valid_systems += ['daint:mc', 'dom:mc',
                                   'eiger:mc', 'pilatus:mc']

        self.exclusive_access = True
        self.strict_check = False
        self.maintainers = ['RS', 'AJ']
        self.tags = {'production', 'benchmark', 'craype'}
        self.extra_resources = {
            'switches': {
                'num_switches': 1
            }
        }


@rfm.simple_test
class p2p_bandwidth_cpu_test(osu_bandwidth, p2p_config_cscs):
    descr = 'P2P bandwidth microbenchmark'
    ctrl_msg_size = 4194304
    perf_msg_size = 4194304
    executable = 'osu_bw'
    osu_binaries = fixture(build_osu_benchmarks, scope='environment',
                           variants=cpu_build_variant)
    reference = {
        'daint:gpu': {
            'bw': (9481.0, -0.10, None, 'MB/s')
        },
        'daint:mc': {
            'bw': (8507, -0.15, None, 'MB/s')
        },
        'dom:gpu': {
            'bw': (9476.3, -0.05, None, 'MB/s')
        },
        'dom:mc': {
            'bw': (9528.0, -0.20, None, 'MB/s')
        },
        'eiger:mc': {
            'bw': (12240.0, -0.10, None, 'MB/s')
        },
        'pilatus:mc': {
            'bw': (12240.0, -0.10, None, 'MB/s')
        },
        # keeping as reference:
        # 'monch:compute': {
        #     'bw': (6317.84, -0.15, None, 'MB/s')
        # },
    }


@rfm.simple_test
class p2p_latency_cpu_test(osu_latency, p2p_config_cscs):
    descr = 'P2P latency microbenchmark'
    ctrl_msg_size = 8
    perf_msg_size = 8
    executable = 'osu_latency'
    osu_binaries = fixture(build_osu_benchmarks, scope='environment',
                           variants=cpu_build_variant)
    reference = {
        'daint:gpu': {
            'latency': (1.40, None, 0.80, 'us')
        },
        'daint:mc': {
            'latency': (1.61, None, 0.70, 'us')
        },
        'dom:gpu': {
            'latency': (1.138, None, 0.10, 'us')
        },
        'dom:mc': {
            'latency': (1.47, None, 0.10, 'us')
        },
        'eiger:mc': {
            'latency': (2.33, None, 0.15, 'us')
        },
        'pilatus:mc': {
            'latency': (2.33, None, 0.15, 'us')
        },
        # keeping as reference:
        # 'monch:compute': {
        #     'latency': (1.27, None, 0.1, 'us')
        # },
    }


class g2g_rdma_cscs(rfm.RegressionMixin):
    @run_before('run')
    def set_rdma_daint(self):
        if self.current_system.name in ['daint', 'dom']:
            self.num_gpus_per_node  = 1
            self.variables = {'MPICH_RDMA_ENABLED_CUDA': '1'}


@rfm.simple_test
class p2p_bandwidth_gpu_test(osu_bandwidth, p2p_config_cscs, g2g_rdma_cscs):
    descr = 'G2G bandwidth microbenchmark'
    device = 'cuda'
    ctrl_msg_size = 4194304
    perf_msg_size = 4194304
    executable = 'osu_bw'
    osu_binaries = fixture(build_osu_benchmarks_gpu, scope='environment',
                           variants=cuda_build_variant)
    reference = {
        'dom:gpu': {
            'bw': (8813.09, -0.05, None, 'MB/s')
        },
        'daint:gpu': {
            'bw': (8560, -0.10, None, 'MB/s')
        },
        '*': {
            'bw': (0, None, None, 'MB/s')
        }
    }


@rfm.simple_test
class p2p_latency_gpu_test(osu_latency, p2p_config_cscs, g2g_rdma_cscs):
    descr = 'G2G latency microbenchmark'
    device = 'cuda'
    ctrl_msg_size = 8
    perf_msg_size = 8
    executable = 'osu_latency'
    osu_binaries = fixture(build_osu_benchmarks_gpu, scope='environment',
                           variants=cuda_build_variant)
    reference = {
        'dom:gpu': {
            'latency': (5.56, None, 0.1, 'us')
        },
        'daint:gpu': {
            'latency': (6.82, None, 0.50, 'us')
        },
        '*': {
            'latency': (0, None, None, 'MB/s')
        }
    }
