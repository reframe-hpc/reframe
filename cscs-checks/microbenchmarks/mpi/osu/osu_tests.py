# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm

from hpctestlib.microbenchmarks.mpi.osu import (osu_pt2pt_latency_check,
                                                osu_pt2pt_bandwidth_check,
                                                osu_allreduce_latency_check,
                                                osu_alltoall_latency_check,
                                                build_osu_benchmarks)


class cscs_build_osu_benchmarks_cuda(build_osu_benchmarks):
    @run_after('init')
    def setup_environment(self):
        if self.current_system.name in ['daint', 'dom']:
            self.modules = ['cudatoolkit/21.3_11.2']
        elif self.current_system.name in ['arolla', 'tsa']:
            self.modules = ['cuda/10.1.243']


cpu_build_variant = build_osu_benchmarks.get_variant_nums(
    build_type='cpu'
)
cuda_build_variant = cscs_build_osu_benchmarks_cuda.get_variant_nums(
    build_type='cuda'
)


@rfm.simple_test
class cscs_allreduce_latency_check(osu_allreduce_latency_check):
    num_nodes = parameter([6, 16])
    osu_binaries = fixture(build_osu_benchmarks, scope='environment',
                           variants=cpu_build_variant)
    valid_systems = ['dom:gpu', 'dom:mc']
    valid_prog_environs = ['PrgEnv-gnu']
    maintainers = ['@rsarm', '@ajocksch']
    tags = {'production', 'benchmark', 'craype'}
    extra_resources = {
        'switches': {
            'num_switches': 1
        }
    }
    allref = {
        6: {
            'dom:gpu': {
                'latency': (5.67, None, 0.05, 'us')
            },
            'daint:gpu': {
                'latency': (9.30, None, 0.75, 'us')
            },
            'daint:mc': {
                'latency': (11.74, None, 1.51, 'us')
            }
        },
        16: {
            'daint:gpu': {
                'latency': (13.62, None, 1.16, 'us')
            },
            'daint:mc': {
                'latency': (19.07, None, 1.64, 'us')
            }
        }
    }

    @run_after('init')
    def setup_from_params(self):
        if self.num_nodes == 16:
            self.valid_systems += ['daint:gpu', 'daint:mc']

        self.num_tasks = self.num_nodes
        self.reference = self.allref[self.num_nodes]


@rfm.simple_test
class cscs_alltoall_check(osu_alltoall_latency_check):
    executable = 'osu_alltoall'
    osu_binaries = fixture(build_osu_benchmarks, scope='environment',
                           variants=cpu_build_variant)
    valid_systems = ['daint:gpu', 'dom:gpu']
    valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                           'PrgEnv-intel', 'PrgEnv-nvidia']
    reference = {
        'dom:gpu': {
            'latency': (8.23, None, 0.1, 'us')
        },
        'daint:gpu': {
            'latency': (20.73, None, 2.0, 'us')
        }
    }
    num_tasks_per_node = 1
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


class p2p_config_cscs(rfm.RegressionMixin):
    @run_after('init')
    def cscs_config(self):
        self.num_warmup_iters = 100
        self.num_iters = 1000
        self.num_tasks = 2
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'arolla:cn', 'tsa:cn']
        if self.current_system.name in ['arolla', 'tsa']:
            self.exclusive_access = True
            self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-pgi']
        else:
            self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                        'PrgEnv-intel', 'PrgEnv-nvidia']

        if not self.device_buffers:
            self.valid_systems += ['daint:mc', 'dom:mc',
                                   'eiger:mc', 'pilatus:mc']

        self.exclusive_access = True
        self.maintainers = ['RS', 'AJ']
        self.tags = {'production', 'benchmark', 'craype'}
        self.extra_resources = {
            'switches': {
                'num_switches': 1
            }
        }


@rfm.simple_test
class cscs_p2p_bandwidth_cpu_test(osu_pt2pt_bandwidth_check, p2p_config_cscs):
    descr = 'P2P bandwidth microbenchmark'
    message_size = 4194304
    executable = 'osu_bw'
    osu_binaries = fixture(build_osu_benchmarks, scope='environment',
                           variants=cpu_build_variant)
    reference = {
        'daint:gpu': {
            'bw': (9607.0, -0.10, None, 'MB/s')
        },
        'daint:mc': {
            'bw': (9649.0, -0.10, None, 'MB/s')
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
class cscs_p2p_latency_cpu_test(osu_latency_pt2pt_check, p2p_config_cscs):
    descr = 'P2P latency microbenchmark'
    executable = 'osu_latency'
    osu_binaries = fixture(build_osu_benchmarks, scope='environment',
                           variants=cpu_build_variant)
    reference = {
        'daint:gpu': {
            'latency': (1.30, None, 0.70, 'us')
        },
        'daint:mc': {
            'latency': (1.61, None, 0.85, 'us')
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
class cscs_p2p_bandwidth_gpu_test(osu_pt2pt_bandwidth_check, p2p_config_cscs,
                                  g2g_rdma_cscs):
    descr = 'G2G bandwidth microbenchmark'
    device_buffers = 'cuda'
    message_size = 4194304
    executable = 'osu_bw'
    osu_binaries = fixture(build_osu_benchmarks_gpu, scope='environment',
                           variants=cuda_build_variant)
    reference = {
        'dom:gpu': {
            'bw': (8813.09, -0.05, None, 'MB/s')
        },
        'daint:gpu': {
            'bw': (8765.65, -0.1, None, 'MB/s')
        }
    }


@rfm.simple_test
class cscs_p2p_latency_gpu_test(osu_pt2pt_latency_check, p2p_config_cscs,
                                g2g_rdma_cscs):
    descr = 'G2G latency microbenchmark'
    device_buffers = 'cuda'
    executable = 'osu_latency'
    osu_binaries = fixture(build_osu_benchmarks_gpu, scope='environment',
                           variants=cuda_build_variant)
    reference = {
        'dom:gpu': {
            'latency': (5.56, None, 0.1, 'us')
        },
        'daint:gpu': {
            'latency': (6.8, None, 0.65, 'us')
        }
    }
