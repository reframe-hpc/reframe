# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.16')
@rfm.parameterized_test(['production'])
class AlltoallTest(rfm.RegressionTest):
    def __init__(self, variant):
        self.strict_check = False
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.descr = 'Alltoall OSU microbenchmark'
        self.build_system = 'Make'
        self.build_system.makefile = 'Makefile_alltoall'
        self.executable = './osu_alltoall'
        # The -m option sets the maximum message size
        # The -x option sets the number of warm-up iterations
        # The -i option sets the number of iterations
        self.executable_opts = ['-m', '8', '-x', '1000', '-i', '20000']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                    'PrgEnv-intel']
        self.maintainers = ['RS', 'AJ']
        self.sanity_patterns = sn.assert_found(r'^8', self.stdout)
        self.perf_patterns = {
            'latency': sn.extractsingle(r'^8\s+(?P<latency>\S+)',
                                        self.stdout, 'latency', float)
        }
        self.tags = {variant, 'benchmark', 'craype'}
        self.reference = {
            'dom:gpu': {
                'latency': (8.23, None, 0.1, 'us')
            },
            'daint:gpu': {
                'latency': (20.73, None, 2.0, 'us')
            }
        }
        self.num_tasks_per_node = 1
        self.num_gpus_per_node  = 1
        if self.current_system.name == 'daint':
            self.num_tasks = 16
        else:
            self.num_tasks = 6

        self.extra_resources = {
            'switches': {
                'num_switches': 1
            }
        }


@rfm.simple_test
class FlexAlltoallTest(rfm.RegressionTest):
    def __init__(self):
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
                              'arolla:cn', 'arolla:pn', 'tsa:cn', 'tsa:pn']
        self.valid_prog_environs = ['PrgEnv-cray']
        if self.current_system.name in ['arolla', 'tsa']:
            self.exclusive_access = True
            self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-pgi']

        self.descr = 'Flexible Alltoall OSU test'
        self.build_system = 'Make'
        self.build_system.makefile = 'Makefile_alltoall'
        self.executable = './osu_alltoall'
        self.maintainers = ['RS', 'AJ']
        self.num_tasks_per_node = 1
        self.num_tasks = 0
        self.sanity_patterns = sn.assert_found(r'^1048576', self.stdout)
        self.tags = {'diagnostic', 'ops', 'benchmark', 'craype'}


@rfm.required_version('>=2.16')
@rfm.parameterized_test(['small'], ['large'])
class AllreduceTest(rfm.RegressionTest):
    def __init__(self, variant):
        self.strict_check = False
        self.valid_systems = ['daint:gpu', 'daint:mc']
        if variant == 'small':
            self.valid_systems += ['dom:gpu', 'dom:mc']

        self.descr = 'Allreduce OSU microbenchmark'
        self.build_system = 'Make'
        self.build_system.makefile = 'Makefile_allreduce'
        self.executable = './osu_allreduce'
        # The -x option controls the number of warm-up iterations
        # The -i option controls the number of iterations
        self.executable_opts = ['-m', '8', '-x', '1000', '-i', '20000']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.maintainers = ['RS', 'AJ']
        self.sanity_patterns = sn.assert_found(r'^8', self.stdout)
        self.perf_patterns = {
            'latency': sn.extractsingle(r'^8\s+(?P<latency>\S+)',
                                        self.stdout, 'latency', float)
        }
        self.tags = {'production', 'benchmark', 'craype'}
        if variant == 'small':
            self.num_tasks = 6
            self.reference = {
                'dom:gpu': {
                    'latency': (5.67, None, 0.05, 'us')
                },
                'daint:gpu': {
                    'latency': (9.30, None, 0.75, 'us')
                },
                'daint:mc': {
                    'latency': (11.74, None, 1.51, 'us')
                }
            }
        else:
            self.num_tasks = 16
            self.reference = {
                'daint:gpu': {
                    'latency': (13.62, None, 1.16, 'us')
                },
                'daint:mc': {
                    'latency': (19.07, None, 1.64, 'us')
                }
            }

        self.num_tasks_per_node = 1
        self.num_gpus_per_node  = 1
        self.extra_resources = {
            'switches': {
                'num_switches': 1
            }
        }


class P2PBaseTest(rfm.RegressionTest):
    def __init__(self):
        self.exclusive_access = True
        self.strict_check = False
        self.num_tasks = 2
        self.num_tasks_per_node = 1
        self.descr = 'P2P microbenchmark'
        self.build_system = 'Make'
        self.build_system.makefile = 'Makefile_p2p'
        if self.current_system.name in ['arolla', 'tsa']:
            self.exclusive_access = True
            self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-pgi']
        else:
            self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                        'PrgEnv-intel']
        self.maintainers = ['RS', 'AJ']
        self.tags = {'production', 'benchmark', 'craype'}
        self.sanity_patterns = sn.assert_found(r'^4194304', self.stdout)

        self.extra_resources = {
            'switches': {
                'num_switches': 1
            }
        }


@rfm.required_version('>=2.16')
@rfm.simple_test
class P2PCPUBandwidthTest(P2PBaseTest):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
                              'arolla:cn', 'tsa:cn']
        self.executable = './p2p_osu_bw'
        self.executable_opts = ['-x', '100', '-i', '1000']

        self.reference = {
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
            # keeping as reference:
            # 'monch:compute': {
            #     'bw': (6317.84, -0.15, None, 'MB/s')
            # },
        }
        self.perf_patterns = {
            'bw': sn.extractsingle(r'^4194304\s+(?P<bw>\S+)',
                                   self.stdout, 'bw', float)
        }


@rfm.required_version('>=2.16')
@rfm.simple_test
class P2PCPULatencyTest(P2PBaseTest):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
                              'arolla:cn', 'tsa:cn']
        self.executable_opts = ['-x', '100', '-i', '1000']

        self.executable = './p2p_osu_latency'
        self.reference = {
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
                'latency': (1.24, None, 0.15, 'us')
            },
            # keeping as reference:
            # 'monch:compute': {
            #     'latency': (1.27, None, 0.1, 'us')
            # },
        }
        self.perf_patterns = {
            'latency': sn.extractsingle(r'^8\s+(?P<latency>\S+)',
                                        self.stdout, 'latency', float)
        }


@rfm.required_version('>=2.16')
@rfm.simple_test
class G2GBandwidthTest(P2PBaseTest):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'arolla:cn', 'tsa:cn']
        self.num_gpus_per_node = 1
        self.executable = './p2p_osu_bw'
        self.executable_opts = ['-x', '100', '-i', '1000', '-d',
                                'cuda', 'D', 'D']

        self.reference = {
            'dom:gpu': {
                'bw': (8813.09, -0.05, None, 'MB/s')
            },
            'daint:gpu': {
                'bw': (8765.65, -0.1, None, 'MB/s')
            },
            '*': {
                'bw': (0, None, None, 'MB/s')
            }
        }
        self.perf_patterns = {
            'bw': sn.extractsingle(r'^4194304\s+(?P<bw>\S+)',
                                   self.stdout, 'bw', float)
        }
        if self.current_system.name in ['daint', 'dom']:
            self.num_gpus_per_node  = 1
            self.modules = ['craype-accel-nvidia60']
            self.variables = {'MPICH_RDMA_ENABLED_CUDA': '1'}
        elif self.current_system.name in ['arolla', 'tsa']:
            self.modules = ['cuda/10.1.243']
            self.build_system.ldflags = ['-L$EBROOTCUDA/lib64',
                                         '-lcudart', '-lcuda']

        self.build_system.cppflags = ['-D_ENABLE_CUDA_']


@rfm.required_version('>=2.16')
@rfm.simple_test
class G2GLatencyTest(P2PBaseTest):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'arolla:cn', 'tsa:cn']
        self.num_gpus_per_node = 1
        self.executable = './p2p_osu_latency'
        self.executable_opts = ['-x', '100', '-i', '1000', '-d',
                                'cuda', 'D', 'D']

        self.reference = {
            'dom:gpu': {
                'latency': (5.56, None, 0.1, 'us')
            },
            'daint:gpu': {
                'latency': (6.8, None, 0.65, 'us')
            },
        }
        self.perf_patterns = {
            'latency': sn.extractsingle(r'^8\s+(?P<latency>\S+)',
                                        self.stdout, 'latency', float)
        }
        if self.current_system.name in ['daint', 'dom']:
            self.num_gpus_per_node  = 1
            self.modules = ['craype-accel-nvidia60']
            self.variables = {'MPICH_RDMA_ENABLED_CUDA': '1'}
        elif self.current_system.name in ['arolla', 'tsa']:
            self.modules = ['cuda/10.1.243']
            self.build_system.ldflags = ['-L$EBROOTCUDA/lib64',
                                         '-lcudart', '-lcuda']

        self.build_system.cppflags = ['-D_ENABLE_CUDA_']
