# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class AlltoallTest(rfm.RegressionTest):
    variant = parameter(['production'])
    strict_check = False
    valid_systems = ['daint:gpu', 'dom:gpu']
    descr = 'Alltoall OSU microbenchmark'
    build_system = 'Make'
    executable = './osu_alltoall'
    # The -m option sets the maximum message size
    # The -x option sets the number of warm-up iterations
    # The -i option sets the number of iterations
    executable_opts = ['-m', '8', '-x', '1000', '-i', '20000']
    valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                           'PrgEnv-intel', 'PrgEnv-nvidia']
    maintainers = ['RS', 'AJ']
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

    @run_after('init')
    def set_tags(self):
        self.tags = {self.variant, 'benchmark', 'craype'}

    @run_before('compile')
    def set_makefile(self):
        self.build_system.makefile = 'Makefile_alltoall'

    @run_before('run')
    def set_num_tasks(self):
        if self.current_system.name == 'daint':
            self.num_tasks = 16
        else:
            self.num_tasks = 6

    @run_before('sanity')
    def set_sanity(self):
        self.sanity_patterns = sn.assert_found(r'^8', self.stdout)

    @run_before('performance')
    def set_performance_patterns(self):
        self.perf_patterns = {
            'latency': sn.extractsingle(r'^8\s+(?P<latency>\S+)',
                                        self.stdout, 'latency', float)
        }


@rfm.simple_test
class FlexAlltoallTest(rfm.RegressionTest):
    valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
                     'arolla:cn', 'arolla:pn', 'tsa:cn', 'tsa:pn']
    valid_prog_environs = ['PrgEnv-cray']
    descr = 'Flexible Alltoall OSU test'
    build_system = 'Make'
    executable = './osu_alltoall'
    maintainers = ['RS', 'AJ']
    num_tasks_per_node = 1
    num_tasks = 0
    tags = {'diagnostic', 'ops', 'benchmark', 'craype'}

    @run_after('init')
    def add_prog_environ(self):
        if self.current_system.name in ['arolla', 'tsa']:
            self.exclusive_access = True
            self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-pgi']

    @run_before('compile')
    def set_makefile(self):
        self.build_system.makefile = 'Makefile_alltoall'

    @run_before('sanity')
    def set_sanity(self):
        self.sanity_patterns = sn.assert_found(r'^1048576', self.stdout)


@rfm.simple_test
class AllreduceTest(rfm.RegressionTest):
    variant = parameter(['small'], ['large'])
    strict_check = False
    valid_systems = ['daint:gpu', 'daint:mc']
    descr = 'Allreduce OSU microbenchmark'
    build_system = 'Make'
    executable = './osu_allreduce'
    # The -x option controls the number of warm-up iterations
    # The -i option controls the number of iterations
    executable_opts = ['-m', '8', '-x', '1000', '-i', '20000']
    valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-nvidia']
    maintainers = ['RS', 'AJ']
    tags = {'production', 'benchmark', 'craype'}
    num_tasks_per_node = 1
    num_gpus_per_node  = 1
    extra_resources = {
        'switches': {
            'num_switches': 1
        }
    }

    @run_after('init')
    def add_valid_systems(self):
        if self.variant == 'small':
            self.valid_systems += ['dom:gpu', 'dom:mc']

    @run_before('compile')
    def set_makefile(self):
        self.build_system.makefile = 'Makefile_allreduce'

    @run_before('run')
    def set_num_tasks(self):
        self.num_tasks = 6 if self.variant == 'small' else 16

    @run_before('sanity')
    def set_sanity(self):
        self.sanity_patterns = sn.assert_found(r'^8', self.stdout)

    @run_before('performance')
    def set_performance_patterns(self):
        if self.variant == 'small':
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


class P2PBaseTest(rfm.RegressionTest):
    exclusive_access = True
    strict_check = False
    num_tasks = 2
    num_tasks_per_node = 1
    descr = 'P2P microbenchmark'
    build_system = 'Make'
    maintainers = ['RS', 'AJ']
    tags = {'production', 'benchmark', 'craype'}
    extra_resources = {
        'switches': {
            'num_switches': 1
        }
    }

    @run_after('init')
    def add_valid_prog_environs(self):
        if self.current_system.name in ['arolla', 'tsa']:
            self.exclusive_access = True
            self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-pgi']
        else:
            self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
                                        'PrgEnv-intel', 'PrgEnv-nvidia']

    @run_before('compile')
    def set_makefile(self):
        self.build_system.makefile = 'Makefile_p2p'

    @run_before('sanity')
    def set_sanity(self):
        self.sanity_patterns = sn.assert_found(r'^4194304', self.stdout)


@rfm.simple_test
class P2PCPUBandwidthTest(P2PBaseTest):
    valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
                     'arolla:cn', 'tsa:cn', 'eiger:mc', 'pilatus:mc']
    executable = './p2p_osu_bw'
    executable_opts = ['-x', '100', '-i', '1000']
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

    @run_before('performance')
    def set_performance_patterns(self):
        self.perf_patterns = {
            'bw': sn.extractsingle(r'^4194304\s+(?P<bw>\S+)',
                                   self.stdout, 'bw', float)
        }


@rfm.simple_test
class P2PCPULatencyTest(P2PBaseTest):
    valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
                     'arolla:cn', 'tsa:cn', 'eiger:mc', 'pilatus:mc']
    executable = './p2p_osu_latency'
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
            'latency': (1.24, None, 0.15, 'us')
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

    @run_before('performance')
    def set_performance_patterns(self):
        self.perf_patterns = {
            'bw': sn.extractsingle(r'^4194304\s+(?P<bw>\S+)',
                                   self.stdout, 'bw', float)
        }
        self.perf_patterns = {
            'latency': sn.extractsingle(r'^8\s+(?P<latency>\S+)',
                                        self.stdout, 'latency', float)
        }


@rfm.simple_test
class G2GBandwidthTest(P2PBaseTest):
    valid_systems = ['daint:gpu', 'dom:gpu', 'arolla:cn', 'tsa:cn']
    num_gpus_per_node = 1
    executable = './p2p_osu_bw'
    executable_opts = ['-x', '100', '-i', '1000', '-d',
                       'cuda', 'D', 'D']

    reference = {
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

    @run_before('performance')
    def set_performance_patterns(self):
        self.perf_patterns = {
            'bw': sn.extractsingle(r'^4194304\s+(?P<bw>\S+)',
                                   self.stdout, 'bw', float)
        }

    @run_before('compile')
    def set_cpp_flags(self):
        self.build_system.cppflags = ['-D_ENABLE_CUDA_']

    @run_before('compile')
    def set_modules(self):
        if self.current_system.name in ['daint', 'dom']:
            self.num_gpus_per_node  = 1
            self.variables = {'MPICH_RDMA_ENABLED_CUDA': '1'}
            if self.current_environ.name == 'PrgEnv-nvidia':
                self.modules = [ 'cudatoolkit/21.3_11.2']
            else:
                self.modules = ['craype-accel-nvidia60']
        elif self.current_system.name in ['arolla', 'tsa']:
            self.modules = ['cuda/10.1.243']
            self.build_system.ldflags = ['-L$EBROOTCUDA/lib64',
                                         '-lcudart', '-lcuda']


@rfm.simple_test
class G2GLatencyTest(P2PBaseTest):
    valid_systems = ['daint:gpu', 'dom:gpu', 'arolla:cn', 'tsa:cn']
    num_gpus_per_node = 1
    executable = './p2p_osu_latency'
    executable_opts = ['-x', '100', '-i', '1000', '-d',
                       'cuda', 'D', 'D']

    reference = {
        'dom:gpu': {
            'latency': (5.56, None, 0.1, 'us')
        },
        'daint:gpu': {
            'latency': (6.8, None, 0.65, 'us')
        },
    }

    @run_before('performance')
    def set_performance_patterns(self):
        self.perf_patterns = {
            'latency': sn.extractsingle(r'^8\s+(?P<latency>\S+)',
                                        self.stdout, 'latency', float)
        }

    @run_before('compile')
    def set_cpp_flags(self):
        self.build_system.cppflags = ['-D_ENABLE_CUDA_']

    @run_before('compile')
    def set_modules(self):
        if self.current_system.name in ['daint', 'dom']:
            self.num_gpus_per_node  = 1
            self.variables = {'MPICH_RDMA_ENABLED_CUDA': '1'}
            if self.current_environ.name == 'PrgEnv-nvidia':
                self.modules = ['cudatoolkit/21.3_11.2']
            else:
                self.modules = ['craype-accel-nvidia60']
        elif self.current_system.name in ['arolla', 'tsa']:
            self.modules = ['cuda/10.1.243']
            self.build_system.ldflags = ['-L$EBROOTCUDA/lib64',
                                         '-lcudart', '-lcuda']
