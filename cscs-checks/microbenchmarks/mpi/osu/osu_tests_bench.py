# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.16')
@rfm.parameterized_test(*([n, t]
                          for n in [2]
                          for t in [1,2,4,8,16]))
class AlltoallvTest(rfm.RegressionTest):
    def __init__(self, n, t):
        self.strict_check = False
        self.valid_systems = ['tsa:pn']
        self.descr = 'Alltoall OSU microbenchmark'
        self.sourcepath = '5.6.3'
        self.build_system = 'Make'
        self.build_system.makefile = 'Makefile_alltoallv'
        self.build_system.cflags = ['-I.']
        self.executable = './5.6.3/osu_alltoallv'
        # The -x option sets the number of warm-up iterations
        # The -i option sets the number of iterations
        self.executable_opts = ['-x', '1000', '-i', '20000']
        self.valid_prog_environs = ['PrgEnv-gnu-nocuda']
        self.num_tasks_per_node = t
        self.num_tasks= n*t
        self.maintainers = ['MKr, AJ']
        self.sanity_patterns = sn.assert_found(r'^1048576', self.stdout)
###MKr         self.perf_patterns = {
###MKr             'latency': sn.extractsingle(r'^8\s+(?P<latency>\S+)',
###MKr                                         self.stdout, 'latency', float)
###MKr         }
###MKr         self.tags = {'benchmark'}
###FIXME ARE REFERENCE VALUES NEEDED FOR OUR PURPOSE?
###MKr        self.reference = {
###MKr            'dom:gpu': {
###MKr                'latency': (8.23, None, 0.1, 'us')
###MKr            },
###MKr            'daint:gpu': {
###MKr                'latency': (20.73, None, 2.0, 'us')
###MKr            }
###MKr        }


###MKr @rfm.simple_test
###MKr class FlexAlltoallTest(rfm.RegressionTest):
###MKr     def __init__(self):
###MKr         self.valid_systems = ['daint:gpu', 'daint:mc',
###MKr                               'dom:gpu', 'dom:mc',
###MKr                               'tiger:gpu', 'kesch:cn',
###MKr                               'arolla:cn', 'arolla:pn',
###MKr                               'tsa:cn', 'tsa:pn']
###MKr         self.valid_prog_environs = ['PrgEnv-cray']
###MKr         if self.current_system.name == 'kesch':
###MKr             self.exclusive_access = True
###MKr             self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
###MKr                                         'PrgEnv-intel']
###MKr         elif self.current_system.name in ['arolla', 'tsa']:
###MKr             self.exclusive_access = True
###MKr             self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-pgi']
###MKr
###MKr         self.descr = 'Flexible Alltoall OSU test'
###MKr         self.build_system = 'Make'
###MKr         self.build_system.makefile = 'Makefile_alltoall'
###MKr         self.executable = './osu_alltoall'
###MKr         self.maintainers = ['RS', 'AJ']
###MKr         self.num_tasks_per_node = 1
###MKr         self.num_tasks = 0
###MKr         self.sanity_patterns = sn.assert_found(r'^1048576', self.stdout)
###MKr         self.tags = {'diagnostic', 'ops', 'benchmark', 'craype'}
###MKr
###MKr
###MKr @rfm.required_version('>=2.16')
###MKr @rfm.parameterized_test(['small'], ['large'])
###MKr class AllreduceTest(rfm.RegressionTest):
###MKr     def __init__(self, variant):
###MKr         self.strict_check = False
###MKr         self.valid_systems = ['daint:gpu', 'daint:mc']
###MKr         if variant == 'small':
###MKr             self.valid_systems += ['dom:gpu', 'dom:mc', 'tiger:gpu']
###MKr
###MKr         self.descr = 'Allreduce OSU microbenchmark'
###MKr         self.build_system = 'Make'
###MKr         self.build_system.makefile = 'Makefile_allreduce'
###MKr         self.executable = './osu_allreduce'
###MKr         # The -x option controls the number of warm-up iterations
###MKr         # The -i option controls the number of iterations
###MKr         self.executable_opts = ['-m', '8', '-x', '1000', '-i', '20000']
###MKr         self.valid_prog_environs = ['PrgEnv-gnu']
###MKr         self.maintainers = ['RS', 'AJ']
###MKr         self.sanity_patterns = sn.assert_found(r'^8', self.stdout)
###MKr         self.perf_patterns = {
###MKr             'latency': sn.extractsingle(r'^8\s+(?P<latency>\S+)',
###MKr                                         self.stdout, 'latency', float)
###MKr         }
###MKr         self.tags = {'production', 'benchmark', 'craype'}
###MKr         if variant == 'small':
###MKr             self.num_tasks = 6
###MKr             self.reference = {
###MKr                 'dom:gpu': {
###MKr                     'latency': (5.67, None, 0.05, 'us')
###MKr                 },
###MKr                 'daint:gpu': {
###MKr                     'latency': (9.30, None, 0.75, 'us')
###MKr                 },
###MKr                 'daint:mc': {
###MKr                     'latency': (11.74, None, 1.51, 'us')
###MKr                 }
###MKr             }
###MKr         else:
###MKr             self.num_tasks = 16
###MKr             self.reference = {
###MKr                 'daint:gpu': {
###MKr                     'latency': (13.62, None, 1.16, 'us')
###MKr                 },
###MKr                 'daint:mc': {
###MKr                     'latency': (19.07, None, 1.64, 'us')
###MKr                 }
###MKr             }
###MKr
###MKr         self.num_tasks_per_node = 1
###MKr         self.num_gpus_per_node  = 1
###MKr         self.extra_resources = {
###MKr             'switches': {
###MKr                 'num_switches': 1
###MKr             }
###MKr         }
###MKr
###MKr
###MKr class P2PBaseTest(rfm.RegressionTest):
###MKr     def __init__(self):
###MKr         self.exclusive_access = True
###MKr         self.strict_check = False
###MKr         self.num_tasks = 2
###MKr         self.num_tasks_per_node = 1
###MKr         self.descr = 'P2P microbenchmark'
###MKr         self.build_system = 'Make'
###MKr         self.build_system.makefile = 'Makefile_p2p'
###MKr         if self.current_system.name == 'kesch':
###MKr             self.exclusive_access = True
###MKr             self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu']
###MKr         elif self.current_system.name in ['arolla', 'tsa']:
###MKr             self.exclusive_access = True
###MKr             self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-pgi']
###MKr         else:
###MKr             self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu',
###MKr                                         'PrgEnv-intel']
###MKr         self.maintainers = ['RS', 'AJ']
###MKr         self.tags = {'production', 'benchmark', 'craype'}
###MKr         self.sanity_patterns = sn.assert_found(r'^4194304', self.stdout)
###MKr
###MKr         self.extra_resources = {
###MKr             'switches': {
###MKr                 'num_switches': 1
###MKr             }
###MKr         }
###MKr
###MKr
###MKr @rfm.required_version('>=2.16')
###MKr @rfm.simple_test
###MKr class P2PCPUBandwidthTest(P2PBaseTest):
###MKr     def __init__(self):
###MKr         super().__init__()
###MKr         self.valid_systems = ['daint:gpu', 'daint:mc', 'tiger:gpu',
###MKr                               'dom:gpu', 'dom:mc', 'kesch:cn',
###MKr                               'arolla:cn', 'tsa:cn']
###MKr         self.executable = './p2p_osu_bw'
###MKr         self.executable_opts = ['-x', '100', '-i', '1000']
###MKr
###MKr         self.reference = {
###MKr             'daint:gpu': {
###MKr                 'bw': (9607.0, -0.10, None, 'MB/s')
###MKr             },
###MKr             'daint:mc': {
###MKr                 'bw': (9649.0, -0.10, None, 'MB/s')
###MKr             },
###MKr             'dom:gpu': {
###MKr                 'bw': (9476.3, -0.05, None, 'MB/s')
###MKr             },
###MKr             'dom:mc': {
###MKr                 'bw': (9528.0, -0.20, None, 'MB/s')
###MKr             },
###MKr             # keeping as reference:
###MKr             # 'monch:compute': {
###MKr             #     'bw': (6317.84, -0.15, None, 'MB/s')
###MKr             # },
###MKr             'kesch:cn': {
###MKr                 'bw': (6311.48, -0.15, None, 'MB/s')
###MKr             },
###MKr         }
###MKr         self.perf_patterns = {
###MKr             'bw': sn.extractsingle(r'^4194304\s+(?P<bw>\S+)',
###MKr                                    self.stdout, 'bw', float)
###MKr         }
###MKr
###MKr
###MKr @rfm.required_version('>=2.16')
###MKr @rfm.simple_test
###MKr class P2PCPULatencyTest(P2PBaseTest):
###MKr     def __init__(self):
###MKr         super().__init__()
###MKr         self.valid_systems = ['daint:gpu', 'daint:mc', 'tiger:gpu',
###MKr                               'dom:gpu', 'dom:mc', 'kesch:cn',
###MKr                               'arolla:cn', 'tsa:cn']
###MKr         self.executable_opts = ['-x', '100', '-i', '1000']
###MKr
###MKr         self.executable = './p2p_osu_latency'
###MKr         self.reference = {
###MKr             'daint:gpu': {
###MKr                 'latency': (1.30, None, 0.70, 'us')
###MKr             },
###MKr             'daint:mc': {
###MKr                 'latency': (1.61, None, 0.85, 'us')
###MKr             },
###MKr             'dom:gpu': {
###MKr                 'latency': (1.138, None, 0.10, 'us')
###MKr             },
###MKr             'dom:mc': {
###MKr                 'latency': (1.24, None, 0.15, 'us')
###MKr             },
###MKr             # keeping as reference:
###MKr             # 'monch:compute': {
###MKr             #     'latency': (1.27, None, 0.1, 'us')
###MKr             # },
###MKr             'kesch:cn': {
###MKr                 'latency': (1.17, None, 0.1, 'us')
###MKr             }
###MKr         }
###MKr         self.perf_patterns = {
###MKr             'latency': sn.extractsingle(r'^8\s+(?P<latency>\S+)',
###MKr                                         self.stdout, 'latency', float)
###MKr         }
###MKr
###MKr
###MKr @rfm.required_version('>=2.16')
###MKr @rfm.simple_test
###MKr class G2GBandwidthTest(P2PBaseTest):
###MKr     def __init__(self):
###MKr         super().__init__()
###MKr         self.valid_systems = ['daint:gpu', 'dom:gpu', 'kesch:cn', 'tiger:gpu',
###MKr                               'arolla:cn', 'tsa:cn']
###MKr         self.num_gpus_per_node = 1
###MKr         self.executable = './p2p_osu_bw'
###MKr         self.executable_opts = ['-x', '100', '-i', '1000', '-d',
###MKr                                 'cuda', 'D', 'D']
###MKr
###MKr         self.reference = {
###MKr             'dom:gpu': {
###MKr                 'bw': (8813.09, -0.05, None, 'MB/s')
###MKr             },
###MKr             'daint:gpu': {
###MKr                 'bw': (8765.65, -0.1, None, 'MB/s')
###MKr             },
###MKr             'kesch:cn': {
###MKr                 'bw': (6288.98, -0.1, None, 'MB/s')
###MKr             },
###MKr             '*': {
###MKr                 'bw': (0, None, None, 'MB/s')
###MKr             }
###MKr         }
###MKr         self.perf_patterns = {
###MKr             'bw': sn.extractsingle(r'^4194304\s+(?P<bw>\S+)',
###MKr                                    self.stdout, 'bw', float)
###MKr         }
###MKr         if self.current_system.name in ['daint', 'dom', 'tiger']:
###MKr             self.num_gpus_per_node  = 1
###MKr             self.modules = ['craype-accel-nvidia60']
###MKr             self.variables = {'MPICH_RDMA_ENABLED_CUDA': '1'}
###MKr         elif self.current_system.name == 'kesch':
###MKr             self.modules = ['cudatoolkit/8.0.61']
###MKr             self.variables = {'MV2_USE_CUDA': '1'}
###MKr         elif self.current_system.name in ['arolla', 'tsa']:
###MKr             self.modules = ['cuda/10.1.243']
###MKr             self.build_system.ldflags = ['-L$EBROOTCUDA/lib64',
###MKr                                          '-lcudart', '-lcuda']
###MKr
###MKr         self.build_system.cppflags = ['-D_ENABLE_CUDA_']
###MKr
###MKr
###MKr @rfm.required_version('>=2.16')
###MKr @rfm.simple_test
###MKr class G2GLatencyTest(P2PBaseTest):
###MKr     def __init__(self):
###MKr         super().__init__()
###MKr         self.valid_systems = ['daint:gpu', 'dom:gpu', 'kesch:cn', 'tiger:gpu',
###MKr                               'arolla:cn', 'tsa:cn']
###MKr         self.num_gpus_per_node = 1
###MKr         self.executable = './p2p_osu_latency'
###MKr         self.executable_opts = ['-x', '100', '-i', '1000', '-d',
###MKr                                 'cuda', 'D', 'D']
###MKr
###MKr         self.reference = {
###MKr             'dom:gpu': {
###MKr                 'latency': (5.56, None, 0.1, 'us')
###MKr             },
###MKr             'daint:gpu': {
###MKr                 'latency': (6.8, None, 0.65, 'us')
###MKr             },
###MKr             'kesch:cn': {
###MKr                 'latency': (23.09, None, 0.1, 'us')
###MKr             }
###MKr         }
###MKr         self.perf_patterns = {
###MKr             'latency': sn.extractsingle(r'^8\s+(?P<latency>\S+)',
###MKr                                         self.stdout, 'latency', float)
###MKr         }
###MKr         if self.current_system.name in ['daint', 'dom', 'tiger']:
###MKr             self.num_gpus_per_node  = 1
###MKr             self.modules = ['craype-accel-nvidia60']
###MKr             self.variables = {'MPICH_RDMA_ENABLED_CUDA': '1'}
###MKr         elif self.current_system.name == 'kesch':
###MKr             self.modules = ['cudatoolkit/8.0.61']
###MKr             self.variables = {'MV2_USE_CUDA': '1'}
###MKr         elif self.current_system.name in ['arolla', 'tsa']:
###MKr             self.modules = ['cuda/10.1.243']
###MKr             self.build_system.ldflags = ['-L$EBROOTCUDA/lib64',
###MKr                                          '-lcudart', '-lcuda']
###MKr
###MKr         self.build_system.cppflags = ['-D_ENABLE_CUDA_']
