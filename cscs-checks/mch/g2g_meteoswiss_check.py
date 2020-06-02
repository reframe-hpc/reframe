# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.parameterized_test([1], [2])
class G2GMeteoswissTest(rfm.RegressionTest):
    def __init__(self, g2g):
        self.descr = 'G2G Meteoswiss check with G2G=%s' % g2g
        self.strict_check = False
        self.valid_systems = ['kesch:cn']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.exclusive_access = True
        self.modules = ['cmake']
        self.prerun_cmds = ["export EXECUTABLE=$(ls src/ | "
                            "grep 'GNU.*MVAPICH.*CUDA.*kesch.*')"]
        self.executable = 'build/src/comm_overlap_benchmark'
        self.sourcesdir = ('https://github.com/MeteoSwiss-APN/'
                           'comm_overlap_bench.git')
        self.prebuild_cmds = ['git checkout barebones']
        self.build_system = 'CMake'
        self.build_system.builddir = 'build'
        self.build_system.config_opts = ['-DMPI_VENDOR=mvapich2',
                                         '-DCUDA_COMPUTE_CAPABILITY="sm_37"',
                                         '-DCMAKE_BUILD_TYPE=Release',
                                         '-DENABLE_MPI_TIMER=ON']
        self.build_system.max_concurrency = 1
        self.maintainers = ['AJ', 'LM']
        self.tags = {'production', 'mch'}
        self.num_tasks = 2
        self.num_gpus_per_node  = 2
        cuda_visible_devices = {1: r'CUDA_VISIBLE_DEVICES: '
                                   r'\[0: \d\] \[1: \d\]',
                                2: r'CUDA_VISIBLE_DEVICES: '
                                   r'\[0: \d,\d\] \[1: \d,\d\]'}
        self.sanity_patterns = sn.all([
            sn.assert_found('ELAPSED TIME:', self.stdout),
            sn.assert_found(cuda_visible_devices[g2g], self.stdout)
        ])
        self.perf_patterns = {
            'time': sn.extractsingle(r'ELAPSED TIME:\s+(?P<time>\S+)',
                                     self.stdout, 'time', float)
        }
        self.reference = {
            'kesch:cn': {'time': (3.461, None, 0.2, 's')}
        }
        self.variables = {'G2G': str(g2g)}
