# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe.utility.sanity as sn
import reframe as rfm


@rfm.parameterized_test(['peerAccess'], ['noPeerAccess'])
class P2pBandwidthCheck(rfm.RegressionTest):
    def __init__(self, peerAccess):
        self.valid_systems = ['tsa:cn', 'arola:cn',
                              'ault:amdv100', 'ault:intelv100',
                              'ault:amda100']
        self.valid_prog_environs = ['PrgEnv-gnu']
        if self.current_system.name in ['arolla', 'tsa']:
            self.valid_prog_environs = ['PrgEnv-gnu-nompi']

        self.build_system = 'SingleSource'
        self.sourcepath = 'p2p_bandwidth.cu'
        self.executable = 'p2p_bandwidth.x'
        self.exclusive_access = True

        # Perform a single bandwidth test with a buffer size of 1024MB
        copy_size = 1073741824

        self.build_system.cxxflags = ['-I.', '-m64',
                                      '-std=c++11', '-lnvidia-ml',
                                      f'-DCOPY={copy_size}']
        if (peerAccess == 'peerAccess'):
            self.build_system.cxxflags += ['-DP2P']
            p2p = True
        else:
            p2p = False

        self.num_tasks = 0
        self.num_tasks_per_node = 1
        self.modules = ['cuda']

        # Gpus per node on each partition.
        self.partition_num_gpus_per_node = {
            'tsa:cn':         8,
            'ault:amda100':   4,
            'ault:amdv100':   2,
            'ault:intelv100':   4,
        }

        self.sanity_patterns = self.do_sanity_check()
        self.perf_patterns = {
            'bw': sn.min(sn.extractall(
                r'^[^,]*\[[^\]]*\]\s+GPU\s+\d+\s+(\s*\d+.\d+\s)+',
                self.stdout, 1, float))
        }

        if p2p:
            self.reference = {
                'tsa:cn': {
                    'bw':   (172.5, -0.05, None, 'GB/s'),
                },
                'arola:cn': {
                    'bw':   (172.5, -0.05, None, 'GB/s'),
                },
                'ault:amda100': {
                    'bw':   (282.07, -0.1, None, 'GB/s'),
                },
                'ault:amdv100': {
                    'bw':   (5.7, -0.1, None, 'GB/s'),
                },
                'ault:intelv100': {
                    'bw':   (31.0, -0.1, None, 'GB/s'),
                }
            }
        else:
            self.reference = {
                'tsa:cn': {
                    'bw': (79.6, -0.05, None, 'GB/s'),
                },
                'arola:cn': {
                    'bw': (79.6, -0.05, None, 'GB/s'),
                },
                'ault:amda100': {
                    'bw': (54.13, -0.1, None, 'GB/s'),
                },
                'ault:amdv100': {
                    'bw': (7.5, -0.1, None, 'GB/s'),
                },
                'ault:intelv100': {
                    'bw': (33.6, -0.1, None, 'GB/s'),
                }
            }

        self.tags = {'diagnostic', 'benchmark', 'mch'}
        self.maintainers = ['JO']

    @rfm.run_before('run')
    def set_num_gpus_per_node(self):
        self.num_gpus_per_node = self.partition_num_gpus_per_node.get(
            self.current_partition.fullname, 1)

    @rfm.run_before('compile')
    def set_nvidia_sm_arch(self):
        nvidia_sm = '60'
        if self.current_system.name in ['arolla', 'tsa', 'ault']:
            nvidia_sm = '70'

        if self.current_partition.fullname == 'ault:amda100':
            nvidia_sm = '80'

        self.build_system.cxxflags += [f'-arch=sm_{nvidia_sm}']

    @sn.sanity_function
    def do_sanity_check(self):
        node_names = set(sn.extractall(
            r'^\s*\[([^,]{1,20})\]\s*Found %s device\(s\).'
            % self.num_gpus_per_node, self.stdout, 1
        ))
        sn.evaluate(sn.assert_eq(
            self.job.num_tasks, len(node_names),
            msg='requested {0} node(s), got {1} (nodelist: %s)' %
            ','.join(sorted(node_names))))
        good_nodes = set(sn.extractall(
            r'^\s*\[([^,]{1,20})\]\s*Test Result\s*=\s*PASS',
            self.stdout, 1
        ))
        sn.evaluate(sn.assert_eq(
            node_names, good_nodes,
            msg='check failed on the following node(s): %s' %
            ','.join(sorted(node_names - good_nodes)))
        )

        return True
