# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import os

import reframe.utility.sanity as sn
import reframe as rfm


@rfm.required_version('>=2.16-dev0')
@rfm.parameterized_test(['peerAccess'], ['noPeerAccess'])
class P2pBandwidthCheck(rfm.RegressionTest):
    def __init__(self, peerAccess):
        self.valid_systems = ['tsa:cn', 'ault:amdv100']
        self.valid_prog_environs = ['PrgEnv-gnu']
        if self.current_system.name in ['arolla', 'tsa']:
            self.valid_prog_environs = ['PrgEnv-gnu-nompi']
            self.exclusive_access = True

        self.build_system = 'SingleSource'
        self.sourcepath = 'p2p_bandwidth.cu'
        self.executable = 'p2p_bandwidth.x'
        self.exclusive_access = True
        # Set nvcc flags
        nvidia_sm = '70'

        # Perform a single bandwidth test with a buffer size of 1024MB
        copy_size = 1073741824

        self.build_system.cxxflags = ['-I.', '-m64', '-arch=sm_%s' % nvidia_sm,
                                      '-std=c++11', '-lnvidia-ml',
                                      '-DCOPY=%d' % copy_size]
        if (peerAccess == 'peerAccess'):
            self.build_system.cxxflags += ['-DP2P']
            p2p = True
        else:
            p2p = False

        self.num_tasks = 0
        self.modules = ['cuda']

        # Gpus per node on each partition.
        self.partition_num_gpus_per_node = {
            'tsa:cn':         8,
            'ault:amdv100':   2,
            'ault:intelv100':   4,
        }

        self.sanity_patterns = self.do_sanity_check(p2p)
        self.perf_patterns = {}
        self.reference = {}
        self.__bwref = {
            'tsa:cn:p2p':  (172.5, -0.05, None, 'GB/s'),
            'tsa:cn:nop2p':  (79.6, -0.05, None, 'GB/s'),
            'ault:amdv100:p2p':  (5.7, -0.1, None, 'GB/s'),
            'ault:amdv100:nop2p':  (7.5, -0.1, None, 'GB/s'),
            'ault:intelv100:p2p':  (31.0, -0.1, None, 'GB/s'),
            'ault:intelv100:nop2p':  (33.6, -0.1, None, 'GB/s'),

        }

        self.tags = {'diagnostic', 'benchmark', 'mch'}
        self.maintainers = ['JO']

    @rfm.run_before('run')
    def set_num_gpus_per_node(self):
        if self.current_partition.fullname in self.partition_num_gpus_per_node:
            self.num_gpus_per_node = self.partition_num_gpus_per_node.get(
                self.current_partition.fullname)
        else:
            self.num_gpus_per_node = 1

    @sn.sanity_function
    def do_sanity_check(self, p2p):
        node_names = set(sn.extractall(
            r'^\s*\[([^,]*)\]\s*Found\s+%s\s+device\(s\).' % self.num_gpus_per_node,
            self.stdout, 1
        ))

        sn.evaluate(sn.assert_eq(
            self.job.num_tasks, len(node_names),
            msg='requested {0} node(s), got {1} (nodelist: %s)' %
            ','.join(sorted(node_names))))

        good_nodes = set(sn.extractall(
            r'^[^,]*\[([^,]*)\]\s*Test Result\s*=\s*PASS',
            self.stdout, 1
        ))

        sn.evaluate(sn.assert_eq(
            node_names, good_nodes,
            msg='check failed on the following node(s): %s' %
            ','.join(sorted(node_names - good_nodes)))
        )

        if (p2p):
            xfer = 'p2p'
        else:
            xfer = 'nop2p'

        for nodename in node_names:
            for devno in range(self.num_gpus_per_node):
                perfvar = 'bw_%s_%s_gpu_%s' % (xfer, nodename, devno)
                self.perf_patterns[perfvar] = sn.extractsingle(
                    r'^[^,]*\[%s\]\s+GPU\s+%d\s+(\s*\d+.\d+\s)+' % (
                        nodename, devno),
                    self.stdout, 1, float, 0
                )
                partname = self.current_partition.fullname
                refkey = '%s:%s' % (partname, perfvar)
                bwkey = '%s:%s' % (partname, xfer)
                with contextlib.suppress(KeyError):
                    self.reference[refkey] = self.__bwref[bwkey]

        return True
