# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import os

import reframe.utility.sanity as sn
import reframe as rfm


@rfm.required_version('>=2.16-dev0')
@rfm.simple_test
class GpuBandwidthCheck(rfm.RegressionTest):
    def __init__(self):
        self.valid_systems = ['kesch:cn', 'daint:gpu', 'dom:gpu', 'tiger:gpu',
                              'arolla:cn', 'tsa:cn']
        self.valid_prog_environs = ['PrgEnv-gnu']
        if self.current_system.name in ['arolla', 'kesch', 'tsa']:
            self.valid_prog_environs = ['PrgEnv-gnu-nompi']
            self.exclusive_access = True

        self.build_system = 'SingleSource'
        self.sourcepath = 'memory_bandwidth.cu'
        self.executable = 'memory_bandwidth.x'

        # Set nvcc flags
        nvidia_sm = '60'
        if self.current_system.name == 'kesch':
            nvidia_sm = '37'
        elif self.current_system.name in ['arolla', 'tsa']:
            nvidia_sm = '70'

        # Perform a single bandwidth test with a buffer size of 1024MB
        self.copy_size = 1073741824
 
        self.build_system.cxxflags = ['-I.', '-m64', '-arch=sm_%s' % nvidia_sm, '-std=c++11', '-DCOPY=%d' % self.copy_size]
        self.num_tasks = 0
        self.num_tasks_per_node = 1
        if self.current_system.name in ['daint', 'dom', 'tiger']:
            self.modules = ['craype-accel-nvidia60']
            self.num_gpus_per_node = 1
        elif self.current_system.name == 'kesch':
            self.modules = ['cudatoolkit/8.0.61']
            self.num_gpus_per_node = 8
        elif self.current_system.name in ['arolla', 'tsa']:
            self.modules = ['cuda/10.1.243']
            self.num_gpus_per_node = 8

        # perf_patterns and reference will be set by the sanity check function
        self.sanity_patterns = self.do_sanity_check()
        self.perf_patterns = {}
        self.reference = {}
        self.__bwref = {
            'daint:gpu:h2d':  (11881, -0.1, None, 'MB/s'),
            'daint:gpu:d2h':  (12571, -0.1, None, 'MB/s'),
            'daint:gpu:d2d': (499000, -0.1, None, 'MB/s'),
            'dom:gpu:h2d':  (11881, -0.1, None, 'MB/s'),
            'dom:gpu:d2h':  (12571, -0.1, None, 'MB/s'),
            'dom:gpu:d2d': (499000, -0.1, None, 'MB/s'),
            'kesch:cn:h2d':   (7583, -0.1, None, 'MB/s'),
            'kesch:cn:d2h':   (7584, -0.1, None, 'MB/s'),
            'kesch:cn:d2d': (137408, -0.1, None, 'MB/s'),
        }
        self.tags = {'diagnostic', 'benchmark', 'mch',
                     'craype', 'external-resources'}
        self.maintainers = ['AJ', 'SK']

    def _xfer_pattern(self, xfer_kind, devno, nodename):
        '''generates search pattern for performance analysis'''
        if xfer_kind == 'h2d':
            direction = 'Host to device'
        elif xfer_kind == 'd2h':
            direction = 'Device to host'
        else:
            direction = 'Device to device'

        # Extract the bandwidth corresponding to the right node, transfer and device.
        return (r'^[^,]*\[[^,]*\]\s*%s\s*bandwidth on device %d is \s*(\S+)\s*Mb/s.' % 
                (direction,devno))

    @sn.sanity_function
    def do_sanity_check(self):
        node_names = set(sn.extractall(
            r'^\s*\[([^,]*)\]\s*Found %s device\(s\).' % self.num_gpus_per_node,
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

        # Sanity is fine, fill in the perf. patterns based on the exact node id
        for nodename in node_names:
            for xfer_kind in ('h2d', 'd2h', 'd2d'):
                for devno in range(self.num_gpus_per_node):
                    perfvar = 'bw_%s_%s_gpu_%s' % (xfer_kind, nodename, devno)
                    self.perf_patterns[perfvar] = sn.extractsingle(
                        self._xfer_pattern(xfer_kind, devno, nodename),
                        self.stdout, 1, float, 0
                    )
                    partname = self.current_partition.fullname
                    refkey = '%s:%s' % (partname, perfvar)
                    bwkey = '%s:%s' % (partname, xfer_kind)
                    with contextlib.suppress(KeyError):
                        self.reference[refkey] = self.__bwref[bwkey]

        return True
