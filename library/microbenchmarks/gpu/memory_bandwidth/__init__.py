# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe.utility.sanity as sn
import reframe as rfm


__all__ = ['GpuBandwidthSingle', 'GpuBandwidthMulti']

class GpuBandwidthBase(rfm.RegressionTest, pin_prefix=True):
    ''' Base class to the gpu bandwidth test.'''

    # Default copy variables
    copy_size = variable(int, value=1073741824)
    num_copies = variable(int, value=20)

    build_system = 'Make'
    executable = 'memory_bandwidth.x'
    num_tasks = 0
    num_tasks_per_node = 1
    exclusive_access = True
    tags = {'benchmark'}
    maintainers = ['AJ', 'SK']

    # GPU build options
    # The build can either be 'cuda' or 'hip'. This variable is required.
    # However, specifying the device's architecture is entirely optional.
    gpu_build = variable(str)
    gpu_arch = variable(str, type(None), value=None)

    @rfm.run_before('compile')
    def set_gpu_build(self):
        '''This hook requires the `gpu_build` variable to be set.

        Both the cuda and hip options are supported by the test sources.
        '''
        if self.gpu_build == 'cuda':
            self.build_system.makefile = 'makefile.cuda'
            if self.gpu_arch:
                self.build_system.cxxflags += [f'-arch=compute_{self.gpu_arch}',
                                              f'-code=sm_{self.gpu_arch}']
        elif self.gpu_build == 'hip':
            self.build_system.makefile = 'makefile.hip'
            if self.gpu_arch:
                self.build_system.cxxflags += [
                    f'--amdgpu-target={self.gpu_arch}'
                ]
        else:
            raise ValueError('unknown gpu_build option')

    @rfm.run_before('run')
    def set_exec_opts(self):
        self.executable_opts += [
            f'--size {self.copy_size}',
            f'--copies {self.num_copies}',
        ]


class GpuBandwidthSingle(GpuBandwidthBase):
    '''GPU memory andwidth benchmark.

    Evaluates the individual host-device, device-host and device-device
    bandwidth for all the GPUs on each node.
    '''

    @rfm.run_before('sanity')
    def set_sanity_patterns(self):
        self.sanity_patterns = self.do_sanity_check()

    @rfm.run_before('performance')
    def set_perf_patterns(self):
       self.perf_patterns = {
           'h2d': sn.min(sn.extractall(self._xfer_pattern('h2d'),
                                       self.stdout, 1, float)),
           'd2h': sn.min(sn.extractall(self._xfer_pattern('d2h'),
                                       self.stdout, 1, float)),
           'd2d': sn.min(sn.extractall(self._xfer_pattern('d2d'),
                                       self.stdout, 1, float)),
       }

    def _xfer_pattern(self, xfer_kind):
        '''generates search pattern for performance analysis'''
        if xfer_kind == 'h2d':
            direction = 'Host to device'
        elif xfer_kind == 'd2h':
            direction = 'Device to host'
        else:
            direction = 'Device to device'

        # Extract the bandwidth corresponding to the right node, transfer and
        # device.
        return (rf'^[^,]*\[[^,]*\]\s*{direction}\s*bandwidth on device'
                r' \d+ is \s*(\S+)\s*GB/s.')

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


class GpuBandwidthMulti(GpuBandwidthBase):
    '''Multi-GPU memory bandwidth benchmark.

    Evaluates the copy bandwidth amongst all devices in a compute node.
    This test assesses the bandwidth with and without direct peer memory
    acess.
    '''

    p2p = parameter([True, False])

    @rfm.run_before('run')
    def set_exec_opts(self):
        self.executable_opts += ['--multi-gpu']
        if self.p2p:
            self.executable_opts += ['--p2p']

    @rfm.run_before('sanity')
    def set_sanity_patterns(self):
        self.sanity_patterns = self.do_sanity_check()

    @rfm.run_before('performance')
    def set_perf_patterns(self):
        self.perf_patterns = {
            'bw': sn.min(sn.extractall(
                r'^[^,]*\[[^\]]*\]\s+GPU\s+\d+\s+(\s*\d+.\d+\s)+',
                self.stdout, 1, float))
        }

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
