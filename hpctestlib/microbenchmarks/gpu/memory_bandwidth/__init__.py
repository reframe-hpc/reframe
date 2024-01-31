# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe.utility.sanity as sn
import reframe as rfm


__all__ = ['GpuBandwidth', 'GpuBandwidthD2D']


class GpuBandwidthBase(rfm.RegressionTest, pin_prefix=True):
    '''Base class to the gpu bandwidth test.

    The test sources can be compiled for both CUDA and HIP. This is set with
    the ``gpu_build`` variable, which must be set by a derived class to either
    ``'cuda'`` or ``'hip'``. This source code can also be compiled for a
    specific device architecture by setting the ``gpu_arch`` variable to an
    AMD or NVIDIA supported architecture code.
    '''

    #: Set the build option to either ``'cuda'`` or ``'hip'``.
    #:
    #: :default: ``required``
    gpu_build = variable(str)

    #: Set the GPU architecture.
    #: This variable will be passed to the compiler to generate the
    #: arch-specific code.
    #:
    #: :default: ``None``
    gpu_arch = variable(str, type(None), value=None)

    #: Size of the array used to measure the bandwidth.
    #:
    #: :default:``1024**3``
    copy_size = variable(int, value=1073741824)

    #: Number of times each type of copies is performed.
    #: The returned bandiwdth values are averaged over this number of times.
    #:
    #: :default:``20``
    num_copies = variable(int, value=20)

    build_system = 'Make'
    executable = 'memory_bandwidth.x'
    num_tasks = required
    num_tasks_per_node = 1
    maintainers = ['AJ', 'SK']

    @run_before('compile')
    def set_gpu_build(self):
        '''Set the build options [pre-compile hook].

        This hook requires the ``gpu_build`` variable to be set.
        The supported options are ``'cuda'`` and ``'hip'``. See the
        vendor-specific docs for the supported options for the ``gpu_arch``
        variable.
        '''

        if self.gpu_build == 'cuda':
            self.build_system.makefile = 'makefile.cuda'
            if self.gpu_arch:
                self.build_system.cxxflags += [
                    f'-arch=compute_{self.gpu_arch}',
                    f'-code=sm_{self.gpu_arch}'
                ]
        elif self.gpu_build == 'hip':
            self.build_system.makefile = 'makefile.hip'
            if self.gpu_arch:
                self.build_system.cxxflags += [
                    f'--amdgpu-target={self.gpu_arch}'
                ]
        else:
            raise ValueError('unknown gpu_build option')

    @run_before('run')
    def set_exec_opts(self):
        '''Pass the copy size and number of copies as executable args.'''

        self.executable_opts += [
            f'--size {self.copy_size}',
            f'--copies {self.num_copies}',
        ]

    @sanity_function
    def assert_successful_completion(self):
        '''Check that all nodes completed successfully.'''

        node_names = set(sn.extractall(
            r'^\s*\[([^\]]*)\]\s*Found %s device\(s\).'
            % self.num_gpus_per_node, self.stdout, 1
        ))
        req_nodes = sn.assert_eq(
            self.job.num_tasks, len(node_names),
            msg='requested {0} node(s), got {1} (nodelist: %s)' %
            ','.join(sorted(node_names)))
        good_nodes = set(sn.extractall(
            r'^\s*\[([^\]]*)\]\s*Test Result\s*=\s*PASS',
            self.stdout, 1
        ))
        failed_nodes = sn.assert_eq(
            node_names, good_nodes,
            msg='check failed on the following node(s): %s' %
            ','.join(sorted(node_names - good_nodes))
        )

        return sn.all([req_nodes, failed_nodes])


class GpuBandwidth(GpuBandwidthBase):
    '''GPU memory bandwidth benchmark.

    Evaluates the individual host-device, device-host and device-device
    bandwidth (in GB/s) for all the GPUs on each node.
    '''

    @run_before('performance')
    def set_perf_patterns(self):
        '''Set the performance patterns.

        These include host-device (h2d), device-host (d2h) and device=device
        (d2d) transfers.
        '''

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
        return (rf'^\[[^\]]*\]\s*{direction}\s*bandwidth on device'
                r' \d+ is \s*(\S+)\s*GB/s.')


class GpuBandwidthD2D(GpuBandwidthBase):
    '''Multi-GPU memory bandwidth benchmark.

    Evaluates the copy bandwidth (in GB/s) amongst all devices in a compute
    node. This test assesses the bandwidth with and without direct peer
    memory access (see the parameter `p2p`).
    '''

    #: Parameter to test the multi-gpu bandwidth with and without P2P
    #: memory access.
    #: This option is passed as an argument to the executable.
    p2p = parameter([True, False])

    @run_before('run')
    def extend_exec_opts(self):
        '''Add the multi-gpu related arguments to the executable options.'''
        self.executable_opts += ['--multi-gpu']
        if self.p2p:
            self.executable_opts += ['--p2p']

    @run_before('performance')
    def set_perf_patterns(self):
        '''Set the performance patterns.

        In addition to the individual transfer rates amongst devices, this test
        also reports the average bandwidth per device with all the other
        devices. Hence, the performance pattern will report the device with the
        lowest average copy bandwidth with all the other devices.
        '''
        self.perf_patterns = {
            'bw': sn.min(sn.extractall(
                r'^\[[^\]]*\]\s+GPU\s+\d+\s+(\s*\d+.\d+\s)+',
                self.stdout, 1, float))
        }
