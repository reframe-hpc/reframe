# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe.utility.sanity as sn
import reframe as rfm


__all__ = ['GPU_bandwidth_single', 'GPU_bandwidth_multi']


class GPU_bandwidth_base(rfm.RegressionTest, pin_prefix=True):
    '''Base class to the gpu bandwidth test.'''

    #: Set the build option to either 'cuda' or 'hip'.
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

    @rfm.run_before('compile')
    def set_gpu_build(self):
        '''Set the build options.

        This hook requires the `gpu_build` variable to be set. Both 'cuda' and
        'hip' options are supported by the test sources.
        The supported options are 'cuda' and 'hip'. See the vendor-specific
        docs for the supported options for the ``gpu_arch`` variable.
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

    @rfm.run_before('run')
    def set_exec_opts(self):
        '''Pass the copy size and number of copies as executable args.'''

        self.executable_opts += [
            f'--size {self.copy_size}',
            f'--copies {self.num_copies}',
        ]

    @rfm.run_before('sanity')
    def set_sanity_patterns(self):
        self.sanity_patterns = self.do_sanity_check()

    @sn.sanity_function
    def do_sanity_check(self):
        '''Check that all nodes completed successfully.'''

        node_names = set(sn.extractall(
            r'^\s*\[([^\]]*)\]\s*Found %s device\(s\).'
            % self.num_gpus_per_node, self.stdout, 1
        ))
        sn.evaluate(sn.assert_eq(
            self.job.num_tasks, len(node_names),
            msg='requested {0} node(s), got {1} (nodelist: %s)' %
            ','.join(sorted(node_names))))
        good_nodes = set(sn.extractall(
            r'^\s*\[([^\]]*)\]\s*Test Result\s*=\s*PASS',
            self.stdout, 1
        ))
        sn.evaluate(sn.assert_eq(
            node_names, good_nodes,
            msg='check failed on the following node(s): %s' %
            ','.join(sorted(node_names - good_nodes)))
        )
        return True


class GPU_bandwidth_single(GPU_bandwidth_base):
    '''GPU memory bandwidth benchmark.

    Evaluates the individual host-device, device-host and device-device
    bandwidth for all the GPUs on each node.

    -- Sanity --
    Tests that the number of nodes and the number of devices per node matches
    the variables specified in the test.

    -- Performance --
    The performance patterns are:
     - h2d: Host to device bandwidth in GB/s.
     - d2h: Device to host bandwidth in GB/s.
     - d2d: Device to device bandwidth in GB/s.

    '''

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
        return (rf'^[^,]*\[[^\]]*\]\s*{direction}\s*bandwidth on device'
                r' \d+ is \s*(\S+)\s*GB/s.')


class GPU_bandwidth_multi(GPU_bandwidth_base):
    '''Multi-GPU memory bandwidth benchmark.

    Evaluates the copy bandwidth amongst all devices in a compute node.
    This test assesses the bandwidth with and without direct peer memory
    access (see the parameter `p2p`).

    -- Sanity --
    Tests that the number of nodes and the number of devices per node matches
    the variables specified in the test.

    -- Performance --
    The performance patterns are:
     - bw: The average bandwidth with all the other devices in the node.
    '''

    #: Parameter to test the multi-gpu bandwidth with and without P2P
    #: memory access.
    #: This option is passed as an argument to the executable.
    p2p = parameter([True, False])

    @rfm.run_before('run')
    def extend_exec_opts(self):
        '''Add the multi-gpu related arguments to the executable options.'''
        self.executable_opts += ['--multi-gpu']
        if self.p2p:
            self.executable_opts += ['--p2p']

    @rfm.run_before('performance')
    def set_perf_patterns(self):
        self.perf_patterns = {
            'bw': sn.min(sn.extractall(
                r'^[^,]*\[[^\]]*\]\s+GPU\s+\d+\s+(\s*\d+.\d+\s)+',
                self.stdout, 1, float))
        }
