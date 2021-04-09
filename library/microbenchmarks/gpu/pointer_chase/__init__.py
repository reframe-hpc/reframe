# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe.utility.sanity as sn
import reframe as rfm


__all__ = ['BuildGpuPChase', 'RunGpuPChaseSingle', 'RunGpuPChaseP2P']


class BuildGpuPChaseBase(rfm.CompileOnlyRegressionTest, pin_prefix=True):
    ''' Base class to build the pointer chase executable.

    Derived classes must define the variable `gpu_build` to indicate if the
    test should be build using cuda or hip.

    The name of the resulting executable is `pChase.x`.
    '''

    num_tasks = 1
    build_system = 'Make'
    postbuild_cmds = ['ls .']
    num_tasks_per_node = 1
    exclusive_access = True
    maintainers = ['JO', 'SK']
    tags = {'benchmark'}

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
                self.build_system.cxxflags = [f'-arch=compute_{self.gpu_arch}',
                                              f'-code=sm_{self.gpu_arch}']
        elif self.gpu_build == 'hip':
            self.build_system.makefile = 'makefile.hip'
            if self.gpu_arch:
                self.build_system.cxxflags = [
                    f'--amdgpu-target={self.gpu_arch}'
                ]
        else:
            raise ValueError('unknown gpu_build option')

    @rfm.run_before('sanity')
    def set_sanity(self):
        self.sanity_patterns = sn.assert_found(r'pChase.x', self.stdout)


class RunGpuPChaseBase(rfm.RunOnlyRegressionTest, pin_prefix=True):
    '''Base RunOnly class for the gpu pointer chase test.

    This runs the pointer chase algo on the linked list from the code compiled
    in the executable from the test above. The list is fully customisable
    through the command line, so the number of nodes, and the stride size for
    each jump will determine where the memory hits occur. This stride is set to
    32 node lengths (a node is 8 Bytes) to ensure that there is only a single
    node per cache line. The number of node jumps is set relatively large to
    ensure that the background effects are averaged out.

    Derived tests MUST set the number of list nodes, the executable and the
    number of gpus per compute node.
    '''

    # Linked list length
    num_list_nodes = variable(int)

    # Use a large stride to ensure there's only a single node per cache line
    stride = variable(int, value=32)  # (128 Bytes)

    # Set a large number of node jumps to smooth out spurious effects
    num_node_jumps = variable(int, value=400000)

    # Mark the required variables
    executable = required
    num_gpus_per_node = required

    maintainers = ['JO', 'SK']
    tags = {'benchmark'}

    @rfm.run_before('run')
    def set_exec_opts(self):
        self.executable_opts += [
            f'--stride {self.stride}',
            f'--nodes {self.num_list_nodes}',
            f'--num-jumps {self.num_node_jumps}'
        ]

    @sn.sanity_function
    def do_sanity_check(self):
        # Check that every node has the right number of GPUs
        # Store this nodes in case they're used later by the perf functions.
        self.my_nodes = set(sn.extractall(
            rf'^\s*\[([^\]]*)\]\s*Found {self.num_gpus_per_node} device\(s\).',
            self.stdout, 1))

        # Check that every node has made it to the end.
        nodes_at_end = len(set(sn.extractall(
            r'^\s*\[([^\]]*)\]\s*Pointer chase complete.',
            self.stdout, 1)))
        return sn.evaluate(sn.assert_eq(
            sn.assert_eq(self.job.num_tasks, len(self.my_nodes)),
            sn.assert_eq(self.job.num_tasks, nodes_at_end)))

    @rfm.run_before('sanity')
    def set_sanity(self):
        self.sanity_patterns = self.do_sanity_check()


class RunGpuPChaseSingle(RunGpuPChaseBase):
    '''Base class for intra-GPU latency tests.'''

    @rfm.run_before('performance')
    def set_performance_patterns(self):
        self.perf_patterns = {
            'average_latency': sn.max(sn.extractall(
                r'^\s*\[[^\]]*\]\s* On device \d+, '
                r'the chase took on average (\d+) '
                r'cycles per node jump.', self.stdout, 1, int)
            ),
        }


class RunGpuPChaseP2P(RunGpuPChaseBase):
    '''Base class for inter-GPU (P2P) latency tests.'''

    executable_opts = ['--multi-gpu']

    @sn.sanity_function
    def average_P2P_latency(self):
        '''
        Extract the average P2P latency. Note that the pChase code
        returns a table with the cummulative latency for all P2P
        list traversals, and the last column of this table has the max
        values for each device.
        '''
        return int(sn.evaluate(
            sn.max(sn.extractall(
                r'^\s*\[[^\]]*\]\s*GPU\s*\d+\s+(\s*\d+.\s+)+',
                self.stdout, 1, int)
            )
        ))

    @rfm.run_before('performance')
    def set_performance_patterns(self):
        self.perf_patterns = {
            'average_latency': self.average_P2P_latency(),
        }
