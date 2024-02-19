# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe.utility.sanity as sn
import reframe as rfm


__all__ = ['BuildGpuPchase', 'RunGpuPchase', 'RunGpuPchaseD2D']


class BuildGpuPchase(rfm.CompileOnlyRegressionTest, pin_prefix=True):
    ''' Base class to build the pointer chase executable.

    The test sources can be compiled for both CUDA and HIP. This is set with
    the `gpu_build` variable, which must be set by a derived class to either
    'cuda' or 'hip'. This source code can also be compiled for a specific
    device architecture by setting the ``gpu_arch`` variable to an AMD or
    NVIDIA supported architecture code.

    The name of the resulting executable is ``pChase.x``.
    '''

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

    num_tasks = 1
    build_system = 'Make'
    postbuild_cmds = ['ls .']
    num_tasks_per_node = 1
    exclusive_access = True
    maintainers = ['JO', 'SK']
    tags = {'benchmark'}

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

    @sanity_function
    def assert_exec_present(self):
        '''Assert that the executable is present.'''

        return sn.assert_found(r'pChase.x', self.stdout)


class RunGpuPchaseBase(rfm.RunOnlyRegressionTest, pin_prefix=True):
    '''Base RunOnly class for the gpu pointer chase test.

    This runs the pointer chase algo on the linked list from the code compiled
    in the executable from the test above. The list is fully customisable
    through the command line, so the number of nodes, and the stride size for
    each jump will determine where the memory hits occur. This stride is set to
    32 node lengths (a node is 8 Bytes) to ensure that there is only a single
    node per cache line. The number of node jumps is set relatively large to
    ensure that the background effects are averaged out.

    Derived tests must set the number of list nodes, the executable and the
    number of gpus per compute node.
    '''

    #: Variable that sets the size of the linked list.
    #:
    #: :default:``required``
    num_list_nodes = variable(int)

    #: Variable to set the stride (in mumber of nodes) amongst nodes in the
    #: linked list. We Use a large stride to ensure there's only a single
    #: node per cache line.
    #:
    #: :default:``32``
    stride = variable(int, value=32)  # (128 Bytes)

    #: Variable to set the total number of node jumps on the list traversal.
    #: We use a relatively large number of jumps to smooth out potential
    #: spurious effects.
    #:
    #: :default:``400000``
    num_node_jumps = variable(int, value=400000)

    # Mark the required variables
    executable = required
    num_gpus_per_node = required

    maintainers = ['JO', 'SK']
    tags = {'benchmark'}

    @run_before('run')
    def set_exec_opts(self):
        '''Set the list travesal options as executable args.'''

        self.executable_opts += [
            f'--stride {self.stride}',
            f'--nodes {self.num_list_nodes}',
            f'--num-jumps {self.num_node_jumps}'
        ]

    @sanity_function
    def assert_correct_num_gpus_per_node(self):
        '''Check that every node has the right number of GPUs.'''

        my_nodes = set(sn.extractall(
            rf'^\s*\[([^\]]*)\]\s*Found {self.num_gpus_per_node} device\(s\).',
            self.stdout, 1))

        # Check that every node has made it to the end.
        nodes_at_end = len(set(sn.extractall(
            r'^\s*\[([^\]]*)\]\s*Pointer chase complete.',
            self.stdout, 1)))
        return sn.assert_eq(
            sn.assert_eq(self.job.num_tasks, sn.count(my_nodes)),
            sn.assert_eq(self.job.num_tasks, nodes_at_end)
        )


class RunGpuPchase(RunGpuPchaseBase):
    '''Base class for intra-GPU latency tests.

    Derived classes must set the dependency with respect to a derived class
    from :class:`BuildGpuPchase`.
    '''

    @run_before('performance')
    def set_performance_patterns(self):
        self.perf_patterns = {
            'average_latency': sn.max(sn.extractall(
                r'^\s*\[[^\]]*\]\s* On device \d+, '
                r'the chase took on average (\d+) '
                r'cycles per node jump.', self.stdout, 1, int)
            ),
        }


class RunGpuPchaseD2D(RunGpuPchaseBase):
    '''Base class for inter-device (D2D) latency tests.

    Derived classes must set the dependency with respect to a derived class
    from :class:`BuildGpuPchase`.
    '''

    executable_opts = ['--multi-gpu']

    @deferrable
    def average_D2D_latency(self):
        '''Extract the average D2D latency.

        The pChase code returns a table with the cummulative latency for all
        D2D list traversals, and the last column of this table has the max
        values for each device.
        '''
        return sn.max(sn.extractall(
            r'^\s*\[[^\]]*\]\s*GPU\s*\d+\s+(\s*\d+.\s+)+',
            self.stdout, 1, int
        ))

    @run_before('performance')
    def set_performance_patterns(self):
        self.perf_patterns = {
            'average_latency': self.average_D2D_latency(),
        }
