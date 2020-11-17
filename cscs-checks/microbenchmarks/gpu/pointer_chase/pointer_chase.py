# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe.utility.sanity as sn
import reframe as rfm

import os


class Pchase:
    '''
    Public storage class to avoid writing the parameters below multiple times.
    '''
    valid_systems = ['ault:intelv100', 'ault:amdv100',
                     'ault:amda100', 'ault:amdvega']
    valid_prog_environs = ['PrgEnv-gnu']


@rfm.simple_test
class CompileGpuPointerChase(rfm.CompileOnlyRegressionTest):
    def __init__(self):
        self.valid_systems = Pchase.valid_systems
        self.valid_prog_environs = Pchase.valid_prog_environs
        self.exclusive_access = True
        self.build_system = 'Make'
        self.num_tasks = 0
        self.num_tasks_per_node = 1
        self.postbuild_cmds = ['ls .']
        self.sanity_patterns = sn.assert_found(r'pChase.x', self.stdout)
        self.maintainers = ['JO']

    @rfm.run_after('setup')
    def select_makefile(self):
        cp = self.current_partition.fullname
        if cp == 'ault:amdvega':
            self.prebuild_cmds = ['cp makefile.hip Makefile']
        else:
            self.prebuild_cmds = ['cp makefile.cuda Makefile']

    @rfm.run_before('compile')
    def set_gpu_arch(self):
        cp = self.current_partition.fullname

        # Deal with the NVIDIA options first
        nvidia_sm = None
        if cp[-4:] == 'v100':
            nvidia_sm = '70'
        elif cp[-4:] == 'a100':
            nvidia_sm = '80'

        if nvidia_sm:
            self.build_system.cxxflags += [f'-arch=sm_{nvidia_sm}']
            self.modules += ['cuda']

        # Deal with the AMD options
        amd_trgt = None
        if cp == 'ault:amdvega':
            amd_trgt = 'gfx908'

        if amd_trgt:
            self.build_system.cxxflags += [f'--amdgpu-target={amd_trgt}']
            self.modules += ['rocm']


class GpuPointerChaseBase(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.depends_on('CompileGpuPointerChase')
        self.valid_systems = Pchase.valid_systems
        self.valid_prog_environs = Pchase.valid_prog_environs
        self.num_tasks = 0
        self.num_tasks_per_node = 1
        self.exclusive_access = True
        self.sanity_patterns = self.do_sanity_check()
        self.maintainers = ['JO']

    @rfm.require_deps
    def set_executable(self, CompileGpuPointerChase):
        self.executable = os.path.join(
            CompileGpuPointerChase().stagedir, 'pChase.x')

    @rfm.run_before('run')
    def set_num_gpus_per_node(self):
        cp = self.current_partition.fullname
        if cp in {'ault:intelv100', 'ault:amda100'}:
            self.num_gpus_per_node = 4
        elif cp in {'ault:amdv100'}:
            self.num_gpus_per_node = 2
        elif cp in {'ault:amdvega'}:
            self.num_gpus_per_node = 3
        else:
            self.num_gpus_per_node = 1

    @sn.sanity_function
    def do_sanity_check(self):

        # Check that every node has the right number of GPUs
        healthy_nodes = len(set(sn.extractall(
            r'^\s*\[([^\]]*)\]\s*Found %d device\(s\).' % self.num_gpus_per_node,
            self.stdout, 1)))

        # Check that every node has made it to the end.
        nodes_at_end = len(set(sn.extractall(
            r'^\s*\[([^\]]*)\]\s*Pointer chase complete.',
            self.stdout, 1)))
        return sn.evaluate(sn.assert_eq(
            sn.assert_eq(self.job.num_tasks, healthy_nodes),
            sn.assert_eq(self.job.num_tasks, nodes_at_end)))


@rfm.parameterized_test([1], [2], [4], [4096])
class GpuPointerChaseSingle(GpuPointerChaseBase):
    def __init__(self, stride):
        super().__init__()
        self.executable_opts = ['--stride', f'{stride}']
        self.perf_patterns = {
            'average': sn.min(sn.extractall(r'^\s*\[[^\]]*\]\s* On device \d+, '
                                            r'the chase took on average (\d+) '
                                            r'cycles per node jump.',
                                            self.stdout, 1, int)),
        }

        if stride == 1:
            self.reference = {
                'ault:amda100': {
                    'average': (76, None, 0.1, 'clock cycles')
                },
                'ault:amdv100': {
                    'average': (77, None, 0.1, 'clock cycles')
                },
                'dom:gpu': {
                    'average': (143, None, 0.1, 'clock cycles')
                },
                'daint:gpu': {
                    'average': (143, None, 0.1, 'clock cycles')
                },
                'ault:amdvega': {
                    'average': (225, None, 0.1, 'clock cycles')
                },
            }
        elif stride == 2:
            self.reference = {
                'ault:amda100': {
                    'average': (116, None, 0.1, 'clock cycles')
                },
                'ault:amdv100': {
                    'average': (118, None, 0.1, 'clock cycles')
                },
                'dom:gpu': {
                    'average': (181, None, 0.1, 'clock cycles')
                },
                'daint:gpu': {
                    'average': (181, None, 0.1, 'clock cycles')
                },
                'ault:amdvega': {
                    'average': (300, None, 0.1, 'clock cycles')
                },
            }
        elif stride == 4:
            self.reference = {
                'ault:amda100': {
                    'average': (198, None, 0.1, 'clock cycles')
                },
                'ault:amdv100': {
                    'average': (200, None, 0.1, 'clock cycles')
                },
                'dom:gpu': {
                    'average': (260, None, 0.1, 'clock cycles')
                },
                'daint:gpu': {
                    'average': (260, None, 0.1, 'clock cycles')
                },
                'ault:amdvega': {
                    'average': (470, None, 0.1, 'clock cycles')
                },
            }
        elif stride == 4096:
            self.reference = {
                'ault:amda100': {
                    'average': (206, None, 0.1, 'clock cycles')
                },
                'ault:amdv100': {
                    'average': (220, None, 0.1, 'clock cycles')
                },
                'dom:gpu': {
                    'average': (260, None, 0.1, 'clock cycles')
                },
                'daint:gpu': {
                    'average': (260, None, 0.1, 'clock cycles')
                },
                'ault:amdvega': {
                    'average': (800, None, 0.1, 'clock cycles')
                },
            }
