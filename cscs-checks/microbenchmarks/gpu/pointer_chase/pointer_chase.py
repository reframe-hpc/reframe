# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe.utility.sanity as sn
import reframe as rfm


@rfm.simple_test
class GpuPointerChase(rfm.RegressionTest):
    def __init__(self):
        self.valid_systems = ['ault:intelv100', 'ault:amdv100',
                              'ault:amda100']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.exclusive_access = True
        self.pre_build_cmds = ['cp makefile.cuda Makefile']
        self.build_system = 'Make'
        self.executable = 'pChase.x'
        self.num_tasks = 0
        self.num_tasks_per_node = 1



    @rfm.run_before('compile')
    def set_gpu_arch(self):
        cp = self.current_partition.fullname
        if cp[-4:] == 'v100':
            nvidia_sm = '70'
        elif cp[-4:] == 'a100':
            nvidia_sm = '80'
        else
            nvidia_sm = None

        self.build_system.cxxflags += [f'-arch=sm_{nvidia_sm}']

    @rfm.run_before('run')
    def set_num_gpus_per_node(self):
        cp = self.current_partition.fullname
        if cp in {'ault:intelv100', 'ault:amda100'}:
            self.num_gpus_per_node = 4
        elif cp in {'ault:amdv100'}:
            self.num_gpus_per_node = 2
        else:
            self.num_gpus_per_node = 1
