# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Hooks specific to the CSCS GPU microbenchmark tests.
#


def set_gpu_arch(self):
    '''Set the compile options for the gpu microbenchmarks.'''

    cs = self.current_system.name
    cp = self.current_partition.fullname
    self.gpu_arch = None

    # Nvidia options
    self.gpu_build = 'cuda'
    if cs in {'dom', 'daint'}:
        self.gpu_arch = '60'
        if self.current_environ.name not in {'PrgEnv-nvidia'}:
            self.modules = ['craype-accel-nvidia60', 'cdt-cuda']

    elif cs in {'arola', 'tsa'}:
        self.gpu_arch = '70'
        self.modules = ['cuda/10.1.243']
    elif cs in {'ault'}:
        self.modules = ['cuda']
        if cp in {'ault:amdv100', 'ault:intelv100'}:
            self.gpu_arch = '70'
        elif cp in {'ault:amda100'}:
            self.gpu_arch = '80'

    # AMD options
    if cp in {'ault:amdvega'}:
        self.gpu_build = 'hip'
        self.modules = ['rocm']
        self.gpu_arch = 'gfx900,gfx906'


def set_num_gpus_per_node(self):
    '''Set the GPUs per node for the GPU microbenchmarks.'''

    cs = self.current_system.name
    cp = self.current_partition.fullname
    if cs in {'dom', 'daint'}:
        self.num_gpus_per_node = 1
    elif cs in {'arola', 'tsa'}:
        self.num_gpus_per_node = 8
    elif cp in {'ault:amda100', 'ault:intelv100'}:
        self.num_gpus_per_node = 4
    elif cp in {'ault:amdv100'}:
        self.num_gpus_per_node = 2
    elif cp in {'ault:amdvega'}:
        self.num_gpus_per_node = 3
    else:
        self.num_gpus_per_node = 1
