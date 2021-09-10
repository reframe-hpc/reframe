# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
from hpctestlib.apps.paraview.base_check import ParaView_BaseCheck


@rfm.simple_test
class ParaViewCheckCSCS(ParaView_BaseCheck):
    valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
                     'eiger:mc', 'pilatus:mc']
    num_tasks = 12
    num_tasks_per_node = 12
    modules = ['ParaView']
    maintainers = ['JF', 'TM']
    tags = {'scs', 'production'}
    mc_vendor = 'VMware, Inc.'
    mc_renderer = 'llvmpipe'
    gpu_vendor = 'NVIDIA Corporation'
    gpu_renderer = 'Tesla P100'

    @run_after('init')
    def set_hierarchical_prgenvs(self):
        if self.current_system.name in ['eiger', 'pilatus']:
            self.valid_prog_environs = ['cpeCray']
        else:
            self.valid_prog_environs = ['builtin']
