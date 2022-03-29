# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class ParaViewCheck(rfm.RunOnlyRegressionTest):
    valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
                     'eiger:mc', 'pilatus:mc']
    valid_prog_environs = ['builtin']
    num_tasks = 12
    num_tasks_per_node = 12
    modules = ['ParaView']
    executable = 'pvbatch'
    executable_opts = ['coloredSphere.py']
    maintainers = ['JF', 'TM']
    tags = {'scs', 'production'}

    @run_after('init')
    def set_prgenv_alps(self):
        if self.current_system.name in {'eiger', 'pilatus'}:
            self.valid_prog_environs = ['cpeCray']

    @sanity_function
    def assert_vendor_renderer(self):
        if self.current_partition.name == 'mc':
            return sn.assert_found('Renderer: llvmpipe', self.stdout)
        elif self.current_partition.name == 'gpu':
            return sn.all([
                sn.assert_found('Vendor:   NVIDIA Corporation', self.stdout),
                sn.assert_found('Renderer: Tesla P100', self.stdout)
            ])
