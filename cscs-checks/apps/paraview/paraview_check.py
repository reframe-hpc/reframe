# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class ParaViewCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc']
        self.valid_prog_environs = ['builtin']
        self.num_tasks = 12
        self.num_tasks_per_node = 12
        self.modules = ['ParaView']

        self.time_limit = '1m'
        self.executable = 'pvbatch'
        self.executable_opts = ['coloredSphere.py']

        self.maintainers = ['JF']
        self.tags = {'scs', 'production'}

    @rfm.run_before('sanity')
    def set_sanity(self):
        if self.current_partition.name == 'mc':
            self.sanity_patterns = sn.all([
                sn.assert_found('Vendor:   VMware, Inc.', self.stdout),
                sn.assert_found('Renderer: llvmpipe', self.stdout)
            ])
        elif self.current_partition.name == 'gpu':
            self.sanity_patterns = sn.all([
                sn.assert_found('Vendor:   NVIDIA Corporation', self.stdout),
                sn.assert_found('Renderer: Tesla P100', self.stdout)
            ])
