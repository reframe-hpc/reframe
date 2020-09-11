# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.os_ext as os_ext
import reframe.utility.sanity as sn


@rfm.simple_test
class OpenCLCheck(rfm.RegressionTest):
    def __init__(self):
        self.maintainers = ['TM', 'SK']
        self.tags = {'production', 'craype'}

        self.valid_systems = ['daint:gpu', 'dom:gpu', 'tiger:gpu']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-pgi']
        self.modules = ['craype-accel-nvidia60']
        self.build_system = 'Make'
        self.sourcesdir += os.path.join('opencl')
        self.num_gpus_per_node = 1
        self.executable = 'vecAdd_opencl'

        self.sanity_patterns = sn.assert_found('SUCCESS', self.stdout)

    @rfm.run_before('compile')
    def setflags(self):
        if self.current_environ.name == 'PrgEnv-pgi':
            self.build_system.cflags = ['-mmmx']

    @rfm.run_before('compile')
    def cdt2006_pgi_workaround(self):
        cdt = os_ext.cray_cdt_version()
        if not cdt:
            return

        if (self.current_environ.name == 'PrgEnv-pgi' and cdt == '20.08'):
            self.variables.update({'CUDA_HOME': '$CUDATOOLKIT_HOME'})
