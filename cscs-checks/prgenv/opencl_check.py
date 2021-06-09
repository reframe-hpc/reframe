# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.osext as osext
import reframe.utility.sanity as sn


@rfm.simple_test
class OpenCLCheck(rfm.RegressionTest):
    def __init__(self):
        self.maintainers = ['TM', 'SK']
        self.tags = {'production', 'craype'}

        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-pgi']
        self.modules = ['craype-accel-nvidia60']
        self.build_system = 'Make'
        self.sourcesdir = 'src/opencl'
        self.num_gpus_per_node = 1
        self.executable = 'vecAdd'

        self.sanity_patterns = sn.assert_found('SUCCESS', self.stdout)

    @run_before('compile')
    def setflags(self):
        if self.current_environ.name == 'PrgEnv-pgi':
            self.build_system.cflags = ['-mmmx']

    @run_before('compile')
    def cdt2006_pgi_workaround(self):
        cdt = osext.cray_cdt_version()
        if not cdt:
            return

        if (self.current_environ.name == 'PrgEnv-pgi' and cdt == '20.08'):
            self.variables.update({'CUDA_HOME': '$CUDATOOLKIT_HOME'})
