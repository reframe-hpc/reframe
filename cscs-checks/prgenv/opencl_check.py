# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.osext as osext
import reframe.utility.sanity as sn


@rfm.simple_test
class OpenCLCheck(rfm.RegressionTest):
    valid_systems = ['daint:gpu', 'dom:gpu']
    valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-pgi',
                           'PrgEnv-nvidia']
    build_system = 'Make'
    sourcesdir = 'src/opencl'
    num_gpus_per_node = 1
    executable = 'vecAdd'
    maintainers = ['TM', 'SK']
    tags = {'production', 'craype'}

    @run_after('setup')
    def setup_nvidia(self):
        if self.current_environ.name == 'PrgEnv-nvidia':
            # This is used by the Makefile for the OpenCL headers
            self.variables.update(
                {'CUDATOOLKIT_HOME': '$CRAY_NVIDIA_PREFIX/cuda'})
        else:
            self.modules = ['craype-accel-nvidia60']

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

    @run_before('sanity')
    def set_sanity(self):
        self.sanity_patterns = sn.assert_found('SUCCESS', self.stdout)
