# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm

from hpctestlib.apps.icon.base_check import RRTMGPTest_Base

@rfm.simple_test
class RRTMGPTest(RRTMGPTest_Base):
    maintainers = ['WS', 'RS']
    valid_systems = ['dom:gpu', 'daint:gpu']
    valid_prog_environs = ['PrgEnv-pgi']
    modules = ['craype-accel-nvidia60', 'cray-netcdf']

    @run_after('init')
    def set_soursedir(self):
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'RRTMGP')

    @run_after('init')
    def set_prebuild_commands(self):
        self.prebuild_cmds = ['cp build/Makefile.conf.dom build/Makefile.conf']
