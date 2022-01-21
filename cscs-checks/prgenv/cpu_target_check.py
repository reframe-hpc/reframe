# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class CrayCPUTargetTest(rfm.RunOnlyRegressionTest):
    descr = 'Checks whether CRAY_CPU_TARGET is set'
    valid_systems = ['daint:login', 'dom:login', 'eiger:login',
                     'pilatus:login']
    valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-gnu', 'PrgEnv-intel',
                           'PrgEnv-pgi', 'PrgEnv-nvidia']
    sourcesdir = None
    executable = 'echo CRAY_CPU_TARGET=$CRAY_CPU_TARGET'
    maintainers = ['TM', 'LM']
    tags = {'production', 'maintenance', 'craype'}

    @run_before('sanity')
    def set_sanity(self):
        self.sanity_patterns = sn.assert_found(r'CRAY_CPU_TARGET=\S+',
                                               self.stdout)
