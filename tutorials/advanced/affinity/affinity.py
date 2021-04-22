# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class AffinityTest(rfm.RegressionTest):
    valid_systems = ['daint:gpu', 'daint:mc']
    valid_prog_environs = ['*']
    sourcesdir = 'https://github.com/vkarak/affinity.git'
    build_system = 'Make'
    build_system.options = ['OPENMP=1']
    executable = './affinity'

    @rfm.run_before('run')
    def set_cpu_binding(self):
        self.job.launcher.options = ['--cpu-bind=cores']

    @rfm.run_before('sanity')
    def set_sanity_patterns(self):
        self.sanity_patterns = sn.assert_found(r'CPU affinity', self.stdout)
