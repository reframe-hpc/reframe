# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
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
    executable = './affinity'

    @run_before('compile')
    def set_build_system_options(self):
        self.build_system.options = ['OPENMP=1']

    @run_before('run')
    def set_cpu_binding(self):
        self.job.launcher.options = ['--cpu-bind=cores']

    @sanity_function
    def validate_test(self):
        return sn.assert_found(r'CPU affinity', self.stdout)
