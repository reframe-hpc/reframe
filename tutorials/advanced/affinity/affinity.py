# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class AffinityTest(rfm.RegressionTest):
    def __init__(self):
        self.valid_systems = ['daint']
        self.valid_prog_environs = ['*']
        self.sourcesdir = 'https://github.com/vkarak/affinity.git'
        self.build_system = 'Make'
        self.build_system.options = ['OPENMP=1']
        self.executable = './affinity'
        self.sanity_patterns = sn.assert_found(r'CPU affinity', self.stdout)

    @rfm.run_before('run')
    def set_cpu_binding(self):
        self.job.launcher.options = ['--cpu-bind=cores']
