# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class MemoryLimitTest(rfm.RegressionTest):
    def __init__(self):
        self.valid_systems = ['daint:gpu', 'daint:mc']
        self.valid_prog_environs = ['gnu']
        self.sourcepath = 'eatmemory.c'
        self.executable_opts = ['4000M']
        self.sanity_patterns = sn.assert_found(
            r'(exceeded memory limit)|(Out Of Memory)', self.stderr
        )

    @rfm.run_before('run')
    def set_memory_limit(self):
        self.job.options = ['--mem=2000']


@rfm.simple_test
class MemoryLimitWithResourcesTest(rfm.RegressionTest):
    def __init__(self):
        self.valid_systems = ['daint:gpu', 'daint:mc']
        self.valid_prog_environs = ['gnu']
        self.sourcepath = 'eatmemory.c'
        self.executable_opts = ['4000M']
        self.sanity_patterns = sn.assert_found(
            r'(exceeded memory limit)|(Out Of Memory)', self.stderr
        )
        self.extra_resources = {
            'memory': {'size': '2000'}
        }
