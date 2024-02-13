# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class MemoryLimitTest(rfm.RegressionTest):
    valid_systems = ['daint:gpu', 'daint:mc']
    valid_prog_environs = ['gnu']
    sourcepath = 'eatmemory.c'
    executable_opts = ['2000M']

    @run_before('run')
    def set_memory_limit(self):
        self.job.options = ['--mem=1000']

    @sanity_function
    def validate_test(self):
        return sn.assert_found(
            r'(exceeded memory limit)|(Out Of Memory)', self.stderr
        )


@rfm.simple_test
class MemoryLimitWithResourcesTest(rfm.RegressionTest):
    valid_systems = ['daint:gpu', 'daint:mc']
    valid_prog_environs = ['gnu']
    sourcepath = 'eatmemory.c'
    executable_opts = ['2000M']
    extra_resources = {
        'memory': {'size': '1000'}
    }

    @sanity_function
    def validate_test(self):
        return sn.assert_found(
            r'(exceeded memory limit)|(Out Of Memory)', self.stderr
        )
