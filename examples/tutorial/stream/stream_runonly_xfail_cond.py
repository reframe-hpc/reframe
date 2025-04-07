# Copyright 2016-2025 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
@rfm.xfail('demo failure', lambda test: test.x <= 2)
class stream_test(rfm.RunOnlyRegressionTest):
    valid_systems = ['*']
    valid_prog_environs = ['*']
    executable = 'stream.x'
    x = variable(int, value=1)

    @sanity_function
    def validate(self):
        if self.x < 2 or self.x > 2:
            return sn.assert_found(r'Slution Validates', self.stdout)
        elif self.x == 2:
            return sn.assert_found(r'Solution Validates', self.stdout)

    @performance_function('MB/s')
    def copy_bw(self):
        return sn.extractsingle(r'Copy:\s+(\S+)', self.stdout, 1, float)

    @performance_function('MB/s')
    def triad_bw(self):
        return sn.extractsingle(r'Triad:\s+(\S+)', self.stdout, 1, float)
