# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class TestA(rfm.RunOnlyRegressionTest):
    z = parameter(range(10))
    executable = 'echo'
    valid_systems = ['*']
    valid_prog_environs = ['*']

    @run_after('init')
    def set_exec_opts(self):
        self.executable_opts = [str(self.z)]

    @sanity_function
    def validate(self):
        return sn.assert_eq(
            sn.extractsingle(r'\d+', self.stdout, 0, int), self.z
        )


@rfm.simple_test
class TestB(rfm.RunOnlyRegressionTest):
    executable = 'echo'
    valid_systems = ['*']
    valid_prog_environs = ['*']
    sanity_patterns = sn.assert_true(1)

    @run_after('init')
    def setdeps(self):
        variants = TestA.get_variant_nums(z=lambda x: x > 5)
        for v in variants:
            self.depends_on(TestA.variant_name(v))
