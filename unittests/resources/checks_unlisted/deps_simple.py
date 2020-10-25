# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.dependencies as udeps
import reframe.utility.sanity as sn


@rfm.simple_test
class Test0(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.valid_systems = ['sys0:p0', 'sys0:p1']
        self.valid_prog_environs = ['e0', 'e1']
        self.executable = 'echo'
        self.executable_opts = [self.name]
        self.sanity_patterns = sn.assert_found(self.name, self.stdout)


@rfm.parameterized_test(*([kind] for kind in ['default', 'always',
                                              'part_equal', 'part_env_equal',
                                              'custom', 'any', 'all']))
class Test1(rfm.RunOnlyRegressionTest):
    def __init__(self, kind):
        def custom_deps(src, dst):
            return (
                src[0] == 'p0' and
                src[1] == 'e0' and
                dst[0] == 'p1' and
                dst[1] == 'e1'
            )

        kindspec = {
            'always': udeps.always,
            'part_equal': udeps.part_equal,
            'part_env_equal': udeps.part_env_equal,
            'any': udeps.any(udeps.source(udeps.part_is('p0')),
                             udeps.dest(udeps.env_is('e1'))),
            'all': udeps.all(udeps.part_is('p0'),
                             udeps.dest(udeps.env_is('e0'))),
            'custom': custom_deps,
        }
        self.valid_systems = ['sys0:p0', 'sys0:p1']
        self.valid_prog_environs = ['e0', 'e1']
        self.executable = 'echo'
        self.executable_opts = [self.name]
        self.sanity_patterns = sn.assert_found(self.name, self.stdout)
        if kind == 'default':
            self.depends_on('Test0')
        else:
            self.depends_on('Test0', kindspec[kind])