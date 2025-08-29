# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.udeps as udeps


@rfm.simple_test
class Test0(rfm.RunOnlyRegressionTest):
    descr = 'Test with meaningful description'
    valid_systems = ['sys0:p0', 'sys0:p1']
    valid_prog_environs = ['e0', 'e1']
    executable = 'echo'
    sanity_patterns = sn.assert_true(1)


@rfm.simple_test
class Test1(rfm.RunOnlyRegressionTest):
    descr = ''
    kind = parameter(['default', 'fully', 'by_part', 'by_case',
                      'custom', 'any', 'all', 'nodeps'])

    @run_after('init')
    def setup_deps(self):
        def custom_deps(src, dst):
            return (
                src[0] == 'p0' and
                src[1] == 'e0' and
                dst[0] == 'p1' and
                dst[1] == 'e1'
            )

        kindspec = {
            'fully': udeps.fully,
            'by_part': udeps.by_part,
            'by_case': udeps.by_case,
            'any': udeps.any(udeps.source(udeps.part_is('p0')),
                             udeps.dest(udeps.env_is('e1'))),
            'all': udeps.all(udeps.part_is('p0'),
                             udeps.dest(udeps.env_is('e0'))),
            'custom': custom_deps,
            'nodeps': lambda s, d: False,
        }
        self.valid_systems = ['sys0:p0', 'sys0:p1']
        self.valid_prog_environs = ['e0', 'e1']
        self.executable = 'echo'
        self.executable_opts = [self.name]
        self.sanity_patterns = sn.assert_found(self.name, self.stdout)
        if self.kind == 'default':
            self.depends_on('Test0')
        else:
            self.depends_on('Test0', kindspec[self.kind])


@rfm.simple_test
class Test2(rfm.RunOnlyRegressionTest):
    valid_systems = ['sys0:p0', 'sys0:p1']
    valid_prog_environs = ['e0', 'e1']
    executable = 'echo'
    executable_opts = ['Test2'] 
    

@rfm.simple_test
class Test3(rfm.RunOnlyRegressionTest):
    descr=required
    valid_systems = ['sys0:p0', 'sys0:p1']
    valid_prog_environs = ['e0', 'e1']
    executable = 'echo'
    executable_opts = ['Test3']
    
    @run_after('init')
    def setup_deps(self):
        self.depends_on('Test2')
