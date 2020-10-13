# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class Test0(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.valid_systems = ['sys0:p0', 'sys0:p1']
        self.valid_prog_environs = ['e0', 'e1']
        self.executable = 'echo'
        self.executable_opts = [self.name]
        self.sanity_patterns = sn.assert_found(self.name, self.stdout)


@rfm.parameterized_test(*([kind] for kind in ['fully', 'by_env',
                                              'exact', 'default']))
class Test1(rfm.RunOnlyRegressionTest):
    def __init__(self, kind):
        kindspec = {
            'fully': rfm.DEPEND_FULLY,
            'by_env': rfm.DEPEND_BY_ENV,
            'exact': rfm.DEPEND_EXACT,
        }
        self.valid_systems = ['sys0:p0', 'sys0:p1']
        self.valid_prog_environs = ['e0', 'e1']
        self.executable = 'echo'
        self.executable_opts = [self.name]
        self.sanity_patterns = sn.assert_found(self.name, self.stdout)
        if kind == 'default':
            self.depends_on('Test0')
        elif kindspec[kind] == rfm.DEPEND_EXACT:
            self.depends_on('Test0', kindspec[kind],
                            {'e0': ['e0', 'e1'], 'e1': ['e1']})
        else:
            self.depends_on('Test0', kindspec[kind])
