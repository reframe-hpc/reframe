# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class MakefileTest(rfm.RegressionTest):
    elem_type = parameter(['float', 'double'])

    def __init__(self):
        self.descr = 'Test demonstrating use of Makefiles'
        self.valid_systems = ['*']
        self.valid_prog_environs = ['clang', 'gnu']
        self.executable = './dotprod'
        self.executable_opts = ['100000']
        self.build_system = 'Make'
        self.build_system.cppflags = [f'-DELEM_TYPE={self.elem_type}']
        self.sanity_patterns = sn.assert_found(
            rf'Result \({self.elem_type}\):', self.stdout
        )


@rfm.simple_test
class MakeOnlyTest(rfm.CompileOnlyRegressionTest):
    elem_type = parameter(['float', 'double'])

    def __init__(self):
        self.descr = 'Test demonstrating use of Makefiles'
        self.valid_systems = ['*']
        self.valid_prog_environs = ['clang', 'gnu']
        self.build_system = 'Make'
        self.build_system.cppflags = [f'-DELEM_TYPE={self.elem_type}']
        self.sanity_patterns = sn.assert_not_found(r'warning', self.stdout)
