# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause


import reframe as rfm
import reframe.utility.sanity as sn


class ElemTypeParam(rfm.RegressionMixin):
    elem_type = parameter(['float', 'double'])


@rfm.simple_test
class MakefileTestAlt(rfm.RegressionTest, ElemTypeParam):
    descr = 'Test demonstrating use of Makefiles'
    valid_systems = ['*']
    valid_prog_environs = ['clang', 'gnu']
    executable = './dotprod'
    executable_opts = ['100000']
    build_system = 'Make'

    @run_before('compile')
    def set_compiler_flags(self):
        self.build_system.cppflags = [f'-DELEM_TYPE={self.elem_type}']

    @sanity_function
    def validate_test(self):
        return sn.assert_found(
            rf'Result \({self.elem_type}\):', self.stdout
        )


@rfm.simple_test
class MakeOnlyTestAlt(rfm.CompileOnlyRegressionTest, ElemTypeParam):
    descr = 'Test demonstrating use of Makefiles'
    valid_systems = ['*']
    valid_prog_environs = ['clang', 'gnu']
    build_system = 'Make'

    @run_before('compile')
    def set_compiler_flags(self):
        self.build_system.cppflags = [f'-DELEM_TYPE={self.elem_type}']

    @sanity_function
    def validate_build(self):
        return sn.assert_not_found(r'warning', self.stdout)
