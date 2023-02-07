# Copyright 2016-2023 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn


class simple_make_build(rfm.CompileOnlyRegressionTest, pin_prefix=True):
    descr = 'Simple Make build fixture'
    sourcesdir = 'src'
    build_system = 'Make'

    @sanity_function
    def assert_success(self):
        return sn.assert_not_found(r'\S+', self.stderr)


@rfm.simple_test
class simple_check(rfm.RunOnlyRegressionTest):
    descr = 'Simple test'
    valid_systems = ['*']
    valid_prog_environs = ['builtin']
    executable = 'hello.x'
    executable_opts = ['World']

    hello_binaries = fixture(simple_make_build, scope='environment')

    @run_before('run')
    def add_exec_prefix(self):
        self.executable = os.path.join(self.hello_binaries.stagedir,
                                       self.executable)

    @sanity_function
    def assert_sanity(self):
        return sn.assert_found(r'Hello World', self.stdout)
