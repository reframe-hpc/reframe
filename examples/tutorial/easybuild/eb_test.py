# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class ZlibEBCheck(rfm.RegressionTest):
    descr = 'Demo test using EasyBuild to build the test code'
    valid_systems = ['*']
    valid_prog_environs = ['builtin']
    executable = 'ls'
    executable_opts = ['$LD_LIBRARY_PATH/libz.so.1.3.1']
    build_system = 'EasyBuild'

    @run_before('compile')
    def setup_build_system(self):
        self.build_system.easyconfigs = ['zlib-1.3.1.eb']
        self.build_system.options = ['-f']

    @run_before('run')
    def prepare_run(self):
        self.modules = self.build_system.generated_modules

    @sanity_function
    def assert_exists(self):
        return sn.assert_eq(self.job.exitcode, 0)
