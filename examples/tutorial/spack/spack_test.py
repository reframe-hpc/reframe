# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class ZlibSpackCheck(rfm.RegressionTest):
    descr = 'Demo test using Spack to build the test code'
    valid_systems = ['*']
    valid_prog_environs = ['builtin']
    executable = 'pkg-config'
    executable_opts = ['--libs', 'zlib']
    build_system = 'Spack'

    @run_before('compile')
    def setup_build_system(self):
        self.build_system.specs = ['zlib@1.3.1']

    @sanity_function
    def assert_version(self):
        return sn.assert_found(
            r'-L.*/spack/linux-.*/zlib-1.3.1-.*/lib -lz', self.stdout
        )
