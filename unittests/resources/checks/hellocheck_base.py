# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn

class HelloBaseTest(rfm.RunOnlyRegressionTest, base_test=True):
    def __init__(self):
        self.executable = './hello.sh'
        self.executable_opts = ['Hello, World!']
        self.local = True
        self.valid_prog_environs = ['*']
        self.valid_systems = ['*']
        self.sanity_patterns = sn.assert_found(
            r'Hello, World\!', self.stdout)

