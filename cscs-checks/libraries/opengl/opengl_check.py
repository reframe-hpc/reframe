# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class OpenGLTest(rfm.RegressionTest):
    def __init__(self):
        self.descr = 'Test for OpenGL'

        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.executable = 'tinyegl.sh'
        self.sanity_patterns = sn.assert_found(
            '0  0  0  0  0  0 76 51 25 76 51 25 76 51 25 76 51 25 76 51 25  0'
            '  0  0  0  0  0', self.stdout
        )
        self.maintainers = ['AJ', 'AJ']
        self.tags = {'scs', 'production'}
