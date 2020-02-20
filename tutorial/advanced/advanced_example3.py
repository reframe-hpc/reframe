# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class ExampleCompileOnlyTest(rfm.CompileOnlyRegressionTest):
    def __init__(self):
        self.descr = ('ReFrame tutorial demonstrating the class'
                      'CompileOnlyRegressionTest')
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.sanity_patterns = sn.assert_not_found('warning', self.stderr)
        self.maintainers = ['put-your-name-here']
        self.tags = {'tutorial'}
