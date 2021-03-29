# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class EmptyTest(rfm.RegressionTest):
    def __init__(self):
        self.valid_systems = []
        self.valid_prog_environs = []
        self.sanity_patterns = sn.assert_true(1)
