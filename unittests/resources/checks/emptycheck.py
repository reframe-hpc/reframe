# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm


@rfm.simple_test
class EmptyTest(rfm.RegressionTest):
    def __init__(self):
        self.valid_systems = []
        self.valid_prog_environs = []
