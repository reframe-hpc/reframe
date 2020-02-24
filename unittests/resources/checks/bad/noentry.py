# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from reframe.core.pipeline import RegressionTest


class EmptyTest(RegressionTest):
    def __init__(self, **kwargs):
        super().__init__('emptycheck', os.path.dirname(__file__), **kwargs)
