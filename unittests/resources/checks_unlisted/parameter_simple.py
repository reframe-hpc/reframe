# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class SimpleParameter(rfm.RunOnlyRegressionTest):
    message = parameter(['foo', 'bar'])
    valid_systems = ['*']
    valid_prog_environs = ['*']
    executable = 'echo'
    sanity_patterns = sn.assert_true(1)
