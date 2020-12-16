# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


class NoParams(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.name = 'noParams'
        self.descr = 'Hello World test'

        # All available systems are supported
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.executable = 'echo "Hello, World!"'
        self.tags = {'foo', 'bar'}
        self.sanity_patterns = sn.assert_found(r'Hello, World', self.stdout)
        self.maintainers = ['JO']


class TwoParams(NoParams):
    parameter('P0', 'a')
    parameter('P1', 'b')


class Abstract(TwoParams):
    parameter('P0')


class ExtendParams(TwoParams):
    parameter('P1', 'c', 'd', 'e', inherit_params=True)
    parameter('P2', 'f', 'g')
