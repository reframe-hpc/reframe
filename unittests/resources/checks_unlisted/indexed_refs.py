# Copyright 2016-2026 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.builtins import (parameter,
                                   sanity_function,
                                   performance_function)


@rfm.simple_test
class IndexedRefsTest(rfm.RunOnlyRegressionTest):
    valid_systems = ['*']
    valid_prog_environs = ['*']
    p = parameter(['foo', 'bar'])
    executable = 'echo "throughput: 100"'
    reference = {
        '$index': ('p',),
        'foo': {
            'throughput': (100, None, None, 'MB/s')
        },
        'bar': {
            'throughput': (200, None, None, 'MB/s')
        }
    }

    @sanity_function
    def validate(self):
        return sn.assert_found(r'throughput', self.stdout)

    @performance_function('MB/s')
    def throughput(self):
        return sn.extractsingle(r'throughput: (\S+)', self.stdout, 1, float)
