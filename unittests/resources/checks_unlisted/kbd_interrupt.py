# Copyright 2016-2023 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# A special check to simulate a keyboard interrupt
#
# The reason this test is in a different file is just for being loaded by the
# CLI unit tests exclusively.
#

import reframe as rfm


@rfm.simple_test
class KeyboardInterruptCheck(rfm.RunOnlyRegressionTest):
    local = True
    executable = 'sleep 1'
    valid_systems = ['*']
    valid_prog_environs = ['*']

    @run_before('setup')
    def raise_keyboard_interrupt(self):
        raise KeyboardInterrupt
