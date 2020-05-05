# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Check for testing handling of the TERM signal
#
import os
import signal
import time

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class SelfKillCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.local = True
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']
        self.executable = 'echo hello'
        self.sanity_patterns = sn.assert_found('hello', self.stdout)
        self.tags = {type(self).__name__}
        self.maintainers = ['TM']

    @rfm.run_before('run')
    def self_kill(self):
        time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGTERM)
