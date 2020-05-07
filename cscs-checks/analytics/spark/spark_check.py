# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import math

import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher


@rfm.simple_test
class SparkAnalyticsCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.descr = 'Simple calculation of pi with Spark'
        self.valid_systems = ['daint:gpu', 'daint:mc']
        self.valid_prog_environs = ['PrgEnv-cray']
        self.modules = ['analytics']
        self.executable = 'start_analytics -t "spark-submit spark_pi.py"'

        pi_value = sn.extractsingle(r'Pi is roughly\s+(?P<pi>\S+)',
                                    self.stdout, 'pi', float)
        self.sanity_patterns = sn.assert_lt(sn.abs(pi_value - math.pi), 0.01)
        self.maintainers = ['TM', 'TR']
        self.tags = {'craype'}

    @rfm.run_after('setup')
    def set_num_tasks(self):
        if self.current_partition.fullname == 'daint:gpu':
            self.num_tasks = 48
            self.num_tasks_per_node = 12
        else:
            self.num_tasks = 72
            self.num_tasks_per_node = 18

    @rfm.run_before('run')
    def set_launcher(self):
        # The job launcher has to be changed since the `start_analytics`
        # script is not used with srun.
        self.job.launcher = getlauncher('local')()
