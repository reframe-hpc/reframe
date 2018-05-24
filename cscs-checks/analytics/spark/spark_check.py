import os
import math

import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.launchers import LauncherWrapper
from reframe.core.launchers.registry import getlauncher


@rfm.simple_test
class SparkAnalyticsCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        super().__init__()
        self.descr = 'Simple calculation of pi with Spark'
        self.valid_systems = ['daint:gpu', 'daint:mc']
        self.valid_prog_environs = ['PrgEnv-cray']
        self.modules = ['analytics']
        self.executable = 'start_analytics -t "spark-submit spark_pi.py"'

        pi_value = sn.extractsingle(r'Pi is roughly\s+(?P<pi>\S+)',
                                    self.stdout, 'pi', float)
        self.sanity_patterns = sn.assert_lt(sn.abs(pi_value - math.pi), 0.01)
        self.maintainers = ['TM']
        self.tags = {'production'}

    def setup(self, partition, environ, **job_opts):
        if partition.fullname == 'daint:gpu':
            self.num_tasks = 48
            self.num_tasks_per_node = 12
        else:
            self.num_tasks = 72
            self.num_tasks_per_node = 18

        super().setup(partition, environ, **job_opts)
        # The job launcher has to be changed since the `start_analytics`
        # script is not used with srun.
        self.job.launcher = getlauncher('local')()
