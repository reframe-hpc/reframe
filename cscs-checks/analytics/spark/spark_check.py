import os
import math

import reframe.utility.sanity as sn
from reframe.core.pipeline import RunOnlyRegressionTest
from reframe.core.launchers import LauncherWrapper
from reframe.core.launchers.registry import getlauncher


class SparkAnalyticsCheck(RunOnlyRegressionTest):
    def __init__(self, **kwargs):
        super().__init__('spark_check',
                         os.path.dirname(__file__), **kwargs)
        self.descr = 'Simple calculation of pi with Spark'
        self.valid_systems = ['daint:gpu', 'daint:mc']
        self.valid_prog_environs = ['PrgEnv-cray']

        self.modules = ['analytics']

        self.executable = 'start_analytics -t "spark-submit spark_pi.py"'

        pi_value = sn.extractsingle(r'Pi is roughly\s+(?P<pi>\S+)',
                                    self.stdout, 'pi', float)
        pi_reference = math.pi
        pi_diff = sn.abs(pi_value - pi_reference)

        self.sanity_patterns = sn.assert_lt(pi_diff, 0.01)

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
        self.job.launcher = getlauncher('local')()


def _get_checks(**kwargs):
    return [SparkAnalyticsCheck(**kwargs)]
