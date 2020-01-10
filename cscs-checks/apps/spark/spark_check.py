import math

import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.launchers.registry import getlauncher


@rfm.simple_test
class SparkAnalyticsCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        super().__init__()
        self.descr = 'Simple calculation of pi with Spark'
        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc']
        self.valid_prog_environs = ['PrgEnv-cray']
        self.modules = ['Spark']
        self.sourcesdir = None
        self.variables = {
            'SPARK_WORKER_CORES': '36',
            'SPARK_LOCAL_DIRS': '"/tmp"',
        }
        # `SPARK_CONF` needs to be defined after running `start-all.sh`.
        self.pre_run = [
            'start-all.sh',
            ('SPARK_CONF="--conf spark.default.parallelism=10 '
             '--conf spark.executor.cores=8 '
             '--conf spark.executor.memory=15g"')
        ]
        self.executable = (
            'spark-submit ${SPARK_CONF} --master $SPARKURL '
            '--class org.apache.spark.examples.SparkPi '
            '$EBROOTSPARK/examples/jars/spark-examples_2.11-2.3.1.jar 10000;')
        self.post_run = ['stop-all.sh']
        self.num_tasks = 2
        self.num_tasks_per_node = 1
        pi_value = sn.extractsingle(r'Pi is roughly\s+(?P<pi>\S+)',
                                    self.stdout, 'pi', float)
        self.sanity_patterns = sn.assert_lt(sn.abs(pi_value - math.pi), 0.01)
        self.maintainers = ['TM', 'TR']
        self.tags = {'production'}

    def setup(self, partition, environ, **job_opts):
        super().setup(partition, environ, **job_opts)
        # The job launcher has to be changed since the `start_analytics`
        # script is not used with srun.
        self.job.launcher = getlauncher('local')()
