# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher


class SparkBaseCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc']
        self.valid_prog_environs = ['builtin']
        self.modules = ['Spark']
        self.prerun_cmds = ['start-all.sh']
        self.postrun_cmds = ['stop-all.sh']
        self.num_tasks = 3
        self.num_tasks_per_node = 1
        pi_value = sn.extractsingle(r'Pi is roughly\s+(?P<pi>\S+)',
                                    self.stdout, 'pi', float)
        self.sanity_patterns = sn.assert_lt(sn.abs(pi_value - math.pi), 0.01)
        self.maintainers = ['TM', 'RS']
        self.tags = {'production'}

    def prepare_run(self):
        if self.current_partition.fullname in ['daint:gpu', 'dom:gpu']:
            num_workers = 12
            exec_cores = 3
        else:
            num_workers = 36
            exec_cores = 9

        self.variables = {
            'SPARK_WORKER_CORES': f'{num_workers}',
            'SPARK_LOCAL_DIRS': '"/tmp"',
        }
        self.executable = 'spark-submit'
        self.executable_opts = [
            f'--conf spark.default.parallelism={num_workers}',
            f'--conf spark.executor.cores={exec_cores}',
            f'--conf spark.executor.memory=15g',
            f'--master $SPARKURL'
        ]
        # The job launcher has to be changed since the `spark-submit`
        # script is not used with srun.
        self.job.launcher = getlauncher('local')()


@rfm.simple_test
class SparkCheck(SparkBaseCheck):
    def __init__(self):
        super().__init__()
        self.descr = 'Simple calculation of pi with Spark'
        self.sourcesdir = None

    @rfm.run_before('run')
    def prepare_run(self):
        super().prepare_run()
        self.executable_opts.extend([
            '--class org.apache.spark.examples.SparkPi',
            '$EBROOTSPARK/examples/jars/spark-examples*.jar 10000'
        ])


@rfm.simple_test
class PySparkCheck(SparkBaseCheck):
    def __init__(self):
        super().__init__()
        self.descr = 'Simple calculation of pi with PySpark'

    @rfm.run_before('run')
    def prepare_run(self):
        super().prepare_run()
        self.executable_opts.append('spark_pi.py')
