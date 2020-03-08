# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher


@rfm.simple_test
class SparkCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.descr = 'Simple calculation of pi with Spark'
        self.valid_systems = ['daint:gpu', 'daint:mc',
                              'dom:gpu', 'dom:mc']
        self.valid_prog_environs = ['builtin']
        self.modules = ['Spark']
        self.sourcesdir = None
        self.pre_run = ['start-all.sh']
        self.post_run = ['stop-all.sh']
        self.num_tasks = 2
        self.num_tasks_per_node = 1
        pi_value = sn.extractsingle(r'Pi is roughly\s+(?P<pi>\S+)',
                                    self.stdout, 'pi', float)
        self.sanity_patterns = sn.assert_lt(sn.abs(pi_value - math.pi), 0.01)
        self.maintainers = ['TM', 'TR']
        self.tags = {'production'}

    @rfm.run_before('run')
    def prepare_run(self):
        if self.current_partition.fullname in ['daint:gpu', 'dom:gpu']:
            num_workers = 12
            exec_cores = 3
        else:
            num_workers = 36
            exec_cores = 9

        self.variables = {
            'SPARK_WORKER_CORES': '%s' % num_workers,
            'SPARK_LOCAL_DIRS': '"/tmp"',
        }
        self.executable = (
            'spark-submit --conf spark.default.parallelism=%s '
            '--conf spark.executor.cores=%s --conf spark.executor.memory=15g '
            '--master $SPARKURL --class org.apache.spark.examples.SparkPi '
            '$EBROOTSPARK/examples/jars/spark-examples_2.11-2.3.1.jar 10000;'
            % (num_workers, exec_cores))
        # The job launcher has to be changed since the `spark-submit`
        # script is not used with srun.
        self.job.launcher = getlauncher('local')()
