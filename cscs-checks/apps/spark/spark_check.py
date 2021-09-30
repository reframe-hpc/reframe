# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm

from reframe.core.backends import getlauncher
from hpctestlib.apps.spark.base_check import Spark_BaseCheck

@rfm.simple_test
class SparkCheck(Spark_BaseCheck):
    valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc']
    valid_prog_environs = ['builtin']
    modules = ['Spark']
    num_tasks = 3
    num_tasks_per_node = 1
    maintainers = ['TM', 'RS']
    tags = {'production'}

    @run_before('run')
    def prepare_run(self):
        if self.current_partition.fullname in ['daint:gpu', 'dom:gpu']:
            num_workers = 12
            exec_cores = 3
        else:
            num_workers = 36
            exec_cores = 9

        self.variables = {
            'SPARK_WORKER_CORES': str(num_workers),
            'SPARK_LOCAL_DIRS': '"/tmp"',
        }
        self.executable_opts = [
            f'--conf spark.default.parallelism={num_workers}',
            f'--conf spark.executor.cores={exec_cores}',
            f'--conf spark.executor.memory=15g',
            f'--master $SPARKURL'
        ]
        if self.variant == 'spark':
            self.executable_opts += [
                '--class org.apache.spark.examples.SparkPi',
                '$EBROOTSPARK/examples/jars/spark-examples*.jar 10000'
            ]

    @run_before('run')
    def set_job_launcher(self):
        # The job launcher has to be changed since the `spark-submit`
        # script is not used with srun.
        self.job.launcher = getlauncher('local')()
