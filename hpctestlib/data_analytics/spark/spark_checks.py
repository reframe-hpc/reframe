# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import math


import reframe as rfm
import reframe.utility.sanity as sn

from reframe.core.backends import getlauncher


@rfm.simple_test
class compute_pi_check(rfm.RunOnlyRegressionTest, pin_prefix=True):
    '''Test Apache Spark by computing PI.

    Apache Spark is a unified analytics engine for large-scale data
    processing. It provides high-level APIs in Java, Scala, Python
    and R, and an optimized engine that supports general execution
    graphs. It also supports a rich set of higher-level tools including
    Spark SQL for SQL and structured data processing, MLlib for machine
    learning, GraphX for graph processing, and Structured Streaming for
    incremental computation and stream processing (see spark.apache.org).

    This test checks that Spark is functioning correctly. To do this, it is
    necessary to define the tolerance of acceptable deviation. The tolerance
    is used to check that the computations are executed correctly, by
    comparing the value of pi calculated to the one obtained from the math
    library. The default assumption is that Spark is already installed on the
    system under test.

    '''

    #: Parameter encoding the variant of the test.
    #:
    #: :type: :class:`str`
    #: :values: ``['spark', 'pyspark']``
    variant = parameter(['spark', 'pyspark'])

    #: The absolute tolerance of the computed value of PI
    #:
    #: :type: :class:`float`
    #: :required: No
    #: :default: `0.01`
    tolerance = variable(float, value=0.01)

    #: The Spark installation prefix path
    #:
    #: :type: :class:`str`
    #: :required: Yes
    spark_prefix = variable(str)

    #: The local directories used by Spark
    #:
    #: :type: :class:`str`
    #: :required: No
    #: :default: `'/tmp'`
    spark_local_dirs = variable(str, value='/tmp')

    #: Amount of memory to use per executor process, following the JVM memory
    #: strings convention, i.e a number with a size unit suffix
    #: ("k", "m", "g" or "t") (e.g. 512m, 2g)
    #:
    #: :type: :class:`str`
    #: :required: Yes
    executor_memory = variable(str)

    #: The number of Spark workers per node
    #:
    #: :type: :class:`int`
    #: :required: No
    #: :default: `1`
    num_workers = variable(int, value=1)

    #: The number of cores per each Spark executor
    #:
    #: :type: :class:`int`
    #: :required: No
    #: :default: `1`
    exec_cores = variable(int, value=1)

    num_tasks = 3
    num_tasks_per_node = 1
    prerun_cmds = ['start-all.sh']
    postrun_cmds = ['stop-all.sh']
    executable = 'spark-submit'
    executable_opts = required
    tags = {'data-science', 'big-data'}

    @run_after('init')
    def set_description(self):
        self.mydescr = f'Simple calculation of pi with {self.variant}'

    @run_before('run')
    def set_job_launcher(self):
        # The job launcher has to be changed since the `spark-submit`
        # script is not used with srun.
        self.job.launcher = getlauncher('local')()

    @run_before('run')
    def prepare_run(self):
        self.variables = {
            'SPARK_WORKER_CORES': str(self.num_workers),
            'SPARK_LOCAL_DIRS': self.spark_local_dirs,
        }
        self.executable_opts = [
            f'--conf spark.default.parallelism={self.num_workers}',
            f'--conf spark.executor.cores={self.exec_cores}',
            f'--conf spark.executor.memory={self.executor_memory}',
            f'--master $SPARKURL'
        ]
        if self.variant == 'spark':
            self.executable_opts += [
                f'--class org.apache.spark.examples.SparkPi',
                f'{self.spark_prefix}/examples/jars/spark-examples*.jar 10000'
            ]
        elif self.variant == 'pyspark':
            self.executable_opts += ['spark_pi.py']

    @sanity_function
    def assert_pi_readout(self):
        '''Assert that the obtained pi value meets the specified tolerances.'''

        pi_value = sn.extractsingle(r'Pi is roughly\s+(?P<pi>\S+)',
                                    self.stdout, 'pi', float)
        return sn.assert_lt(sn.abs(pi_value - math.pi), self.tolerance)
