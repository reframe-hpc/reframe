# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import reframe as rfm
import reframe.utility.sanity as sn


class Spark_BaseCheck(rfm.RunOnlyRegressionTest, pin_prefix=True):
    '''Base class for the Spark Test.

    Apache Spark is a unified analytics engine for large-scale data
    processing. It provides high-level APIs in Java, Scala, Python
    and R, and an optimized engine that supports general execution
    graphs. It also supports a rich set of higher-level tools including
    Spark SQL for SQL and structured data processing, MLlib for machine
    learning, GraphX for graph processing, and Structured Streaming for
    incremental computation and stream processing (see spark.apache.org).

    The present abstract run-only class checks the Spark perfomance.
    To do this, it is necessary to define the tolerance of acceptable
    deviation. The tolerance is used to check if the task executed correctly,
    comparing the value of pi calculated to the one obtained from the math
    library. The default assumption is that Spark is already installed on the
    system under test.
    '''

    variant = parameter(['spark', 'pyspark'])
    tolerance = variable(float, value=0.01)
    prerun_cmds = ['start-all.sh']
    postrun_cmds = ['stop-all.sh']
    executable = 'spark-submit'
    executable_opts = required

    @run_after('init')
    def set_description(self):
        self.mydescr = f'Simple calculation of pi with {self.variant}'

    @run_before('run')
    def set_pyspark_opts(self):
        if self.variant == 'pyspark':
            self.executable_opts.append('spark_pi.py')

    @sanity_function
    def assert_pi_readout(self):
        '''Assert that the obtained pi value meets the specified tolerances.'''

        pi_value = sn.extractsingle(r'Pi is roughly\s+(?P<pi>\S+)',
                                    self.stdout, 'pi', float)
        return sn.assert_lt(sn.abs(pi_value - math.pi), self.tolerance)
