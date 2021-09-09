# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher


class Spark_BaseCheck(rfm.RunOnlyRegressionTest, pin_prefix=True):
    '''Base class for the Spark Test.

    Apache Spark is a unified analytics engine for large-scale data
    processing. It provides high-level APIs in Java, Scala, Python
    and R, and an optimized engine that supports general execution
    graphs. It also supports a rich set of higher-level tools including
    Spark SQL for SQL and structured data processing, MLlib for machine
    learning, GraphX for graph processing, and Structured Streaming for
    incremental computation and stream processing (see spark.apache.org).

    The presented abstract run-only class checks the spark perfomance.
    To do this, it is necessary to define the tolerance of admissible
    deviation . This data is used to check if
    the task is being executed correctly, that is, the final value of pi
    is correct (approximately the same as obtained from library math).
    The default assumption is that Spark is already installed on
    the device under test.
    '''

    #: Name of the package to be checked
    variant = parameter(['spark', 'pyspark'])

    #: Maximum deviation from the table value of pi,
    #: that is acceptable.
    #:
    #: :type: float
    #: :default: 0.01
    tolerance = variable(float, value=0.01)

    @run_after('init')
    def set_description(self):
        self.mydescr = f'Simple calculation of pi with {self.variant}'

    @run_before('run')
    def set_run_cmds(self):
        self.prerun_cmds = ['start-all.sh']
        self.postrun_cmds = ['stop-all.sh']

    @run_before('run')
    def set_executable_opts(self):
        self.executable = 'spark-submit'
        if self.variant == 'pyspark':
            self.executable_opts.append('spark_pi.py')

    @sanity_function
    def assert_pi_readout(self):
        '''Assert the obtained pi value meets the specified tolerances.'''

        pi_value = sn.extractsingle(r'Pi is roughly\s+(?P<pi>\S+)',
                                    self.stdout, 'pi', float)
        return sn.assert_lt(sn.abs(pi_value - math.pi), self.tolerance)
