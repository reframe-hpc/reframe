# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm


@rfm.simple_test
class TestSimple(rfm.RegressionTest):
    # The test should not raise a deprecation warning even though
    # it overrides __init__
    def __init__(self):
        pass


@rfm.simple_test
class TestDeprecated(rfm.RegressionTest):
    def setup(self, partition, environ, **job_opts):
        super().setup(system, environ, **job_opts)


@rfm.simple_test
class TestDeprecatedRunOnly(rfm.RunOnlyRegressionTest):
    def setup(self, partition, environ, **job_opts):
        super().setup(system, environ, **job_opts)


@rfm.simple_test
class TestDeprecatedCompileOnly(rfm.CompileOnlyRegressionTest):
    def setup(self, partition, environ, **job_opts):
        super().setup(system, environ, **job_opts)


@rfm.simple_test
@rfm.extend_test
class TestExtended(rfm.RegressionTest):
    def __init__(self):
        pass

    def setup(self, partition, environ, **job_opts):
        super().setup(system, environ, **job_opts)


@rfm.simple_test
@rfm.extend_test
class TestExtendedRunOnly(rfm.RunOnlyRegressionTest):
    def __init__(self):
        pass

    def setup(self, partition, environ, **job_opts):
        super().setup(system, environ, **job_opts)


@rfm.simple_test
@rfm.extend_test
class TestExtendedCompileOnly(rfm.CompileOnlyRegressionTest):
    def __init__(self):
        pass

    def setup(self, partition, environ, **job_opts):
        super().setup(system, environ, **job_opts)