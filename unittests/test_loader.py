# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import pytest
import unittest

import reframe as rfm
from reframe.core.exceptions import (ConfigError, NameConflictError,
                                     ReframeDeprecationWarning,
                                     RegressionTestLoadError)
from reframe.core.systems import System
from reframe.frontend.loader import RegressionCheckLoader


class TestRegressionCheckLoader(unittest.TestCase):
    def setUp(self):
        self.loader = RegressionCheckLoader(['.'], ignore_conflicts=True)
        self.loader_with_path = RegressionCheckLoader(
            ['unittests/resources/checks', 'unittests/foobar'],
            ignore_conflicts=True)

    def test_load_file_relative(self):
        checks = self.loader.load_from_file(
            'unittests/resources/checks/emptycheck.py')
        assert 1 == len(checks)
        assert checks[0].name == 'EmptyTest'

    def test_load_file_absolute(self):
        checks = self.loader.load_from_file(
            os.path.abspath('unittests/resources/checks/emptycheck.py'))
        assert 1 == len(checks)
        assert checks[0].name == 'EmptyTest'

    def test_load_recursive(self):
        checks = self.loader.load_from_dir('unittests/resources/checks',
                                           recurse=True)
        assert 12 == len(checks)

    def test_load_all(self):
        checks = self.loader_with_path.load_all()
        assert 11 == len(checks)

    def test_load_new_syntax(self):
        checks = self.loader.load_from_file(
            'unittests/resources/checks_unlisted/good.py')
        assert 13 == len(checks)

    def test_conflicted_checks(self):
        self.loader_with_path._ignore_conflicts = False
        with pytest.raises(NameConflictError):
            self.loader_with_path.load_all()

    def test_load_error(self):
        with pytest.raises(OSError):
            self.loader.load_from_file('unittests/resources/checks/foo.py')

    def test_load_bad_required_version(self):
        with pytest.raises(ValueError):
            self.loader.load_from_file('unittests/resources/checks_unlisted/'
                                       'no_required_version.py')

    def test_load_bad_init(self):
        tests = self.loader.load_from_file(
            'unittests/resources/checks_unlisted/bad_init_check.py')
        assert 0 == len(tests)

    def test_special_test(self):
        with pytest.warns(ReframeDeprecationWarning):
            @rfm.simple_test
            class TestDeprecated(rfm.RegressionTest):
                def setup(self, partition, environ, **job_opts):
                    super().setup(system, environ, **job_opts)

        with pytest.warns(ReframeDeprecationWarning):
            @rfm.simple_test
            class TestDeprecatedRunOnly(rfm.RunOnlyRegressionTest):
                def setup(self, partition, environ, **job_opts):
                    super().setup(system, environ, **job_opts)

        with pytest.warns(ReframeDeprecationWarning):
            @rfm.simple_test
            class TestDeprecatedCompileOnly(rfm.CompileOnlyRegressionTest):
                def setup(self, partition, environ, **job_opts):
                    super().setup(system, environ, **job_opts)

        with pytest.warns(ReframeDeprecationWarning):
            @rfm.simple_test
            class TestDeprecatedCompileOnlyDerived(TestDeprecatedCompileOnly):
                def setup(self, partition, environ, **job_opts):
                    super().setup(system, environ, **job_opts)

        with pytest.warns(None) as warnings:
            @rfm.simple_test
            class TestSimple(rfm.RegressionTest):
                def __init__(self):
                    pass

            @rfm.simple_test
            class TestSpecial(rfm.RegressionTest, special=True):
                def __init__(self):
                    pass

                def setup(self, partition, environ, **job_opts):
                    super().setup(system, environ, **job_opts)

            @rfm.simple_test
            class TestSpecialRunOnly(rfm.RunOnlyRegressionTest,
                                     special=True):
                def __init__(self):
                    pass

                def setup(self, partition, environ, **job_opts):
                    super().setup(system, environ, **job_opts)

                def run(self):
                    super().run()

            @rfm.simple_test
            class TestSpecialCompileOnly(rfm.CompileOnlyRegressionTest,
                                         special=True):
                def __init__(self):
                    pass

                def setup(self, partition, environ, **job_opts):
                    super().setup(system, environ, **job_opts)

                def run(self):
                    super().run()

        assert not any(isinstance(w.message, ReframeDeprecationWarning)
                       for w in warnings)

        with pytest.warns(ReframeDeprecationWarning) as warnings:
            @rfm.simple_test
            class TestSpecialDerived(TestSpecial):
                def __init__(self):
                    pass

                def setup(self, partition, environ, **job_opts):
                    super().setup(system, environ, **job_opts)

                def run(self):
                    super().run()

        assert len(warnings) == 2

        @rfm.simple_test
        class TestFinal(rfm.RegressionTest):
            def __init__(self):
                pass

            @rfm.final
            def my_new_final(seld):
                pass

        with pytest.warns(ReframeDeprecationWarning):
            @rfm.simple_test
            class TestFinalDerived(TestFinal):
                def my_new_final(self, a, b):
                    pass
