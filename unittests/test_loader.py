# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import pytest
import unittest

from reframe.core.exceptions import (ConfigError, NameConflictError,
                                     RegressionTestLoadError)
from reframe.core.systems import System
from reframe.frontend.loader import RegressionCheckLoader


class TestRegressionCheckLoader(unittest.TestCase):
    def setUp(self):
        self.loader = RegressionCheckLoader(['.'], ignore_conflicts=True)
        self.loader_with_path = RegressionCheckLoader(
            ['unittests/resources/checks', 'unittests/foobar'],
            ignore_conflicts=True)
        self.loader_with_prefix = RegressionCheckLoader(
            load_path=['bad'],
            prefix=os.path.abspath('unittests/resources/checks'))

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

    # FIXME: Remove this test
    def _test_load_all_with_prefix(self):
        print(self.loader_with_prefix._load_path)
        checks = self.loader_with_prefix.load_all()
        assert 1 == len(checks)

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
