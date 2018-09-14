import os
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
        self.assertEqual(1, len(checks))
        self.assertEqual(checks[0].name, 'EmptyTest')

    def test_load_file_absolute(self):
        checks = self.loader.load_from_file(
            os.path.abspath('unittests/resources/checks/emptycheck.py'))
        self.assertEqual(1, len(checks))
        self.assertEqual(checks[0].name, 'EmptyTest')

    def test_load_recursive(self):
        checks = self.loader.load_from_dir('unittests/resources/checks',
                                           recurse=True)
        self.assertEqual(11, len(checks))

    def test_load_all(self):
        checks = self.loader_with_path.load_all()
        self.assertEqual(10, len(checks))

    def test_load_all_with_prefix(self):
        checks = self.loader_with_prefix.load_all()
        self.assertEqual(1, len(checks))

    def test_load_new_syntax(self):
        checks = self.loader.load_from_file(
            'unittests/resources/checks_unlisted/good.py')
        self.assertEqual(13, len(checks))

    def test_load_mixed_syntax(self):
        self.assertRaises(RegressionTestLoadError, self.loader.load_from_file,
                          'unittests/resources/checks_unlisted/mixed.py')

    def test_conflicted_checks(self):
        self.loader_with_path._ignore_conflicts = False
        self.assertRaises(NameConflictError, self.loader_with_path.load_all)

    def test_load_error(self):
        self.assertRaises(OSError, self.loader.load_from_file,
                          'unittests/resources/checks/foo.py')

    def test_load_bad_required_version(self):
        with self.assertRaises(ValueError):
            self.loader.load_from_file('unittests/resources/checks_unlisted/'
                                       'no_required_version.py')
