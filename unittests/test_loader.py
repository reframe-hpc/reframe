import os
import unittest

from reframe.core.exceptions import ConfigError, NameConflictError
from reframe.core.systems import System
from reframe.frontend.loader import RegressionCheckLoader
from reframe.frontend.resources import ResourcesManager


class TestRegressionCheckLoader(unittest.TestCase):
    def setUp(self):
        self.loader = RegressionCheckLoader(['.'], ignore_conflicts=True)
        self.loader_with_path = RegressionCheckLoader(
            ['unittests/resources', 'unittests/foobar'],
            ignore_conflicts=True)
        self.loader_with_prefix = RegressionCheckLoader(
            load_path=['badchecks'],
            prefix=os.path.abspath('unittests/resources'))

        self.system = System('foo')
        self.resources = ResourcesManager()

    def test_load_file_relative(self):
        checks = self.loader.load_from_file(
            'unittests/resources/emptycheck.py',
            system=self.system, resources=self.resources
        )
        self.assertEqual(1, len(checks))
        self.assertEqual(checks[0].name, 'emptycheck')

    def test_load_file_absolute(self):
        checks = self.loader.load_from_file(
            os.path.abspath('unittests/resources/emptycheck.py'),
            system=self.system, resources=self.resources
        )
        self.assertEqual(1, len(checks))
        self.assertEqual(checks[0].name, 'emptycheck')

    def test_load_recursive(self):
        checks = self.loader.load_from_dir(
            'unittests/resources', recurse=True,
            system=self.system, resources=self.resources
        )
        self.assertEqual(11, len(checks))

    def test_load_all(self):
        checks = self.loader_with_path.load_all(system=self.system,
                                                resources=self.resources)
        self.assertEqual(10, len(checks))

    def test_load_all_with_prefix(self):
        checks = self.loader_with_prefix.load_all(system=self.system,
                                                  resources=self.resources)
        self.assertEqual(1, len(checks))

    def test_conflicted_checks(self):
        self.loader_with_path._ignore_conflicts = False
        self.assertRaises(NameConflictError, self.loader_with_path.load_all,
                          system=self.system, resources=self.resources)

    def test_load_error(self):
        self.assertRaises(OSError, self.loader.load_from_file,
                          'unittests/resources/foo.py')
