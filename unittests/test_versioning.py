import unittest

from reframe.frontend.loader import RegressionCheckLoader
from reframe.utility.versioning import Version, VersionValidator


class TestVersioning(unittest.TestCase):
    def test_version_format(self):
        Version('1.2')
        Version('1.2.3')
        Version('1.2-dev0')
        Version('1.2-dev5')
        Version('1.2.3-dev2')
        self.assertRaises(ValueError, Version, None)
        self.assertRaises(ValueError, Version, '')
        self.assertRaises(ValueError, Version, '1')
        self.assertRaises(ValueError, Version, '1.2a')
        self.assertRaises(ValueError, Version, 'a.b.c')
        self.assertRaises(ValueError, Version, '1.2.3-dev')

    def test_comparing_versions(self):
        self.assertLess(Version('1.2'), Version('1.2.1'))
        self.assertLess(Version('1.2.1'), Version('1.2.2'))
        self.assertLess(Version('1.2.2'), Version('1.3-dev0'))
        self.assertLess(Version('1.3-dev0'), Version('1.3-dev1'))
        self.assertLess(Version('1.3-dev1'), Version('1.3'))
        self.assertEqual(Version('1.3'), Version('1.3.0'))
        self.assertEqual(Version('1.3-dev1'), Version('1.3.0-dev1'))
        self.assertGreater(Version('1.12.3'), Version('1.2.3'))
        self.assertGreater(Version('1.2.23'), Version('1.2.3'))

    def test_version_validation(self):
        conditions = [VersionValidator('<=1.0.0'),
                      VersionValidator('2.0.0..2.5'),
                      VersionValidator('3.0')]
        self.assertTrue(any(c.validate('0.1') for c in conditions))
        self.assertTrue(any(c.validate('2.0.0') for c in conditions))
        self.assertTrue(any(c.validate('2.2') for c in conditions))
        self.assertTrue(any(c.validate('2.5') for c in conditions))
        self.assertTrue(any(c.validate('3.0') for c in conditions))
        self.assertFalse(any(c.validate('3.1') for c in conditions))
        self.assertRaises(ValueError, VersionValidator, '2.0.0..')
        self.assertRaises(ValueError, VersionValidator, '..2.0.0')
        self.assertRaises(ValueError, VersionValidator, '1.0.0..2.0.0..3.0.0')
        self.assertRaises(ValueError, VersionValidator, '=>2.0.0')
        self.assertRaises(ValueError, VersionValidator, '2.0.0>')
        self.assertRaises(ValueError, VersionValidator, '2.0.0>1.0.0')
        self.assertRaises(ValueError, VersionValidator, '=>')
        self.assertRaises(ValueError, VersionValidator, '>1')
