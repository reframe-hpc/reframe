import unittest

from reframe.utility.versioning import Version


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
