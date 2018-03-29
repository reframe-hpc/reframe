import unittest

from reframe.core.exceptions import ConfigError, EnvironError
from reframe.utility.versioning import Version, DevelopmentTag, ReleaseTag


class TestVersioning(unittest.TestCase):
    def test_version_format(self):
        self.assertIsInstance(Version('1.2'), Version)
        self.assertIsInstance(Version('1.2.3'), Version)
        self.assertIsInstance(Version('1.2-dev0'), Version)
        self.assertIsInstance(Version('1.2-dev5'), Version)
        self.assertIsInstance(Version('1.2.3-dev2'), Version)

    def test_comparing_versions(self):
        self.assertTrue(Version('1.2') == Version('1.2'))
        self.assertTrue(Version('1.2') < Version('1.3-dev0'))
        self.assertTrue(Version('1.3-dev0') < Version('1.3-dev1'))
        self.assertTrue(Version('1.3-dev1') < Version('1.4'))
        self.assertTrue(Version('1.2') == Version('1.2.0'))
        self.assertTrue(Version('1.2') < Version('1.2.1'))
        self.assertTrue(Version('1.2.1') < Version('1.2.2'))
        self.assertTrue(Version('1.2-dev1') == Version('1.2.0-dev1'))
        self.assertTrue(Version('1.2') > Version('1.2.0-dev1'))
        self.assertTrue(Version('1.2.0') > Version('1.2.0-dev1'))
        self.assertTrue(Version('1.2.1') > Version('1.2.0-dev1'))
        self.assertTrue(Version('1.2.1-dev1') > Version('1.2.0-dev1'))
        self.assertTrue(Version('1.12.3') > Version('1.2.3'))
        self.assertTrue(Version('1.2.23') > Version('1.2.3'))

    def test_version_tags(self):
        self.assertIsInstance(Version('1.2').tag, ReleaseTag)
        self.assertIsInstance(Version('1.2-dev0').tag, DevelopmentTag)

    def test_comparing_version_tags(self):
        self.assertTrue(DevelopmentTag(0) == DevelopmentTag(0))
        self.assertFalse(DevelopmentTag(0) != DevelopmentTag(0))
        self.assertTrue(DevelopmentTag(1) > DevelopmentTag(0))
        self.assertTrue(ReleaseTag() > DevelopmentTag(0))
        self.assertTrue(DevelopmentTag(0) < ReleaseTag())
        self.assertTrue(ReleaseTag() == ReleaseTag())
        self.assertFalse(ReleaseTag() != ReleaseTag())
        self.assertFalse(ReleaseTag() > ReleaseTag())
