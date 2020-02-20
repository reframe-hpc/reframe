import pytest
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
        with pytest.raises(ValueError):
            Version(None)
        with pytest.raises(ValueError):
            Version('')
        with pytest.raises(ValueError):
            Version('1')
        with pytest.raises(ValueError):
            Version('1.2a')
        with pytest.raises(ValueError):
            Version('a.b.c')
        with pytest.raises(ValueError):
            Version('1.2.3-dev')

    def test_comparing_versions(self):
        assert Version('1.2') < Version('1.2.1')
        assert Version('1.2.1') < Version('1.2.2')
        assert Version('1.2.2') < Version('1.3-dev0')
        assert Version('1.3-dev0') < Version('1.3-dev1')
        assert Version('1.3-dev1') < Version('1.3')
        assert Version('1.3') == Version('1.3.0')
        assert Version('1.3-dev1') == Version('1.3.0-dev1')
        assert Version('1.12.3') > Version('1.2.3')
        assert Version('1.2.23') > Version('1.2.3')

    def test_version_validation(self):
        conditions = [VersionValidator('<=1.0.0'),
                      VersionValidator('2.0.0..2.5'),
                      VersionValidator('3.0')]
        assert any(c.validate('0.1') for c in conditions)
        assert any(c.validate('2.0.0') for c in conditions)
        assert any(c.validate('2.2') for c in conditions)
        assert any(c.validate('2.5') for c in conditions)
        assert any(c.validate('3.0') for c in conditions)
        assert not any(c.validate('3.1') for c in conditions)
        with pytest.raises(ValueError):
            VersionValidator('2.0.0..')
        with pytest.raises(ValueError):
            VersionValidator('..2.0.0')
        with pytest.raises(ValueError):
            VersionValidator('1.0.0..2.0.0..3.0.0')
        with pytest.raises(ValueError):
            VersionValidator('=>2.0.0')
        with pytest.raises(ValueError):
            VersionValidator('2.0.0>')
        with pytest.raises(ValueError):
            VersionValidator('2.0.0>1.0.0')
        with pytest.raises(ValueError):
            VersionValidator('=>')
        with pytest.raises(ValueError):
            VersionValidator('>1')
