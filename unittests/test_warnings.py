import pytest
import semver
import warnings

import reframe
import reframe.core.runtime as rt
import reframe.core.warnings as warn
import reframe.utility.color as color
import unittests.fixtures as fixtures


@pytest.fixture(params=['colors', 'nocolors'])
def with_colors(request):
    with rt.temp_runtime(fixtures.BUILTIN_CONFIG_FILE, 'generic',
                         {'general/colorize': request.param == 'colors'}):
        yield request.param == 'colors'


def test_deprecation_warning():
    with pytest.warns(warn.ReframeDeprecationWarning):
        warn.user_deprecation_warning('deprecated')


def test_deprecation_warning_from_version():
    version = semver.VersionInfo.parse(reframe.VERSION).bump_minor()
    with warnings.catch_warnings(record=True) as w:
        warn.user_deprecation_warning('deprecated', str(version))
        assert len(w) == 0


def test_deprecation_warning_from_prerelease_version(monkeypatch):
    monkeypatch.setattr(reframe, 'VERSION', '1.0.0-dev.0')
    with pytest.warns(warn.ReframeDeprecationWarning):
        warn.user_deprecation_warning('deprecated', '1.0.0')


def test_deprecation_warning_formatting(with_colors):
    message = warnings.formatwarning(
        'deprecated', warn.ReframeDeprecationWarning, 'file', 10, 'a = 1'
    )
    expected = 'file:10: WARNING: deprecated\na = 1\n'
    if with_colors:
        expected = color.colorize(expected, color.YELLOW)

    assert message == expected


def test_deprecation_warning_formatting_noline(tmp_path, with_colors):
    srcfile = tmp_path / 'file'
    srcfile.touch()

    message = warnings.formatwarning(
        'deprecated', warn.ReframeDeprecationWarning, srcfile, 10
    )
    expected = f'{srcfile}:10: WARNING: deprecated\n<no line information>\n'
    if with_colors:
        expected = color.colorize(expected, color.YELLOW)

    assert message == expected


def test_random_warning_formatting():
    message = warnings.formatwarning(
        'deprecated', UserWarning, 'file', 10, 'a = 1'
    )
    assert message == f'file:10: UserWarning: deprecated\n  a = 1\n'
