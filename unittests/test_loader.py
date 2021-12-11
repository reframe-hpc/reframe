# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import pytest

import reframe as rfm
from reframe.core.exceptions import ReframeSyntaxError
from reframe.core.warnings import ReframeDeprecationWarning
from reframe.frontend.loader import RegressionCheckLoader


@pytest.fixture
def loader():
    return RegressionCheckLoader(['.'])


@pytest.fixture
def loader_with_path():
    return RegressionCheckLoader(
        ['unittests/resources/checks', 'unittests/foobar']
    )


def test_load_file_relative(loader):
    checks = loader.load_from_file('unittests/resources/checks/emptycheck.py')
    assert 1 == len(checks)
    assert checks[0].name == 'EmptyTest'


def test_load_file_absolute(loader):
    checks = loader.load_from_file(
        os.path.abspath('unittests/resources/checks/emptycheck.py')
    )
    assert 1 == len(checks)
    assert checks[0].name == 'EmptyTest'


def test_load_recursive(loader):
    checks = loader.load_from_dir('unittests/resources/checks', recurse=True)
    assert 13 == len(checks)


def test_load_all(loader_with_path):
    checks = loader_with_path.load_all()
    assert 12 == len(checks)


def test_load_error(loader):
    with pytest.raises(OSError):
        loader.load_from_file('unittests/resources/checks/foo.py')


def test_load_bad_required_version(loader):
    with pytest.warns(ReframeDeprecationWarning):
        loader.load_from_file('unittests/resources/checks_unlisted/'
                              'no_required_version.py')


def test_load_bad_init(loader):
    tests = loader.load_from_file(
        'unittests/resources/checks_unlisted/bad_init_check.py'
    )
    assert 0 == len(tests)


def test_load_fixtures(loader):
    tests = loader.load_from_file(
        'unittests/resources/checks_unlisted/fixtures_simple.py'
    )
    assert 5 == len(tests)


def test_special_test():
    with pytest.raises(ReframeSyntaxError):
        @rfm.simple_test
        class TestOverride(rfm.RegressionTest):
            def setup(self, partition, environ, **job_opts):
                super().setup(partition, environ, **job_opts)

    with pytest.raises(ReframeSyntaxError):
        @rfm.simple_test
        class TestOverrideRunOnly(rfm.RunOnlyRegressionTest):
            def setup(self, partition, environ, **job_opts):
                super().setup(partition, environ, **job_opts)

    with pytest.raises(ReframeSyntaxError):
        @rfm.simple_test
        class TestOverrideCompileOnly(rfm.CompileOnlyRegressionTest):
            def setup(self, partition, environ, **job_opts):
                super().setup(partition, environ, **job_opts)

    @rfm.simple_test
    class TestSimple(rfm.RegressionTest):
        pass

    @rfm.simple_test
    class TestSpecial(rfm.RegressionTest, special=True):
        def setup(self, partition, environ, **job_opts):
            super().setup(partition, environ, **job_opts)

    @rfm.simple_test
    class TestSpecialRunOnly(rfm.RunOnlyRegressionTest,
                             special=True):
        def setup(self, partition, environ, **job_opts):
            super().setup(partition, environ, **job_opts)

    @rfm.simple_test
    class TestSpecialCompileOnly(rfm.CompileOnlyRegressionTest,
                                 special=True):
        def setup(self, partition, environ, **job_opts):
            super().setup(partition, environ, **job_opts)

    with pytest.raises(ReframeSyntaxError):
        @rfm.simple_test
        class TestSpecialDerived(TestSpecial):
            def setup(self, partition, environ, **job_opts):
                super().setup(partition, environ, **job_opts)

    with pytest.warns(ReframeDeprecationWarning):
        @rfm.simple_test
        class TestFinal(rfm.RegressionTest):
            @rfm.final
            def my_new_final(self):
                pass
