# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import pytest
import shutil

import reframe as rfm
import reframe.utility.osext as osext
from reframe.core.exceptions import ReframeSyntaxError
from reframe.frontend.loader import RegressionCheckLoader


@pytest.fixture
def loader():
    return RegressionCheckLoader(['.'])


@pytest.fixture
def loader_with_path():
    return RegressionCheckLoader(
        ['unittests/resources/checks', 'unittests/foobar']
    )


@pytest.fixture
def loader_with_path_tmpdir(tmp_path):
    test_dir_a = tmp_path / 'a'
    test_dir_b = tmp_path / 'b'
    os.mkdir(test_dir_a)
    os.mkdir(test_dir_b)
    test_a = 'unittests/resources/checks/emptycheck.py'
    test_b = 'unittests/resources/checks/hellocheck.py'
    shutil.copyfile(test_a, test_dir_a / 'test.py')
    shutil.copyfile(test_b, test_dir_b / 'test.py')
    return RegressionCheckLoader(
        [test_dir_a.as_posix(), test_dir_b.as_posix()]
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


def test_existing_module_name(loader, tmp_path):
    test_file = tmp_path / 'os.py'
    shutil.copyfile('unittests/resources/checks/emptycheck.py', test_file)
    checks = loader.load_from_file(test_file)
    assert 1 == len(checks)
    assert checks[0].name == 'EmptyTest'


def test_same_filename_different_path(loader_with_path_tmpdir):
    checks = loader_with_path_tmpdir.load_all()
    assert 3 == len(checks)


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


def test_relative_import_outside_rfm_prefix(loader, tmp_path):
    # If a test file resides under the reframe installation prefix, it will be
    # imported as a hierarchical module. If not, we want to make sure that
    # reframe will still load its parent modules

    osext.copytree(
        os.path.abspath('unittests/resources/checks_unlisted/testlib'),
        tmp_path / 'testlib', dirs_exist_ok=True
    )
    tests = loader.load_from_file(str(tmp_path / 'testlib' / 'simple.py'))
    assert len(tests) == 2

    # Test nested library tests
    tests = loader.load_from_file(
        str(tmp_path / 'testlib' / 'nested' / 'dummy.py')
    )
    assert len(tests) == 2
