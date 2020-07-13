# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import pytest
import random
import shutil
import sys
import tempfile
import unittest

import reframe
import reframe.core.fields as fields
import reframe.utility as util
import reframe.utility.os_ext as os_ext
from reframe.core.exceptions import (SpawnedProcessError,
                                     SpawnedProcessTimeout)


class TestOSTools(unittest.TestCase):
    def test_command_success(self):
        completed = os_ext.run_command('echo foobar')
        assert completed.returncode == 0
        assert completed.stdout == 'foobar\n'

    def test_command_error(self):
        with pytest.raises(SpawnedProcessError,
                           match=r"command 'false' failed with exit code 1"):
            os_ext.run_command('false', check=True)

    def test_command_timeout(self):
        with pytest.raises(
                SpawnedProcessTimeout,
                match=r"command 'sleep 3' timed out after 2s"
        ) as exc_info:
            os_ext.run_command('sleep 3', timeout=2)

        assert exc_info.value.timeout == 2

        # Try to get the string repr. of the exception: see bug #658
        s = str(exc_info.value)

    def test_command_async(self):
        from datetime import datetime

        t_launch = datetime.now()
        t_sleep  = t_launch
        proc = os_ext.run_command_async('sleep 1')
        t_launch = datetime.now() - t_launch

        proc.wait()
        t_sleep = datetime.now() - t_sleep

        # Now check the timings
        assert t_launch.seconds < 1
        assert t_sleep.seconds >= 1

    def test_copytree(self):
        dir_src = tempfile.mkdtemp()
        dir_dst = tempfile.mkdtemp()
        os_ext.copytree(dir_src, dir_dst, dirs_exist_ok=True)
        shutil.rmtree(dir_src)
        shutil.rmtree(dir_dst)

    def test_copytree_src_parent_of_dst(self):
        dst_path = tempfile.mkdtemp()
        src_path = os.path.abspath(os.path.join(dst_path, '..'))

        with pytest.raises(ValueError):
            os_ext.copytree(src_path, dst_path)

        shutil.rmtree(dst_path)

    def _test_rmtree(self, *args, **kwargs):
        testdir = tempfile.mkdtemp()
        with open(os.path.join(testdir, 'foo.txt'), 'w') as fp:
            fp.write('hello\n')

        os_ext.rmtree(testdir, *args, **kwargs)
        assert not os.path.exists(testdir)

    def test_rmtree(self):
        self._test_rmtree()

    def test_rmtree_onerror(self):
        self._test_rmtree(onerror=lambda *args: None)

    def test_rmtree_error(self):
        # Try to remove an inexistent directory
        testdir = tempfile.mkdtemp()
        os.rmdir(testdir)
        with pytest.raises(OSError):
            os_ext.rmtree(testdir)

    def test_inpath(self):
        assert os_ext.inpath('/foo/bin', '/bin:/foo/bin:/usr/bin')
        assert not os_ext.inpath('/foo/bin', '/bin:/usr/local/bin')

    def _make_testdirs(self, prefix):
        # Create a temporary directory structure
        # foo/
        #   bar/
        #     boo/
        #   goo/
        # loo/
        #   bar/
        os.makedirs(os.path.join(prefix, 'foo', 'bar'), exist_ok=True)
        os.makedirs(os.path.join(prefix, 'foo', 'bar', 'boo'), exist_ok=True)
        os.makedirs(os.path.join(prefix, 'foo', 'goo'), exist_ok=True)
        os.makedirs(os.path.join(prefix, 'loo', 'bar'), exist_ok=True)

    def test_subdirs(self):
        prefix = tempfile.mkdtemp()
        self._make_testdirs(prefix)

        # Try to fool the algorithm by adding normal files
        open(os.path.join(prefix, 'foo', 'bar', 'file.txt'), 'w').close()
        open(os.path.join(prefix, 'loo', 'file.txt'), 'w').close()

        expected_subdirs = {prefix,
                            os.path.join(prefix, 'foo'),
                            os.path.join(prefix, 'foo', 'bar'),
                            os.path.join(prefix, 'foo', 'bar', 'boo'),
                            os.path.join(prefix, 'foo', 'goo'),
                            os.path.join(prefix, 'loo'),
                            os.path.join(prefix, 'loo', 'bar')}

        returned_subdirs = os_ext.subdirs(prefix)
        assert [prefix] == returned_subdirs

        returned_subdirs = os_ext.subdirs(prefix, recurse=True)
        assert expected_subdirs == set(returned_subdirs)
        shutil.rmtree(prefix)

    def test_samefile(self):
        # Create a temporary directory structure
        prefix = tempfile.mkdtemp()
        self._make_testdirs(prefix)

        # Try to fool the algorithm by adding symlinks
        os.symlink(os.path.join(prefix, 'foo'),
                   os.path.join(prefix, 'foolnk'))
        os.symlink(os.path.join(prefix, 'foolnk'),
                   os.path.join(prefix, 'foolnk1'))

        # Create a broken link on purpose
        os.symlink('/foo', os.path.join(prefix, 'broken'))
        os.symlink(os.path.join(prefix, 'broken'),
                   os.path.join(prefix, 'broken1'))

        assert os_ext.samefile('/foo', '/foo')
        assert os_ext.samefile('/foo', '/foo/')
        assert os_ext.samefile('/foo/bar', '/foo//bar/')
        assert os_ext.samefile(os.path.join(prefix, 'foo'),
                               os.path.join(prefix, 'foolnk'))
        assert os_ext.samefile(os.path.join(prefix, 'foo'),
                               os.path.join(prefix, 'foolnk1'))
        assert not os_ext.samefile('/foo', '/bar')
        assert os_ext.samefile('/foo', os.path.join(prefix, 'broken'))
        assert os_ext.samefile(os.path.join(prefix, 'broken'),
                               os.path.join(prefix, 'broken1'))
        shutil.rmtree(prefix)

    # FIXME: This should be changed in order to use the `monkeypatch`
    # fixture of `pytest` instead of creating an instance of `MonkeyPatch`
    def test_is_interactive(self):
        from _pytest.monkeypatch import MonkeyPatch  # noqa: F401, F403

        monkey = MonkeyPatch()
        with monkey.context() as c:
            # Set `sys.ps1` to immitate an interactive session
            c.setattr(sys, 'ps1', 'rfm>>> ', raising=False)
            assert os_ext.is_interactive()

    def test_is_url(self):
        repo_https = 'https://github.com/eth-cscs/reframe.git'
        repo_ssh = 'git@github.com:eth-cscs/reframe.git'
        assert os_ext.is_url(repo_https)
        assert not os_ext.is_url(repo_ssh)

    def test_git_repo_hash(self):
        # A git branch hash consists of 8(short) or 40 characters.
        assert len(os_ext.git_repo_hash()) == 8
        assert len(os_ext.git_repo_hash(short=False)) == 40
        assert os_ext.git_repo_hash(branch='invalid') is None
        assert os_ext.git_repo_hash(branch='') is None

    def test_git_repo_exists(self):
        assert os_ext.git_repo_exists(
            'https://github.com/eth-cscs/reframe.git', timeout=3)
        assert not os_ext.git_repo_exists('reframe.git', timeout=3)
        assert not os_ext.git_repo_exists(
            'https://github.com/eth-cscs/xxx', timeout=3)

    def test_force_remove_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as fp:
            pass

        assert os.path.exists(fp.name)
        os_ext.force_remove_file(fp.name)
        assert not os.path.exists(fp.name)

        # Try to remove a non-existent file
        os_ext.force_remove_file(fp.name)

    def test_expandvars_dollar(self):
        text = 'Hello, $(echo World)'
        assert 'Hello, World' == os_ext.expandvars(text)

        # Test nested expansion
        text = '$(echo Hello, $(echo World))'
        assert 'Hello, World' == os_ext.expandvars(text)

    def test_expandvars_backticks(self):
        text = 'Hello, `echo World`'
        assert 'Hello, World' == os_ext.expandvars(text)

        # Test nested expansion
        text = '`echo Hello, `echo World``'
        assert 'Hello, World' == os_ext.expandvars(text)

    def test_expandvars_mixed_syntax(self):
        text = '`echo Hello, $(echo World)`'
        assert 'Hello, World' == os_ext.expandvars(text)

        text = '$(echo Hello, `echo World`)'
        assert 'Hello, World' == os_ext.expandvars(text)

    def test_expandvars_error(self):
        text = 'Hello, $(foo)'
        with pytest.raises(SpawnedProcessError):
            os_ext.expandvars(text)

    def test_strange_syntax(self):
        text = 'Hello, $(foo`'
        assert 'Hello, $(foo`' == os_ext.expandvars(text)

        text = 'Hello, `foo)'
        assert 'Hello, `foo)' == os_ext.expandvars(text)

    def test_expandvars_nocmd(self):
        os.environ['FOO'] = 'World'
        text = 'Hello, $FOO'
        assert 'Hello, World' == os_ext.expandvars(text)

        text = 'Hello, ${FOO}'
        assert 'Hello, World' == os_ext.expandvars(text)
        del os.environ['FOO']


class TestCopyTree(unittest.TestCase):
    def setUp(self):
        # Create a test directory structure
        #
        # prefix/
        #   bar/
        #     bar.txt
        #     foo.txt
        #     foobar.txt
        #   foo/
        #     bar.txt
        #   bar.txt
        #   foo.txt
        #
        self.prefix = os.path.abspath(tempfile.mkdtemp())
        self.target = os.path.abspath(tempfile.mkdtemp())
        os.makedirs(os.path.join(self.prefix, 'bar'), exist_ok=True)
        os.makedirs(os.path.join(self.prefix, 'foo'), exist_ok=True)
        open(os.path.join(self.prefix, 'bar', 'bar.txt'), 'w').close()
        open(os.path.join(self.prefix, 'bar', 'foo.txt'), 'w').close()
        open(os.path.join(self.prefix, 'bar', 'foobar.txt'), 'w').close()
        open(os.path.join(self.prefix, 'foo', 'bar.txt'), 'w').close()
        open(os.path.join(self.prefix, 'bar.txt'), 'w').close()
        open(os.path.join(self.prefix, 'foo.txt'), 'w').close()

        # Create also a subdirectory in target, so as to check the recursion
        os.makedirs(os.path.join(self.target, 'foo'), exist_ok=True)

    def verify_target_directory(self, file_links=[]):
        '''Verify the directory structure'''
        assert os.path.exists(os.path.join(self.target, 'bar', 'bar.txt'))
        assert os.path.exists(os.path.join(self.target, 'bar', 'foo.txt'))
        assert os.path.exists(os.path.join(self.target, 'bar', 'foobar.txt'))
        assert os.path.exists(os.path.join(self.target, 'foo', 'bar.txt'))
        assert os.path.exists(os.path.join(self.target, 'bar.txt'))
        assert os.path.exists(os.path.join(self.target, 'foo.txt'))

        # Verify the symlinks
        for lf in file_links:
            target_name = os.path.abspath(os.path.join(self.prefix, lf))
            link_name = os.path.abspath(os.path.join(self.target, lf))
            assert os.path.islink(link_name)
            assert target_name == os.readlink(link_name)

    def test_virtual_copy_nolinks(self):
        os_ext.copytree_virtual(self.prefix, self.target, dirs_exist_ok=True)
        self.verify_target_directory()

    def test_virtual_copy_nolinks_dirs_exist(self):
        with pytest.raises(FileExistsError):
            os_ext.copytree_virtual(self.prefix, self.target)

    def test_virtual_copy_valid_links(self):
        file_links = ['bar/', 'foo/bar.txt', 'foo.txt']
        os_ext.copytree_virtual(self.prefix, self.target,
                                file_links, dirs_exist_ok=True)
        self.verify_target_directory(file_links)

    def test_virtual_copy_inexistent_links(self):
        file_links = ['foobar/', 'foo/bar.txt', 'foo.txt']
        with pytest.raises(ValueError):
            os_ext.copytree_virtual(self.prefix, self.target,
                                    file_links, dirs_exist_ok=True)

    def test_virtual_copy_absolute_paths(self):
        file_links = [os.path.join(self.prefix, 'bar'),
                      'foo/bar.txt', 'foo.txt']
        with pytest.raises(ValueError):
            os_ext.copytree_virtual(self.prefix, self.target,
                                    file_links, dirs_exist_ok=True)

    def test_virtual_copy_irrelevenant_paths(self):
        file_links = ['/bin', 'foo/bar.txt', 'foo.txt']
        with pytest.raises(ValueError):
            os_ext.copytree_virtual(self.prefix, self.target,
                                    file_links, dirs_exist_ok=True)

        file_links = [os.path.dirname(self.prefix), 'foo/bar.txt', 'foo.txt']
        with pytest.raises(ValueError):
            os_ext.copytree_virtual(self.prefix, self.target,
                                    file_links, dirs_exist_ok=True)

    def test_virtual_copy_linkself(self):
        file_links = ['.']
        with pytest.raises(ValueError):
            os_ext.copytree_virtual(self.prefix, self.target,
                                    file_links, dirs_exist_ok=True)

    def test_virtual_copy_linkparent(self):
        file_links = ['..']
        with pytest.raises(ValueError):
            os_ext.copytree_virtual(self.prefix, self.target,
                                    file_links, dirs_exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.prefix)
        shutil.rmtree(self.target)


class TestImportFromFile(unittest.TestCase):
    def test_load_relpath(self):
        module = util.import_module_from_file('reframe/__init__.py')
        assert reframe.VERSION == module.VERSION
        assert 'reframe' == module.__name__
        assert module is sys.modules.get('reframe')

    def test_load_directory(self):
        module = util.import_module_from_file('reframe')
        assert reframe.VERSION == module.VERSION
        assert 'reframe' == module.__name__
        assert module is sys.modules.get('reframe')

    def test_load_abspath(self):
        filename = os.path.abspath('reframe/__init__.py')
        module = util.import_module_from_file(filename)
        assert reframe.VERSION == module.VERSION
        assert 'reframe' == module.__name__
        assert module is sys.modules.get('reframe')

    def test_load_unknown_path(self):
        try:
            util.import_module_from_file('/foo')
            pytest.fail()
        except ImportError as e:
            assert 'foo' == e.name
            assert '/foo' == e.path

    def test_load_directory_relative(self):
        with os_ext.change_dir('reframe'):
            module = util.import_module_from_file('../reframe')
            assert reframe.VERSION == module.VERSION
            assert 'reframe' == module.__name__
            assert module is sys.modules.get('reframe')

    def test_load_relative(self):
        with os_ext.change_dir('reframe'):
            # Load a module from a directory up
            module = util.import_module_from_file('../reframe/__init__.py')
            assert reframe.VERSION == module.VERSION
            assert 'reframe' == module.__name__
            assert module is sys.modules.get('reframe')

            # Load a module from the current directory
            module = util.import_module_from_file('utility/os_ext.py')
            assert 'reframe.utility.os_ext' == module.__name__
            assert module is sys.modules.get('reframe.utility.os_ext')

    def test_load_outside_pkg(self):
        module = util.import_module_from_file(os.path.__file__)

        # os imports the OS-specific path libraries under the name `path`. Our
        # importer will import the actual file, thus the module name should be
        # the real one.
        assert (module is sys.modules.get('posixpath') or
                module is sys.modules.get('ntpath') or
                module is sys.modules.get('macpath'))

    def test_load_twice(self):
        filename = os.path.abspath('reframe')
        module1 = util.import_module_from_file(filename)
        module2 = util.import_module_from_file(filename)
        assert module1 is module2

    def test_load_namespace_package(self):
        module = util.import_module_from_file('unittests/resources')
        assert 'unittests' in sys.modules
        assert 'unittests.resources' in sys.modules


class TestPpretty:
    def test_simple_types(self):
        assert util.ppretty(1) == repr(1)
        assert util.ppretty(1.2) == repr(1.2)
        assert util.ppretty('a string') == repr('a string')
        assert util.ppretty([]) == '[]'
        assert util.ppretty(()) == '()'
        assert util.ppretty(set()) == 'set()'
        assert util.ppretty({}) == '{}'
        assert util.ppretty([1, 2, 3]) == '[\n    1,\n    2,\n    3\n]'
        assert util.ppretty((1, 2, 3)) == '(\n    1,\n    2,\n    3\n)'
        assert util.ppretty({1, 2, 3}) == '{\n    1,\n    2,\n    3\n}'
        assert util.ppretty({'a': 1, 'b': 2}) == ("{\n"
                                                  "    'a': 1,\n"
                                                  "    'b': 2\n"
                                                  "}")

    def test_mixed_types(self):
        assert (
            util.ppretty(['a string', 2, 'another string']) ==
            "[\n"
            "    'a string',\n"
            "    2,\n"
            "    'another string'\n"
            "]"
        )
        assert util.ppretty({'a': 1, 'b': (2, 3)}) == ("{\n"
                                                       "    'a': 1,\n"
                                                       "    'b': (\n"
                                                       "        2,\n"
                                                       "        3\n"
                                                       "    )\n"
                                                       "}")
        assert (
            util.ppretty({'a': 1, 'b': {2: {3: 4, 5: {}}}, 'c': 6}) ==
            "{\n"
            "    'a': 1,\n"
            "    'b': {\n"
            "        2: {\n"
            "            3: 4,\n"
            "            5: {}\n"
            "        }\n"
            "    },\n"
            "    'c': 6\n"
            "}")
        assert (
            util.ppretty({'a': 2, 34: (2, 3),
                          'b': [[], [1.2, 3.4], {1, 2}]}) ==
            "{\n"
            "    'a': 2,\n"
            "    34: (\n"
            "        2,\n"
            "        3\n"
            "    ),\n"
            "    'b': [\n"
            "        [],\n"
            "        [\n"
            "            1.2,\n"
            "            3.4\n"
            "        ],\n"
            "        {\n"
            "            1,\n"
            "            2\n"
            "        }\n"
            "    ]\n"
            "}"
        )

    def test_obj_print(self):
        class C:
            def __repr__(self):
                return '<class C>'

        class D:
            def __repr__(self):
                return '<class D>'

        c = C()
        d = D()
        assert util.ppretty(c) == '<class C>'
        assert util.ppretty(['a', 'b', c, d]) == ("[\n"
                                                  "    'a',\n"
                                                  "    'b',\n"
                                                  "    <class C>,\n"
                                                  "    <class D>\n"
                                                  "]")


class _X:
    def __init__(self):
        self._a = False


class _Y:
    def __init__(self, x, a=None):
        self.x = x
        self.y = 'foo'
        self.z = self
        self.a = a


def test_repr_default():
    c0, c1 = _Y(1), _Y(2, _X())
    s = util.repr([c0, c1])
    assert s == f'''[
    _Y({{
        'x': 1,
        'y': 'foo',
        'z': _Y(...)@{hex(id(c0))},
        'a': None
    }})@{hex(id(c0))},
    _Y({{
        'x': 2,
        'y': 'foo',
        'z': _Y(...)@{hex(id(c1))},
        'a': _X({{
            '_a': False
        }})@{hex(id(c1.a))}
    }})@{hex(id(c1))}
]'''


class TestChangeDirCtxManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.wd_save = os.getcwd()

    def test_change_dir_working(self):
        with os_ext.change_dir(self.temp_dir):
            assert os.getcwd(), self.temp_dir

        assert os.getcwd() == self.wd_save

    def test_exception_propagation(self):
        try:
            with os_ext.change_dir(self.temp_dir):
                raise RuntimeError
        except RuntimeError:
            assert os.getcwd() == self.wd_save
        else:
            pytest.fail('exception not propagated by the ctx manager')

    def tearDown(self):
        os.rmdir(self.temp_dir)


class TestMiscUtilities(unittest.TestCase):
    def test_allx(self):
        l1 = [1, 1, 1]
        l2 = [True, False]
        assert all(l1), util.allx(l1)
        assert not all(l2), util.allx(l2)
        assert not util.allx([])
        assert util.allx(i for i in [1, 1, 1])
        assert util.allx(i for i in range(1, 2))
        assert not util.allx(i for i in range(1))
        assert not util.allx(i for i in range(0))
        with pytest.raises(TypeError):
            util.allx(None)

    def test_decamelize(self):
        assert '' == util.decamelize('')
        assert 'my_base_class' == util.decamelize('MyBaseClass')
        assert 'my_base_class12' == util.decamelize('MyBaseClass12')
        assert 'my_class_a' == util.decamelize('MyClass_A')
        assert 'my_class' == util.decamelize('my_class')
        with pytest.raises(TypeError):
            util.decamelize(None)

        with pytest.raises(TypeError):
            util.decamelize(12)

    def test_sanitize(self):
        assert '' == util.toalphanum('')
        assert 'ab12' == util.toalphanum('ab12')
        assert 'ab1_2' == util.toalphanum('ab1_2')
        assert 'ab1__2' == util.toalphanum('ab1**2')
        assert 'ab__12_' == util.toalphanum('ab (12)')
        with pytest.raises(TypeError):
            util.toalphanum(None)

        with pytest.raises(TypeError):
            util.toalphanum(12)


class TestScopedDict(unittest.TestCase):
    def test_construction(self):
        d = {
            'a': {'k1': 3, 'k2': 4},
            'b': {'k3': 5}
        }
        namespace_dict = reframe.utility.ScopedDict()
        namespace_dict = reframe.utility.ScopedDict(d)

        # Change local dict and verify that the stored values are not affected
        d['a']['k1'] = 10
        d['b']['k3'] = 10
        assert 3 == namespace_dict['a:k1']
        assert 5 == namespace_dict['b:k3']
        del d['b']
        assert 'b:k3' in namespace_dict

        with pytest.raises(TypeError):
            reframe.utility.ScopedDict(1)

        with pytest.raises(TypeError):
            reframe.utility.ScopedDict({'a': 1, 'b': 2})

        with pytest.raises(TypeError):
            reframe.utility.ScopedDict([('a', 1), ('b', 2)])

        with pytest.raises(TypeError):
            reframe.utility.ScopedDict({'a': {1: 'k1'}, 'b': {2: 'k2'}})

    def test_contains(self):
        scoped_dict = reframe.utility.ScopedDict({
            'a': {'k1': 1, 'k2': 2},
            'a:b': {'k1': 3, 'k3': 4},
            'a:b:c': {'k2': 5, 'k3': 6},
            '*': {'k1': 7, 'k3': 9, 'k4': 10}
        })

        # Test simple lookup
        assert 'a:k1' in scoped_dict
        assert 'a:k2' in scoped_dict
        assert 'a:k3' in scoped_dict
        assert 'a:k4' in scoped_dict

        assert 'a:b:k1' in scoped_dict
        assert 'a:b:k2' in scoped_dict
        assert 'a:b:k3' in scoped_dict
        assert 'a:b:k4' in scoped_dict

        assert 'a:b:c:k1' in scoped_dict
        assert 'a:b:c:k2' in scoped_dict
        assert 'a:b:c:k3' in scoped_dict
        assert 'a:b:c:k4' in scoped_dict

        # Test global scope
        assert 'k1' in scoped_dict
        assert 'k2' not in scoped_dict
        assert 'k3' in scoped_dict
        assert 'k4' in scoped_dict

        assert ':k1' in scoped_dict
        assert ':k2' not in scoped_dict
        assert ':k3' in scoped_dict
        assert ':k4' in scoped_dict

        assert '*:k1' in scoped_dict
        assert '*:k2' not in scoped_dict
        assert '*:k3' in scoped_dict
        assert '*:k4' in scoped_dict

        # Try to get full scopes as keys
        assert 'a' not in scoped_dict
        assert 'a:b' not in scoped_dict
        assert 'a:b:c' not in scoped_dict
        assert 'a:b:c:d' not in scoped_dict
        assert '*' not in scoped_dict
        assert '' not in scoped_dict

    def test_iter_keys(self):
        scoped_dict = reframe.utility.ScopedDict({
            'a': {'k1': 1, 'k2': 2},
            'a:b': {'k1': 3, 'k3': 4},
            'a:b:c': {'k2': 5, 'k3': 6},
            '*': {'k1': 7, 'k3': 9, 'k4': 10}
        })

        expected_keys = [
            'a:k1', 'a:k2',
            'a:b:k1', 'a:b:k3',
            'a:b:c:k2', 'a:b:c:k3',
            '*:k1', '*:k3', '*:k4'
        ]
        assert sorted(expected_keys) == sorted(k for k in scoped_dict.keys())

    def test_iter_items(self):
        scoped_dict = reframe.utility.ScopedDict({
            'a': {'k1': 1, 'k2': 2},
            'a:b': {'k1': 3, 'k3': 4},
            'a:b:c': {'k2': 5, 'k3': 6},
            '*': {'k1': 7, 'k3': 9, 'k4': 10}
        })

        expected_items = [
            ('a:k1', 1), ('a:k2', 2),
            ('a:b:k1', 3), ('a:b:k3', 4),
            ('a:b:c:k2', 5), ('a:b:c:k3', 6),
            ('*:k1', 7), ('*:k3', 9), ('*:k4', 10)
        ]
        assert (sorted(expected_items) ==
                sorted(item for item in scoped_dict.items()))

    def test_iter_values(self):
        scoped_dict = reframe.utility.ScopedDict({
            'a': {'k1': 1, 'k2': 2},
            'a:b': {'k1': 3, 'k3': 4},
            'a:b:c': {'k2': 5, 'k3': 6},
            '*': {'k1': 7, 'k3': 9, 'k4': 10}
        })

        expected_values = [1, 2, 3, 4, 5, 6, 7, 9, 10]
        assert expected_values == sorted(v for v in scoped_dict.values())

    def test_key_resolution(self):
        scoped_dict = reframe.utility.ScopedDict({
            'a': {'k1': 1, 'k2': 2},
            'a:b': {'k1': 3, 'k3': 4},
            'a:b:c': {'k2': 5, 'k3': 6},
            '*': {'k1': 7, 'k3': 9, 'k4': 10}
        })

        assert 1 == scoped_dict['a:k1']
        assert 2 == scoped_dict['a:k2']
        assert 9 == scoped_dict['a:k3']
        assert 10 == scoped_dict['a:k4']

        assert 3 == scoped_dict['a:b:k1']
        assert 2 == scoped_dict['a:b:k2']
        assert 4 == scoped_dict['a:b:k3']
        assert 10 == scoped_dict['a:b:k4']

        assert 3 == scoped_dict['a:b:c:k1']
        assert 5 == scoped_dict['a:b:c:k2']
        assert 6 == scoped_dict['a:b:c:k3']
        assert 10 == scoped_dict['a:b:c:k4']

        # Test global scope
        assert 7 == scoped_dict['k1']
        with pytest.raises(KeyError):
            scoped_dict['k2']

        assert 9 == scoped_dict['k3']
        assert 10 == scoped_dict['k4']

        assert 7 == scoped_dict[':k1']
        with pytest.raises(KeyError):
            scoped_dict[':k2']

        assert 9 == scoped_dict[':k3']
        assert 10 == scoped_dict[':k4']

        assert 7 == scoped_dict['*:k1']
        with pytest.raises(KeyError):
            scoped_dict['*:k2']

        assert 9 == scoped_dict['*:k3']
        assert 10 == scoped_dict['*:k4']

        # Try to fool it, by requesting keys with scope names
        with pytest.raises(KeyError):
            scoped_dict['a']

        with pytest.raises(KeyError):
            scoped_dict['a:b']

        with pytest.raises(KeyError):
            scoped_dict['a:b:c']

        with pytest.raises(KeyError):
            scoped_dict['a:b:c:d']

        with pytest.raises(KeyError):
            scoped_dict['*']

        with pytest.raises(KeyError):
            scoped_dict['']

    def test_setitem(self):
        scoped_dict = reframe.utility.ScopedDict({
            'a': {'k1': 1, 'k2': 2},
            'a:b': {'k1': 3, 'k3': 4},
            'a:b:c': {'k2': 5, 'k3': 6},
            '*': {'k1': 7, 'k3': 9, 'k4': 10}
        })

        scoped_dict['a:k2'] = 20
        scoped_dict['c:k2'] = 30
        scoped_dict[':k4'] = 40
        scoped_dict['*:k5'] = 50
        scoped_dict['k6'] = 60
        assert 20 == scoped_dict['a:k2']
        assert 30 == scoped_dict['c:k2']
        assert 40 == scoped_dict[':k4']
        assert 50 == scoped_dict['k5']
        assert 60 == scoped_dict['k6']

    def test_delitem(self):
        scoped_dict = reframe.utility.ScopedDict({
            'a': {'k1': 1, 'k2': 2},
            'a:b': {'k1': 3, 'k3': 4},
            'a:b:c': {'k2': 5, 'k3': 6},
            '*': {'k1': 7, 'k3': 9, 'k4': 10}
        })

        # delete key
        del scoped_dict['a:k1']
        assert 7 == scoped_dict['a:k1']

        # delete key from global scope
        del scoped_dict['k1']
        assert 9 == scoped_dict['k3']
        assert 10 == scoped_dict['k4']
        with pytest.raises(KeyError):
            scoped_dict['k1']

        # delete a whole scope
        del scoped_dict['*']
        with pytest.raises(KeyError):
            scoped_dict[':k4']

        with pytest.raises(KeyError):
            scoped_dict['a:k3']

        # try to delete a non-existent key
        with pytest.raises(KeyError):
            del scoped_dict['a:k4']

        # test deletion of parent scope keeping a nested one
        scoped_dict = reframe.utility.ScopedDict()
        scoped_dict['s0:k0'] = 1
        scoped_dict['s0:s1:k0'] = 2
        scoped_dict['*:k0'] = 3
        del scoped_dict['s0']
        assert 3 == scoped_dict['s0:k0']
        assert 2 == scoped_dict['s0:s1:k0']

    def test_scope_key_name_pseudoconflict(self):
        scoped_dict = reframe.utility.ScopedDict({
            's0': {'s1': 1},
            's0:s1': {'k0': 2}
        })

        assert 1 == scoped_dict['s0:s1']
        assert 2 == scoped_dict['s0:s1:k0']

        del scoped_dict['s0:s1']
        assert 2 == scoped_dict['s0:s1:k0']
        with pytest.raises(KeyError):
            scoped_dict['s0:s1']

    def test_update(self):
        scoped_dict = util.ScopedDict({
            'a': {'k1': 1, 'k2': 2},
            'a:b': {'k1': 3, 'k3': 4},
            'a:b:c': {'k2': 5, 'k3': 6},
            '*': {'k1': 7, 'k3': 9, 'k4': 10}
        })

        scoped_dict_alt = util.ScopedDict({'a': {'k1': 3, 'k2': 5}})
        scoped_dict_alt.update({
            'a': {'k1': 1, 'k2': 2},
            'a:b': {'k1': 3, 'k3': 4},
            'a:b:c': {'k2': 5, 'k3': 6},
            '*': {'k1': 7, 'k3': 9, 'k4': 10}
        })
        assert scoped_dict == scoped_dict_alt


class TestReadOnlyViews(unittest.TestCase):
    def test_sequence(self):
        l = util.SequenceView([1, 2, 2])
        assert 1 == l[0]
        assert 3 == len(l)
        assert 2 in l
        assert l == [1, 2, 2]
        assert l == util.SequenceView([1, 2, 2])
        assert list(reversed(l)) == [2, 2, 1]
        assert 1 == l.index(2)
        assert 2 == l.count(2)
        assert str(l) == str([1, 2, 2])

        # Assert immutability
        m = l + [3, 4]
        assert [1, 2, 2, 3, 4] == m
        assert isinstance(m, util.SequenceView)

        m = l
        l += [3, 4]
        assert m is not l
        assert [1, 2, 2] == m
        assert [1, 2, 2, 3, 4] == l
        assert isinstance(l, util.SequenceView)

        with pytest.raises(TypeError):
            l[1] = 3

        with pytest.raises(TypeError):
            l[1:2] = [3]

        with pytest.raises(TypeError):
            l *= 3

        with pytest.raises(TypeError):
            del l[:1]

        with pytest.raises(AttributeError):
            l.append(3)

        with pytest.raises(AttributeError):
            l.clear()

        with pytest.raises(AttributeError):
            s = l.copy()

        with pytest.raises(AttributeError):
            l.extend([3, 4])

        with pytest.raises(AttributeError):
            l.insert(1, 4)

        with pytest.raises(AttributeError):
            l.pop()

        with pytest.raises(AttributeError):
            l.remove(2)

        with pytest.raises(AttributeError):
            l.reverse()

    def test_mapping(self):
        d = util.MappingView({'a': 1, 'b': 2})
        assert 1 == d['a']
        assert 2 == len(d)
        assert {'a': 1, 'b': 2} == dict(d)
        assert 'b' in d
        assert {'a', 'b'} == set(d.keys())
        assert {1, 2} == set(d.values())
        assert {('a', 1), ('b', 2)} == set(d.items())
        assert 2 == d.get('b')
        assert 3 == d.get('c', 3)
        assert {'a': 1, 'b': 2} == d
        assert d == util.MappingView({'b': 2, 'a': 1})
        assert str(d) == str({'a': 1, 'b': 2})
        assert {'a': 1, 'b': 2, 'c': 3} != d

        # Assert immutability
        with pytest.raises(TypeError):
            d['c'] = 3

        with pytest.raises(TypeError):
            del d['b']

        with pytest.raises(AttributeError):
            d.pop('a')

        with pytest.raises(AttributeError):
            d.popitem()

        with pytest.raises(AttributeError):
            d.clear()

        with pytest.raises(AttributeError):
            d.update({'a': 4, 'b': 5})

        with pytest.raises(AttributeError):
            d.setdefault('c', 3)


class TestOrderedSet(unittest.TestCase):
    def setUp(self):
        # Initialize all tests with the same seed
        random.seed(1)

    def test_construction(self):
        l = list(range(10))
        random.shuffle(l)

        s = util.OrderedSet(l + l)
        assert len(s) == 10
        for i in range(10):
            assert i in s

        assert list(s) == l

    def test_construction_empty(self):
        s = util.OrderedSet()
        assert s == set()
        assert set() == s

    def test_str(self):
        l = list(range(10))
        random.shuffle(l)

        s = util.OrderedSet(l)
        assert str(s) == str(l).replace('[', '{').replace(']', '}')

        s = util.OrderedSet()
        assert str(s) == type(s).__name__ + '()'

    def test_construction_error(self):
        with pytest.raises(TypeError):
            s = util.OrderedSet(2)

        with pytest.raises(TypeError):
            s = util.OrderedSet(1, 2, 3)

    def test_operators(self):
        s0 = util.OrderedSet(range(10))
        s1 = util.OrderedSet(range(20))
        s2 = util.OrderedSet(range(10, 20))

        assert s0 == set(range(10))
        assert set(range(10)) == s0
        assert s0 != s1
        assert s1 != s0

        assert s0 < s1
        assert s0 <= s1
        assert s0 <= s0
        assert s1 > s0
        assert s1 >= s0
        assert s1 >= s1

        assert s0.issubset(s1)
        assert s1.issuperset(s0)

        assert (s0 & s1) == s0
        assert (s0 & s2) == set()
        assert (s0 | s2) == s1

        assert (s1 - s0) == s2
        assert (s2 - s0) == s2

        assert (s0 ^ s1) == s2

        assert s0.isdisjoint(s2)
        assert not s0.isdisjoint(s1)
        assert s0.symmetric_difference(s1) == s2

    def test_union(self):
        l0 = list(range(10))
        l1 = list(range(10, 20))
        l2 = list(range(20, 30))
        random.shuffle(l0)
        random.shuffle(l1)
        random.shuffle(l2)

        s0 = util.OrderedSet(l0)
        s1 = util.OrderedSet(l1)
        s2 = util.OrderedSet(l2)

        assert list(s0.union(s1, s2)) == l0 + l1 + l2

    def test_intersection(self):
        l0 = list(range(10, 40))
        l1 = list(range(20, 40))
        l2 = list(range(20, 30))
        random.shuffle(l0)
        random.shuffle(l1)
        random.shuffle(l2)

        s0 = util.OrderedSet(l0)
        s1 = util.OrderedSet(l1)
        s2 = util.OrderedSet(l2)

        assert s0.intersection(s1, s2) == s2

    def test_difference(self):
        l0 = list(range(10, 40))
        l1 = list(range(20, 40))
        l2 = list(range(20, 30))
        random.shuffle(l0)
        random.shuffle(l1)
        random.shuffle(l2)

        s0 = util.OrderedSet(l0)
        s1 = util.OrderedSet(l1)
        s2 = util.OrderedSet(l2)

        assert s0.difference(s1, s2) == set(range(10, 20))

    def test_reversed(self):
        l = list(range(10))
        random.shuffle(l)

        s = util.OrderedSet(l)
        assert list(reversed(s)) == list(reversed(l))

    def test_concat_files(self):
        with tempfile.TemporaryDirectory(dir='unittests') as tmpdir:
            with os_ext.change_dir(tmpdir):
                file1 = 'in1.txt'
                file2 = 'in2.txt'
                concat_file = 'out.txt'
                with open(file1, 'w') as f1:
                    f1.write('Hello1')

                with open(file2, 'w') as f2:
                    f2.write('Hello2')

                os_ext.concat_files(concat_file, file1, file2, overwrite=True)
                with open(concat_file) as cf:
                    out = cf.read()
                    assert out == 'Hello1\nHello2\n'

    def test_unique_abs_paths(self):
        p1 = 'a/b/c'
        p2 = p1[:]
        p3 = 'a/b'
        p4 = '/d/e//'
        p5 = '/d/e/f'
        expected_paths = [os.path.abspath('a/b'), '/d/e']
        actual_paths = os_ext.unique_abs_paths(
            [p1, p2, p3, p4, p5])
        assert expected_paths == actual_paths

        expected_paths = [os.path.abspath('a/b/c'),  os.path.abspath('a/b'),
                          '/d/e', '/d/e/f']
        actual_paths = os_ext.unique_abs_paths(
            [p1, p2, p3, p4, p5], prune_children=False)
        assert expected_paths == actual_paths

        with pytest.raises(TypeError):
            os_ext.unique_abs_paths(None)


def test_cray_cdt_version(tmp_path, monkeypatch):
    # Mock up a CDT file
    rcfile = tmp_path / 'rcfile'
    with open(rcfile, 'w') as fp:
        fp.write('#%Module CDT 20.06\nblah blah\n')

    monkeypatch.setenv('MODULERCFILE', str(tmp_path / 'rcfile'))
    assert os_ext.cray_cdt_version() == '20.06'


def test_cray_cdt_version_unknown_fmt(tmp_path, monkeypatch):
    # Mock up a CDT file
    rcfile = tmp_path / 'rcfile'
    with open(rcfile, 'w') as fp:
        fp.write('random stuff')

    monkeypatch.setenv('MODULERCFILE', str(tmp_path / 'rcfile'))
    assert os_ext.cray_cdt_version() is None


def test_cray_cdt_version_empty_file(tmp_path, monkeypatch):
    # Mock up a CDT file
    rcfile = tmp_path / 'rcfile'
    rcfile.touch()
    monkeypatch.setenv('MODULERCFILE', str(tmp_path / 'rcfile'))
    assert os_ext.cray_cdt_version() is None


def test_cray_cdt_version_no_such_file(tmp_path, monkeypatch):
    # Mock up a CDT file
    rcfile = tmp_path / 'rcfile'
    monkeypatch.setenv('MODULERCFILE', str(tmp_path / 'rcfile'))
    assert os_ext.cray_cdt_version() is None
