# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import pytest
import random
import shutil
import sys

import reframe
import reframe.core.fields as fields
import reframe.core.runtime as rt
import reframe.utility as util
import reframe.utility.os_ext as os_ext
import unittests.fixtures as fixtures

from reframe.core.exceptions import (ConfigError,
                                     SpawnedProcessError,
                                     SpawnedProcessTimeout)


def test_command_success():
    completed = os_ext.run_command('echo foobar')
    assert completed.returncode == 0
    assert completed.stdout == 'foobar\n'


def test_command_error():
    with pytest.raises(SpawnedProcessError,
                       match=r"command 'false' failed with exit code 1"):
        os_ext.run_command('false', check=True)


def test_command_timeout():
    with pytest.raises(
        SpawnedProcessTimeout, match=r"command 'sleep 3' timed out "
                                     r'after 2s') as exc_info:

        os_ext.run_command('sleep 3', timeout=2)

    assert exc_info.value.timeout == 2

    # Try to get the string repr. of the exception: see bug #658
    s = str(exc_info.value)


def test_command_async():
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


def test_copytree(tmp_path):
    dir_src = tmp_path / 'src'
    dir_src.mkdir()
    dir_dst = tmp_path / 'dst'
    dir_dst.mkdir()
    os_ext.copytree(str(dir_src), str(dir_dst), dirs_exist_ok=True)


def test_copytree_src_parent_of_dst(tmp_path):
    dst_path = tmp_path / 'dst'
    src_path = (dst_path / '..').resolve()

    with pytest.raises(ValueError):
        os_ext.copytree(str(src_path), str(dst_path))


@pytest.fixture
def rmtree(tmp_path):
    testdir = tmp_path / 'test'
    testdir.mkdir()
    with open(os.path.join(str(testdir), 'foo.txt'), 'w') as fp:
        fp.write('hello\n')

    def _rmtree(*args, **kwargs):
        os_ext.rmtree(testdir, *args, **kwargs)
        assert not os.path.exists(testdir)

    return _rmtree


def test_rmtree(rmtree):
    rmtree()


def test_rmtree_onerror(rmtree):
    rmtree(onerror=lambda *args: None)


def test_rmtree_error(tmp_path):
    # Try to remove an inexistent directory
    testdir = tmp_path / 'tmp'
    testdir.mkdir()
    os.rmdir(str(testdir))
    with pytest.raises(OSError):
        os_ext.rmtree(testdir)


def test_inpath():
    assert os_ext.inpath('/foo/bin', '/bin:/foo/bin:/usr/bin')
    assert not os_ext.inpath('/foo/bin', '/bin:/usr/local/bin')


@pytest.fixture
def tempdirs(tmp_path):
    # Create a temporary directory structure
    # foo/
    #   bar/
    #     boo/
    #   goo/
    # loo/
    #   bar/
    prefix = tmp_path / 'prefix'
    (prefix / 'foo' / 'bar' / 'boo').mkdir(parents=True)
    (prefix / 'foo' / 'goo').mkdir()
    (prefix / 'loo' / 'bar').mkdir(parents=True)
    return prefix


def test_subdirs(tempdirs):
    # Try to fool the algorithm by adding normal files
    prefix_name = str(tempdirs)
    open(os.path.join(prefix_name, 'foo', 'bar', 'file.txt'), 'w').close()
    open(os.path.join(prefix_name, 'loo', 'file.txt'), 'w').close()

    expected_subdirs = {prefix_name,
                        os.path.join(prefix_name, 'foo'),
                        os.path.join(prefix_name, 'foo', 'bar'),
                        os.path.join(prefix_name, 'foo', 'bar', 'boo'),
                        os.path.join(prefix_name, 'foo', 'goo'),
                        os.path.join(prefix_name, 'loo'),
                        os.path.join(prefix_name, 'loo', 'bar')}

    returned_subdirs = os_ext.subdirs(prefix_name)
    assert [prefix_name] == returned_subdirs

    returned_subdirs = os_ext.subdirs(prefix_name, recurse=True)
    assert expected_subdirs == set(returned_subdirs)


def test_samefile(tempdirs):
    prefix_name = str(tempdirs)

    # Try to fool the algorithm by adding symlinks
    os.symlink(os.path.join(prefix_name, 'foo'),
               os.path.join(prefix_name, 'foolnk'))
    os.symlink(os.path.join(prefix_name, 'foolnk'),
               os.path.join(prefix_name, 'foolnk1'))

    # Create a broken link on purpose
    os.symlink('/foo', os.path.join(prefix_name, 'broken'))
    os.symlink(os.path.join(prefix_name, 'broken'),
               os.path.join(prefix_name, 'broken1'))

    assert os_ext.samefile('/foo', '/foo')
    assert os_ext.samefile('/foo', '/foo/')
    assert os_ext.samefile('/foo/bar', '/foo//bar/')
    assert os_ext.samefile(os.path.join(prefix_name, 'foo'),
                           os.path.join(prefix_name, 'foolnk'))
    assert os_ext.samefile(os.path.join(prefix_name, 'foo'),
                           os.path.join(prefix_name, 'foolnk1'))
    assert not os_ext.samefile('/foo', '/bar')
    assert os_ext.samefile('/foo', os.path.join(prefix_name, 'broken'))
    assert os_ext.samefile(os.path.join(prefix_name, 'broken'),
                           os.path.join(prefix_name, 'broken1'))


def test_is_interactive(monkeypatch):
    # Set `sys.ps1` to immitate an interactive session
    monkeypatch.setattr(sys, 'ps1', 'rfm>>> ', raising=False)
    assert os_ext.is_interactive()


def test_is_url():
    repo_https = 'https://github.com/eth-cscs/reframe.git'
    repo_ssh = 'git@github.com:eth-cscs/reframe.git'
    assert os_ext.is_url(repo_https)
    assert not os_ext.is_url(repo_ssh)


def test_git_repo_hash(monkeypatch):
    # A git branch hash consists of 8(short) or 40 characters.
    assert len(os_ext.git_repo_hash()) == 8
    assert len(os_ext.git_repo_hash(short=False)) == 40
    assert os_ext.git_repo_hash(branch='invalid') is None
    assert os_ext.git_repo_hash(branch='') is None

    # Imitate a system with no git installed by emptying the PATH
    monkeypatch.setenv('PATH', '')
    assert os_ext.git_repo_hash() is None


def test_git_repo_exists():
    assert os_ext.git_repo_exists('https://github.com/eth-cscs/reframe.git',
                                  timeout=3)
    assert not os_ext.git_repo_exists('reframe.git', timeout=3)
    assert not os_ext.git_repo_exists('https://github.com/eth-cscs/xxx',
                                      timeout=3)


def test_force_remove_file(tmp_path):
    fp = tmp_path / 'tmp_file'
    fp.touch()
    fp_name = str(fp)

    assert os.path.exists(fp_name)
    os_ext.force_remove_file(fp_name)
    assert not os.path.exists(fp_name)

    # Try to remove a non-existent file
    os_ext.force_remove_file(fp_name)


def test_expandvars_dollar():
    text = 'Hello, $(echo World)'
    assert 'Hello, World' == os_ext.expandvars(text)

    # Test nested expansion
    text = '$(echo Hello, $(echo World))'
    assert 'Hello, World' == os_ext.expandvars(text)


def test_expandvars_backticks():
    text = 'Hello, `echo World`'
    assert 'Hello, World' == os_ext.expandvars(text)

    # Test nested expansion
    text = '`echo Hello, `echo World``'
    assert 'Hello, World' == os_ext.expandvars(text)


def test_expandvars_mixed_syntax():
    text = '`echo Hello, $(echo World)`'
    assert 'Hello, World' == os_ext.expandvars(text)

    text = '$(echo Hello, `echo World`)'
    assert 'Hello, World' == os_ext.expandvars(text)


def test_expandvars_error():
    text = 'Hello, $(foo)'
    with pytest.raises(SpawnedProcessError):
        os_ext.expandvars(text)


def test_strange_syntax():
    text = 'Hello, $(foo`'
    assert 'Hello, $(foo`' == os_ext.expandvars(text)

    text = 'Hello, `foo)'
    assert 'Hello, `foo)' == os_ext.expandvars(text)


def test_expandvars_nocmd(monkeypatch):
    monkeypatch.setenv('FOO', 'World')
    text = 'Hello, $FOO'
    assert 'Hello, World' == os_ext.expandvars(text)

    text = 'Hello, ${FOO}'
    assert 'Hello, World' == os_ext.expandvars(text)


@pytest.fixture
def direntries(tmp_path):
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
    prefix = tmp_path / 'prefix'
    target = tmp_path / 'target'
    prefix.mkdir()
    target.mkdir()

    (prefix / 'bar').mkdir(parents=True)
    (prefix / 'foo').mkdir(parents=True)
    (prefix / 'bar' / 'bar.txt').touch()
    (prefix / 'bar' / 'foo.txt').touch()
    (prefix / 'bar' / 'foobar.txt').touch()
    (prefix / 'foo' / 'bar.txt').touch()
    (prefix / 'bar.txt').touch()
    (prefix / 'foo.txt').touch()

    # Create also a subdirectory in target, so as to check the recursion
    (target / 'foo').mkdir(parents=True)
    return prefix.resolve(), target.resolve()


def assert_target_directory(src_prefix, dst_prefix, file_links=[]):
    '''Verify the directory structure'''
    assert os.path.exists(dst_prefix / 'bar' / 'bar.txt')
    assert os.path.exists(dst_prefix / 'bar' / 'foo.txt')
    assert os.path.exists(dst_prefix / 'bar' / 'foobar.txt')
    assert os.path.exists(dst_prefix / 'foo' / 'bar.txt')
    assert os.path.exists(dst_prefix / 'bar.txt')
    assert os.path.exists(dst_prefix / 'foo.txt')

    # Verify the symlinks
    for lf in file_links:
        target_link_name = os.path.abspath(src_prefix / lf)
        link_name = os.path.abspath(dst_prefix / lf)
        assert os.path.islink(link_name)
        assert target_link_name == os.readlink(link_name)


def test_virtual_copy_nolinks(direntries):
    os_ext.copytree_virtual(*direntries, dirs_exist_ok=True)
    assert_target_directory(*direntries)


def test_virtual_copy_nolinks_dirs_exist(direntries):
    with pytest.raises(FileExistsError):
        os_ext.copytree_virtual(*direntries)


def test_virtual_copy_valid_links(direntries):
    file_links = ['bar/', 'foo/bar.txt', 'foo.txt']
    os_ext.copytree_virtual(*direntries, file_links, dirs_exist_ok=True)
    assert_target_directory(*direntries, file_links)


def test_virtual_copy_inexistent_links(direntries):
    file_links = ['foobar/', 'foo/bar.txt', 'foo.txt']
    with pytest.raises(ValueError):
        os_ext.copytree_virtual(*direntries, file_links, dirs_exist_ok=True)


def test_virtual_copy_absolute_paths(direntries):
    file_links = [direntries[0] / 'bar', 'foo/bar.txt', 'foo.txt']
    with pytest.raises(ValueError):
        os_ext.copytree_virtual(*direntries, file_links, dirs_exist_ok=True)


def test_virtual_copy_irrelevant_paths(direntries):
    file_links = ['/bin', 'foo/bar.txt', 'foo.txt']
    with pytest.raises(ValueError):
        os_ext.copytree_virtual(*direntries, file_links, dirs_exist_ok=True)

    file_links = [os.path.dirname(direntries[0]), 'foo/bar.txt', 'foo.txt']
    with pytest.raises(ValueError):
        os_ext.copytree_virtual(*direntries, file_links, dirs_exist_ok=True)


def test_virtual_copy_linkself(direntries):
    file_links = ['.']
    with pytest.raises(ValueError):
        os_ext.copytree_virtual(*direntries, file_links, dirs_exist_ok=True)


def test_virtual_copy_linkparent(direntries):
    file_links = ['..']
    with pytest.raises(ValueError):
        os_ext.copytree_virtual(*direntries, file_links, dirs_exist_ok=True)


def test_import_from_file_load_relpath():
    module = util.import_module_from_file('reframe/__init__.py')
    assert reframe.VERSION == module.VERSION
    assert 'reframe' == module.__name__
    assert module is sys.modules.get('reframe')


def test_import_from_file_load_directory():
    module = util.import_module_from_file('reframe')
    assert reframe.VERSION == module.VERSION
    assert 'reframe' == module.__name__
    assert module is sys.modules.get('reframe')


def test_import_from_file_load_abspath():
    filename = os.path.abspath('reframe/__init__.py')
    module = util.import_module_from_file(filename)
    assert reframe.VERSION == module.VERSION
    assert 'reframe' == module.__name__
    assert module is sys.modules.get('reframe')


def test_import_from_file_load_unknown_path():
    try:
        util.import_module_from_file('/foo')
        pytest.fail()
    except ImportError as e:
        assert 'foo' == e.name
        assert '/foo' == e.path


def test_import_from_file_load_directory_relative():
    with os_ext.change_dir('reframe'):
        module = util.import_module_from_file('../reframe')
        assert reframe.VERSION == module.VERSION
        assert 'reframe' == module.__name__
        assert module is sys.modules.get('reframe')


def test_import_from_file_load_relative():
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


def test_import_from_file_load_outside_pkg():
    module = util.import_module_from_file(os.path.__file__)

    # os imports the OS-specific path libraries under the name `path`. Our
    # importer will import the actual file, thus the module name should be
    # the real one.
    assert (module is sys.modules.get('posixpath') or
            module is sys.modules.get('ntpath') or
            module is sys.modules.get('macpath'))


def test_import_from_file_load_twice():
    filename = os.path.abspath('reframe')
    module1 = util.import_module_from_file(filename)
    module2 = util.import_module_from_file(filename)
    assert module1 is module2


def test_import_from_file_load_namespace_package():
    module = util.import_module_from_file('unittests/resources')
    assert 'unittests' in sys.modules
    assert 'unittests.resources' in sys.modules


def test_ppretty_simple_types():
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


def test_ppretty_mixed_types():
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


def test_ppretty_obj_print():
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


def test_change_dir_working(tmpdir):
    wd_save = os.getcwd()
    with os_ext.change_dir(tmpdir):
        assert os.getcwd() == tmpdir

    assert os.getcwd() == wd_save


def test_exception_propagation(tmpdir):
    wd_save = os.getcwd()
    try:
        with os_ext.change_dir(tmpdir):
            raise RuntimeError
    except RuntimeError:
        assert os.getcwd() == wd_save
    else:
        pytest.fail('exception not propagated by the ctx manager')


def test_allx():
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


def test_decamelize():
    assert '' == util.decamelize('')
    assert 'my_base_class' == util.decamelize('MyBaseClass')
    assert 'my_base_class12' == util.decamelize('MyBaseClass12')
    assert 'my_class_a' == util.decamelize('MyClass_A')
    assert 'my_class' == util.decamelize('my_class')
    with pytest.raises(TypeError):
        util.decamelize(None)

    with pytest.raises(TypeError):
        util.decamelize(12)


def test_sanitize():
    assert '' == util.toalphanum('')
    assert 'ab12' == util.toalphanum('ab12')
    assert 'ab1_2' == util.toalphanum('ab1_2')
    assert 'ab1__2' == util.toalphanum('ab1**2')
    assert 'ab__12_' == util.toalphanum('ab (12)')
    with pytest.raises(TypeError):
        util.toalphanum(None)

    with pytest.raises(TypeError):
        util.toalphanum(12)


def test_scoped_dict_construction():
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


def test_scoped_dict_contains():
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


def test_scoped_dict_iter_keys():
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


def test_scoped_dict_iter_items():
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


def test_scoped_dict_iter_values():
    scoped_dict = reframe.utility.ScopedDict({
        'a': {'k1': 1, 'k2': 2},
        'a:b': {'k1': 3, 'k3': 4},
        'a:b:c': {'k2': 5, 'k3': 6},
        '*': {'k1': 7, 'k3': 9, 'k4': 10}
    })

    expected_values = [1, 2, 3, 4, 5, 6, 7, 9, 10]
    assert expected_values == sorted(v for v in scoped_dict.values())


def test_scoped_dict_key_resolution():
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


def test_scoped_dict_setitem():
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


def test_scoped_dict_delitem():
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


def test_scoped_dict_scope_key_name_pseudoconflict():
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


def test_scoped_dict_update():
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


def test_sequence_view():
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


def test_mapping_view():
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


@pytest.fixture
def random_seed():
    random.seed(1)


def test_shortest_sequence():
    s0 = 'abcde'
    s1 = [1, 2, 3]
    assert util.shortest(s0, s1) == s1
    assert id(util.shortest(s0, s1)) == id(s1)
    assert util.shortest(s0, s0) == s0
    with pytest.raises(TypeError):
        util.shortest(12)

    with pytest.raises(TypeError):
        util.shortest(x for x in range(10))

    with pytest.raises(TypeError):
        util.shortest([1], 2)


def test_longest_sequence():
    s0 = 'abcde'
    s1 = [1, 2, 3]
    assert util.longest(s0, s1) == s0
    assert id(util.longest(s0, s1)) == id(s0)
    assert util.longest(s0, s0) == s0
    with pytest.raises(TypeError):
        util.longest(12)

    with pytest.raises(TypeError):
        util.longest(x for x in range(10))

    with pytest.raises(TypeError):
        util.longest([1], 2)


def test_ordered_set_construction(random_seed):
    l = list(range(10))
    random.shuffle(l)

    s = util.OrderedSet(l + l)
    assert len(s) == 10
    for i in range(10):
        assert i in s

    assert list(s) == l


def test_ordered_set_construction_empty():
    s = util.OrderedSet()
    assert s == set()
    assert set() == s


def test_ordered_set_str(random_seed):
    l = list(range(10))
    random.shuffle(l)

    s = util.OrderedSet(l)
    assert str(s) == str(l).replace('[', '{').replace(']', '}')

    s = util.OrderedSet()
    assert str(s) == type(s).__name__ + '()'


def test_ordered_set_construction_error():
    with pytest.raises(TypeError):
        s = util.OrderedSet(2)

    with pytest.raises(TypeError):
        s = util.OrderedSet(1, 2, 3)


def test_ordered_set_repr():
    assert repr(util.OrderedSet('abc')) == "{'a', 'b', 'c'}"
    assert str(util.OrderedSet('abc'))  == "{'a', 'b', 'c'}"


def test_ordered_set_operators():
    s0 = util.OrderedSet('abc')
    s1 = util.OrderedSet('abced')
    s2 = util.OrderedSet('ed')

    assert s0 == set('abc')
    assert s0 == util.OrderedSet('abc')
    assert set('abc') == s0
    assert util.OrderedSet('abc') == s0
    assert s0 != s1
    assert s1 != s0
    assert s0 != util.OrderedSet('cab')

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


def test_ordered_set_union(random_seed):
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


def test_ordered_set_intersection(random_seed):
    l0 = list(range(10, 40))
    l1 = list(range(20, 40))
    l2 = list(range(20, 30))
    random.shuffle(l0)
    random.shuffle(l1)
    random.shuffle(l2)

    s0 = util.OrderedSet(l0)
    s1 = util.OrderedSet(l1)
    s2 = util.OrderedSet(l2)

    # OrderedSet must keep the order of elements in s0
    assert list(s0.intersection(s1, s2)) == [x for x in l0
                                             if x >= 20 and x < 30]


def test_ordered_set_difference():
    l0 = list(range(10, 40))
    l1 = list(range(20, 40))
    l2 = list(range(20, 30))
    random.shuffle(l0)
    random.shuffle(l1)
    random.shuffle(l2)

    s0 = util.OrderedSet(l0)
    s1 = util.OrderedSet(l1)
    s2 = util.OrderedSet(l2)

    # OrderedSet must keep the order of elements in s0
    assert list(s0.difference(s1, s2)) == [x for x in l0 if x >= 10 and x < 20]


def test_ordered_set_reversed():
    l = list(range(10))
    random.shuffle(l)

    s = util.OrderedSet(l)
    assert list(reversed(s)) == list(reversed(l))


def test_concat_files(tmpdir):
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


def test_unique_abs_paths():
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

    monkeypatch.setenv('MODULERCFILE', str(rcfile))
    assert os_ext.cray_cdt_version() == '20.06'


def test_cray_cdt_version_unknown_fmt(tmp_path, monkeypatch):
    # Mock up a CDT file
    rcfile = tmp_path / 'rcfile'
    with open(rcfile, 'w') as fp:
        fp.write('random stuff')

    monkeypatch.setenv('MODULERCFILE', str(rcfile))
    assert os_ext.cray_cdt_version() is None


def test_cray_cdt_version_empty_file(tmp_path, monkeypatch):
    # Mock up a CDT file
    rcfile = tmp_path / 'rcfile'
    rcfile.touch()
    monkeypatch.setenv('MODULERCFILE', str(rcfile))
    assert os_ext.cray_cdt_version() is None


def test_cray_cdt_version_no_such_file(tmp_path, monkeypatch):
    # Mock up a CDT file
    rcfile = tmp_path / 'rcfile'
    monkeypatch.setenv('MODULERCFILE', str(rcfile))
    assert os_ext.cray_cdt_version() is None


def test_cray_cle_info(tmp_path):
    # Mock up a CLE release
    cle_info_file = tmp_path / 'cle-release'
    with open(cle_info_file, 'w') as fp:
        fp.write('RELEASE=7.0.UP01\n'
                 'BUILD=7.0.1227\n'
                 'DATE=20200326\n'
                 'ARCH=noarch\n'
                 'NETWORK=ari\n'
                 'PATCHSET=09-202003261814\n')

    cle_info = os_ext.cray_cle_info(cle_info_file)
    assert cle_info.release == '7.0.UP01'
    assert cle_info.build == '7.0.1227'
    assert cle_info.date == '20200326'
    assert cle_info.network == 'ari'
    assert cle_info.patchset == '09'


def test_cray_cle_info_no_such_file(tmp_path):
    cle_info_file = tmp_path / 'cle-release'
    assert os_ext.cray_cle_info(cle_info_file) is None


def test_cray_cle_info_missing_parts(tmp_path):
    # Mock up a CLE release
    cle_info_file = tmp_path / 'cle-release'
    with open(cle_info_file, 'w') as fp:
        fp.write('RELEASE=7.0.UP01\n'
                 'PATCHSET=09-202003261814\n')

    cle_info = os_ext.cray_cle_info(cle_info_file)
    assert cle_info.release == '7.0.UP01'
    assert cle_info.build is None
    assert cle_info.date is None
    assert cle_info.network is None
    assert cle_info.patchset == '09'


@pytest.fixture
def temp_runtime(tmp_path):
    def _temp_runtime(site_config, system=None, options={}):
        options.update({'systems/prefix': tmp_path})
        with rt.temp_runtime(site_config, system, options):
            yield rt.runtime

    yield _temp_runtime


@pytest.fixture(params=['tmod', 'tmod4', 'lmod', 'nomod'])
def runtime_with_modules(request, temp_runtime):
    if fixtures.USER_CONFIG_FILE:
        config_file, system = fixtures.USER_CONFIG_FILE, fixtures.USER_SYSTEM
    else:
        config_file, system = fixtures.BUILTIN_CONFIG_FILE, 'generic'

    try:
        yield from temp_runtime(config_file, system,
                                {'systems/modules_system': request.param})
    except ConfigError as e:
        pytest.skip(str(e))


def test_find_modules(monkeypatch, runtime_with_modules):
    # Pretend to be on a clean modules environment
    monkeypatch.setenv('MODULEPATH', '')
    monkeypatch.setenv('LOADEDMODULES', '')
    monkeypatch.setenv('_LMFILES_', '')

    ms = rt.runtime().system.modules_system
    ms.searchpath_add(fixtures.TEST_MODULES)
    found_modules = list(util.find_modules('testmod'))
    if ms.name == 'nomod':
        assert found_modules == []
    else:
        assert found_modules == [
            ('generic:default', 'builtin', 'testmod_bar'),
            ('generic:default', 'builtin', 'testmod_base'),
            ('generic:default', 'builtin', 'testmod_boo'),
            ('generic:default', 'builtin', 'testmod_foo')
        ]


def test_find_modules_toolchains(monkeypatch, runtime_with_modules):
    # Pretend to be on a clean modules environment
    monkeypatch.setenv('MODULEPATH', '')
    monkeypatch.setenv('LOADEDMODULES', '')
    monkeypatch.setenv('_LMFILES_', '')

    ms = rt.runtime().system.modules_system
    ms.searchpath_add(fixtures.TEST_MODULES)
    found_modules = list(
        util.find_modules('testmod',
                          toolchain_mapping={r'.*_ba.*': 'builtin',
                                             r'testmod_foo': 'foo'})
    )
    if ms.name == 'nomod':
        assert found_modules == []
    else:
        assert found_modules == [
            ('generic:default', 'builtin', 'testmod_bar'),
            ('generic:default', 'builtin', 'testmod_base')
        ]
