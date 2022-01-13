# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import pytest
import random
import sys
import time

import reframe
import reframe.core.fields as fields
import reframe.core.runtime as rt
import reframe.utility as util
import reframe.utility.jsonext as jsonext
import reframe.utility.osext as osext
import reframe.utility.sanity as sn
import unittests.utility as test_util

from reframe.core.exceptions import (ConfigError,
                                     SpawnedProcessError,
                                     SpawnedProcessTimeout)


def test_command_success():
    completed = osext.run_command('echo foobar')
    assert completed.returncode == 0
    assert completed.stdout == 'foobar\n'


def test_command_success_cmd_seq():
    completed = osext.run_command(['echo', 'foobar'])
    assert completed.returncode == 0
    assert completed.stdout == 'foobar\n'


def test_command_error():
    with pytest.raises(SpawnedProcessError,
                       match=r"command 'false' failed with exit code 1"):
        osext.run_command('false', check=True)


def test_command_error_cmd_seq():
    with pytest.raises(SpawnedProcessError,
                       match=r"command 'false' failed with exit code 1"):
        osext.run_command(['false'], check=True)


def test_command_timeout():
    with pytest.raises(
        SpawnedProcessTimeout, match=r"command 'sleep 3' timed out "
                                     r'after 2s') as exc_info:

        osext.run_command('sleep 3', timeout=2)

    assert exc_info.value.timeout == 2

    # Try to get the string repr. of the exception: see bug #658
    str(exc_info.value)


def test_command_async():
    t_launch = time.time()
    t_sleep  = t_launch
    proc = osext.run_command_async('sleep 1')
    t_launch = time.time() - t_launch

    proc.wait()
    t_sleep = time.time() - t_sleep

    # Now check the timings
    assert t_launch < 1
    assert t_sleep >= 1


def test_copytree(tmp_path):
    dir_src = tmp_path / 'src'
    dir_src.mkdir()
    dir_dst = tmp_path / 'dst'
    dir_dst.mkdir()
    osext.copytree(str(dir_src), str(dir_dst), dirs_exist_ok=True)


def test_copytree_src_parent_of_dst(tmp_path):
    dst_path = tmp_path / 'dst'
    src_path = (dst_path / '..').resolve()

    with pytest.raises(ValueError):
        osext.copytree(str(src_path), str(dst_path))


@pytest.fixture(params=['dirs_exist_ok=True', 'dirs_exist_ok=False'])
def dirs_exist_ok(request):
    return 'True' in request.param


def test_copytree_dst_notdir(tmp_path, dirs_exist_ok):
    dir_src = tmp_path / 'src'
    dir_src.mkdir()
    dst = tmp_path / 'dst'
    dst.touch()
    with pytest.raises(FileExistsError, match=fr'{dst}'):
        osext.copytree(str(dir_src), str(dst), dirs_exist_ok=dirs_exist_ok)


def test_copytree_src_notdir(tmp_path, dirs_exist_ok):
    src = tmp_path / 'src'
    src.touch()
    dst = tmp_path / 'dst'
    dst.mkdir()
    with pytest.raises(NotADirectoryError, match=fr'{src}'):
        osext.copytree(str(src), str(dst), dirs_exist_ok=dirs_exist_ok)


def test_copytree_src_does_not_exist(tmp_path, dirs_exist_ok):
    src = tmp_path / 'src'
    dst = tmp_path / 'dst'
    dst.mkdir()
    with pytest.raises(FileNotFoundError, match=fr'{src}'):
        osext.copytree(str(src), str(dst), dirs_exist_ok=dirs_exist_ok)


@pytest.fixture
def rmtree(tmp_path):
    testdir = tmp_path / 'test'
    testdir.mkdir()
    with open(os.path.join(str(testdir), 'foo.txt'), 'w') as fp:
        fp.write('hello\n')

    def _rmtree(*args, **kwargs):
        osext.rmtree(testdir, *args, **kwargs)
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
        osext.rmtree(testdir)


def test_inpath():
    assert osext.inpath('/foo/bin', '/bin:/foo/bin:/usr/bin')
    assert not osext.inpath('/foo/bin', '/bin:/usr/local/bin')


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

    returned_subdirs = osext.subdirs(prefix_name)
    assert [prefix_name] == returned_subdirs

    returned_subdirs = osext.subdirs(prefix_name, recurse=True)
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

    assert osext.samefile('/foo', '/foo')
    assert osext.samefile('/foo', '/foo/')
    assert osext.samefile('/foo/bar', '/foo//bar/')
    assert osext.samefile(os.path.join(prefix_name, 'foo'),
                          os.path.join(prefix_name, 'foolnk'))
    assert osext.samefile(os.path.join(prefix_name, 'foo'),
                          os.path.join(prefix_name, 'foolnk1'))
    assert not osext.samefile('/foo', '/bar')
    assert osext.samefile('/foo', os.path.join(prefix_name, 'broken'))
    assert osext.samefile(os.path.join(prefix_name, 'broken'),
                          os.path.join(prefix_name, 'broken1'))


def test_is_interactive(monkeypatch):
    # Set `sys.ps1` to immitate an interactive session
    monkeypatch.setattr(sys, 'ps1', 'rfm>>> ', raising=False)
    assert osext.is_interactive()


def test_is_url():
    repo_https = 'https://github.com/eth-cscs/reframe.git'
    repo_ssh = 'git@github.com:eth-cscs/reframe.git'
    assert osext.is_url(repo_https)
    assert not osext.is_url(repo_ssh)


@pytest.fixture
def git_only():
    try:
        osext.run_command('git --version', check=True, log=False)
    except (SpawnedProcessError, FileNotFoundError):
        pytest.skip('no git installation found on system')

    try:
        osext.run_command('git status', check=True, log=False)
    except (SpawnedProcessError, FileNotFoundError):
        pytest.skip('not inside a git repository')


def test_git_repo_hash(git_only):
    # A git branch hash consists of 8(short) or 40 characters.
    assert len(osext.git_repo_hash()) == 8
    assert len(osext.git_repo_hash(short=False)) == 40
    assert osext.git_repo_hash(commit='invalid') is None
    assert osext.git_repo_hash(commit='') is None


def test_git_repo_hash_no_git(git_only, monkeypatch):
    # Emulate a system with no git installed
    monkeypatch.setenv('PATH', '')
    assert osext.git_repo_hash() is None


def test_git_repo_hash_no_git_repo(git_only, monkeypatch, tmp_path):
    # Emulate trying to get the hash from somewhere where there is no repo
    monkeypatch.setenv('GIT_DIR', str(tmp_path))
    assert osext.git_repo_hash() is None


def test_git_repo_exists(git_only):
    assert osext.git_repo_exists('https://github.com/eth-cscs/reframe.git',
                                 timeout=10)
    assert not osext.git_repo_exists('reframe.git', timeout=10)
    assert not osext.git_repo_exists('https://github.com/eth-cscs/xxx',
                                     timeout=10)


def test_force_remove_file(tmp_path):
    fp = tmp_path / 'tmp_file'
    fp.touch()
    fp_name = str(fp)

    assert os.path.exists(fp_name)
    osext.force_remove_file(fp_name)
    assert not os.path.exists(fp_name)

    # Try to remove a non-existent file
    osext.force_remove_file(fp_name)


def test_expandvars_dollar():
    text = 'Hello, $(echo World)'
    assert 'Hello, World' == osext.expandvars(text)

    # Test nested expansion
    text = '$(echo Hello, $(echo World))'
    assert 'Hello, World' == osext.expandvars(text)


def test_expandvars_backticks():
    text = 'Hello, `echo World`'
    assert 'Hello, World' == osext.expandvars(text)

    # Test nested expansion
    text = '`echo Hello, `echo World``'
    assert 'Hello, World' == osext.expandvars(text)


def test_expandvars_mixed_syntax():
    text = '`echo Hello, $(echo World)`'
    assert 'Hello, World' == osext.expandvars(text)

    text = '$(echo Hello, `echo World`)'
    assert 'Hello, World' == osext.expandvars(text)


def test_expandvars_error():
    text = 'Hello, $(foo)'
    with pytest.raises(SpawnedProcessError):
        osext.expandvars(text)


def test_strange_syntax():
    text = 'Hello, $(foo`'
    assert 'Hello, $(foo`' == osext.expandvars(text)

    text = 'Hello, `foo)'
    assert 'Hello, `foo)' == osext.expandvars(text)


def test_expandvars_nocmd(monkeypatch):
    monkeypatch.setenv('FOO', 'World')
    text = 'Hello, $FOO'
    assert 'Hello, World' == osext.expandvars(text)

    text = 'Hello, ${FOO}'
    assert 'Hello, World' == osext.expandvars(text)


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
    osext.copytree_virtual(*direntries, dirs_exist_ok=True)
    assert_target_directory(*direntries)


def test_virtual_copy_nolinks_dirs_exist(direntries):
    with pytest.raises(FileExistsError):
        osext.copytree_virtual(*direntries)


def test_virtual_copy_valid_links(direntries):
    file_links = ['bar/', 'foo/bar.txt', 'foo.txt']
    osext.copytree_virtual(*direntries, file_links, dirs_exist_ok=True)
    assert_target_directory(*direntries, file_links)


def test_virtual_copy_inexistent_links(direntries):
    file_links = ['foobar/', 'foo/bar.txt', 'foo.txt']
    with pytest.raises(ValueError):
        osext.copytree_virtual(*direntries, file_links, dirs_exist_ok=True)


def test_virtual_copy_absolute_paths(direntries):
    file_links = [direntries[0] / 'bar', 'foo/bar.txt', 'foo.txt']
    with pytest.raises(ValueError):
        osext.copytree_virtual(*direntries, file_links, dirs_exist_ok=True)


def test_virtual_copy_irrelevant_paths(direntries):
    file_links = ['/bin', 'foo/bar.txt', 'foo.txt']
    with pytest.raises(ValueError):
        osext.copytree_virtual(*direntries, file_links, dirs_exist_ok=True)

    file_links = [os.path.dirname(direntries[0]), 'foo/bar.txt', 'foo.txt']
    with pytest.raises(ValueError):
        osext.copytree_virtual(*direntries, file_links, dirs_exist_ok=True)


def test_virtual_copy_linkself(direntries):
    file_links = ['.']
    with pytest.raises(ValueError):
        osext.copytree_virtual(*direntries, file_links, dirs_exist_ok=True)


def test_virtual_copy_linkparent(direntries):
    file_links = ['..']
    with pytest.raises(ValueError):
        osext.copytree_virtual(*direntries, file_links, dirs_exist_ok=True)


@pytest.fixture(params=['symlinks=True', 'symlinks=False'])
def symlinks(request):
    return 'True' in request.param


def test_virtual_copy_symlinks_dirs_exist(tmp_path, symlinks):
    src = tmp_path / 'src'
    src.mkdir()
    dst = tmp_path / 'dst'
    dst.mkdir()
    foo = src / 'foo'
    foo.touch()
    foo_link = src / 'foo.link'
    foo_link.symlink_to(foo)
    osext.copytree_virtual(src, dst, symlinks=symlinks, dirs_exist_ok=True)
    assert (dst / 'foo').exists()
    assert (dst / 'foo.link').exists()
    assert (dst / 'foo.link').is_symlink() == symlinks


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
    with osext.change_dir('reframe'):
        module = util.import_module_from_file('../reframe')
        assert reframe.VERSION == module.VERSION
        assert 'reframe' == module.__name__
        assert module is sys.modules.get('reframe')


def test_import_from_file_load_relative():
    with osext.change_dir('reframe'):
        # Load a module from a directory up
        module = util.import_module_from_file('../reframe/__init__.py')
        assert reframe.VERSION == module.VERSION
        assert 'reframe' == module.__name__
        assert module is sys.modules.get('reframe')

        # Load a module from the current directory
        module = util.import_module_from_file('utility/osext.py')
        assert 'reframe.utility.osext' == module.__name__
        assert module is sys.modules.get('reframe.utility.osext')


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
    util.import_module_from_file('unittests/resources')
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


def test_attrs():
    class B:
        z = fields.TypedField(int)

        def __init__(self, x, y):
            self.x = x
            self.y = y

    class C(B):
        def __init__(self, x, y):
            self._x = x
            self.y = y
            self.z = 3

        def foo():
            pass

        @property
        def x(self):
            return self._x

    class D(C):
        pass

    # Test undefined descriptors are not returned
    b = B(-1, 0)
    b_attrs = util.attrs(b)
    assert b_attrs['x'] == -1
    assert b_attrs['y'] == 0
    assert 'z' not in b_attrs

    c = C(1, 2)
    c_attrs = util.attrs(c)
    assert c_attrs['x'] == 1
    assert c_attrs['y'] == 2
    assert c_attrs['z'] == 3
    assert 'foo' not in c_attrs

    # Test inherited attributes
    d = D(4, 5)
    d_attrs = util.attrs(d)
    assert d_attrs['x'] == 4
    assert d_attrs['y'] == 5
    assert d_attrs['z'] == 3
    assert 'foo' not in d_attrs


def test_change_dir_working(tmpdir):
    wd_save = os.getcwd()
    with osext.change_dir(tmpdir):
        assert os.getcwd() == tmpdir

    assert os.getcwd() == wd_save


def test_exception_propagation(tmpdir):
    wd_save = os.getcwd()
    try:
        with osext.change_dir(tmpdir):
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

    # Scopes must be requested with scope()
    assert scoped_dict.scope('a') == {'k1': 1, 'k2': 2, 'k3': 9, 'k4': 10}
    assert scoped_dict.scope('a:b') == {'k1': 3, 'k2': 2, 'k3': 4, 'k4': 10}
    assert scoped_dict.scope('a:b:c') == {'k1': 3, 'k2': 5, 'k3': 6, 'k4': 10}
    assert scoped_dict.scope('*') == {'k1': 7, 'k3': 9, 'k4': 10}

    # This is resolved in scope 'a'
    assert scoped_dict.scope('a:z') == {'k1': 1, 'k2': 2, 'k3': 9, 'k4': 10}
    assert scoped_dict.scope(None) == {}


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
    assert isinstance(m, list)

    m_orig = m = util.SequenceView([1])
    m += [3, 4]
    assert m is not m_orig
    assert [1] == m_orig
    assert [1, 3, 4] == m
    assert isinstance(m, list)

    n = m + l
    assert [1, 3, 4, 1, 2, 2] == n
    assert isinstance(n, list)

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
        l.copy()

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
        util.OrderedSet(2)

    with pytest.raises(TypeError):
        util.OrderedSet(1, 2, 3)


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
    with osext.change_dir(tmpdir):
        file1 = 'in1.txt'
        file2 = 'in2.txt'
        concat_file = 'out.txt'
        with open(file1, 'w') as f1:
            f1.write('Hello1')

        with open(file2, 'w') as f2:
            f2.write('Hello2')

        osext.concat_files(concat_file, file1, file2, overwrite=True)
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
    actual_paths = osext.unique_abs_paths(
        [p1, p2, p3, p4, p5])
    assert expected_paths == actual_paths

    expected_paths = [os.path.abspath('a/b/c'),  os.path.abspath('a/b'),
                      '/d/e', '/d/e/f']
    actual_paths = osext.unique_abs_paths(
        [p1, p2, p3, p4, p5], prune_children=False)
    assert expected_paths == actual_paths

    with pytest.raises(TypeError):
        osext.unique_abs_paths(None)


def test_cray_cdt_version(tmp_path, monkeypatch):
    # Mock up a CDT file
    rcfile = tmp_path / 'rcfile'
    with open(rcfile, 'w') as fp:
        fp.write('#%Module CDT 20.06\nblah blah\n')

    monkeypatch.setenv('MODULERCFILE', str(rcfile))
    assert osext.cray_cdt_version() == '20.06'


def test_cray_cdt_version_unknown_fmt(tmp_path, monkeypatch):
    # Mock up a CDT file
    rcfile = tmp_path / 'rcfile'
    with open(rcfile, 'w') as fp:
        fp.write('random stuff')

    monkeypatch.setenv('MODULERCFILE', str(rcfile))
    assert osext.cray_cdt_version() is None


def test_cray_cdt_version_empty_file(tmp_path, monkeypatch):
    # Mock up a CDT file
    rcfile = tmp_path / 'rcfile'
    rcfile.touch()
    monkeypatch.setenv('MODULERCFILE', str(rcfile))
    assert osext.cray_cdt_version() is None


def test_cray_cdt_version_no_such_file(tmp_path, monkeypatch):
    # Mock up a CDT file
    rcfile = tmp_path / 'rcfile'
    monkeypatch.setenv('MODULERCFILE', str(rcfile))
    assert osext.cray_cdt_version() is None


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

    cle_info = osext.cray_cle_info(cle_info_file)
    assert cle_info.release == '7.0.UP01'
    assert cle_info.build == '7.0.1227'
    assert cle_info.date == '20200326'
    assert cle_info.network == 'ari'
    assert cle_info.patchset == '09'


def test_cray_cle_info_no_such_file(tmp_path):
    cle_info_file = tmp_path / 'cle-release'
    assert osext.cray_cle_info(cle_info_file) is None


def test_cray_cle_info_missing_parts(tmp_path):
    # Mock up a CLE release
    cle_info_file = tmp_path / 'cle-release'
    with open(cle_info_file, 'w') as fp:
        fp.write('RELEASE=7.0.UP01\n'
                 'PATCHSET=09-202003261814\n')

    cle_info = osext.cray_cle_info(cle_info_file)
    assert cle_info.release == '7.0.UP01'
    assert cle_info.build is None
    assert cle_info.date is None
    assert cle_info.network is None
    assert cle_info.patchset == '09'


@pytest.fixture(params=['tmod', 'tmod4', 'lmod', 'nomod'])
def user_exec_ctx(request, make_exec_ctx_g):
    if test_util.USER_CONFIG_FILE:
        config_file, system = test_util.USER_CONFIG_FILE, test_util.USER_SYSTEM
    else:
        config_file, system = test_util.BUILTIN_CONFIG_FILE, 'generic'

    try:
        yield from make_exec_ctx_g(config_file, system,
                                   {'systems/modules_system': request.param})
    except ConfigError as e:
        pytest.skip(str(e))


@pytest.fixture
def modules_system(user_exec_ctx, monkeypatch):
    # Pretend to be on a clean modules environment
    monkeypatch.setenv('MODULEPATH', '')
    monkeypatch.setenv('LOADEDMODULES', '')
    monkeypatch.setenv('_LMFILES_', '')

    ms = rt.runtime().system.modules_system
    ms.searchpath_add(test_util.TEST_MODULES)
    yield ms
    ms.searchpath_remove(test_util.TEST_MODULES)


def test_find_modules(modules_system):
    # The test modules will be found as many times as there are partitions and
    # environments in the current system
    current_system = rt.runtime().system
    ntimes = sum(len(p.environs) for p in current_system.partitions)

    found_modules = [m[2] for m in util.find_modules('testmod')]
    if modules_system.name == 'nomod':
        assert found_modules == []
    else:
        assert found_modules == ['testmod_bar', 'testmod_base', 'testmod_boo',
                                 'testmod_ext', 'testmod_foo']*ntimes


def test_find_modules_env_mapping(modules_system):
    # The test modules will be found as many times as there are partitions and
    # environments in the current system
    current_system = rt.runtime().system
    ntimes = sum(len(p.environs) for p in current_system.partitions)

    found_modules = [
        m[2] for m in util.find_modules('testmod',
                                        environ_mapping={
                                            r'.*_ba.*': 'builtin',
                                            r'testmod_foo': 'foo'
                                        })
    ]
    if modules_system.name == 'nomod':
        assert found_modules == []
    else:
        assert found_modules == ['testmod_bar', 'testmod_base']*ntimes


def test_find_modules_errors():
    with pytest.raises(TypeError):
        list(util.find_modules(1))

    with pytest.raises(TypeError):
        list(util.find_modules(None))

    with pytest.raises(TypeError):
        list(util.find_modules('foo', 1))


def test_jsonext_dump(tmp_path):
    json_dump = tmp_path / 'test.json'
    with open(json_dump, 'w') as fp:
        jsonext.dump({'foo': sn.defer(['bar'])}, fp)

    with open(json_dump, 'r') as fp:
        assert '{"foo": null}' == fp.read()

    with open(json_dump, 'w') as fp:
        jsonext.dump({'foo': sn.defer(['bar']).evaluate()}, fp)

    with open(json_dump, 'r') as fp:
        assert '{"foo": ["bar"]}' == fp.read()

    with open(json_dump, 'w') as fp:
        jsonext.dump({'foo': sn.defer(['bar'])}, fp, separators=(',', ':'))

    with open(json_dump, 'r') as fp:
        assert '{"foo":null}' == fp.read()


def test_jsonext_dumps():
    assert '"foo"' == jsonext.dumps('foo')
    assert '{"foo": ["bar"]}' == jsonext.dumps(
        {'foo': sn.defer(['bar']).evaluate()}
    )
    assert '{"foo":["bar"]}' == jsonext.dumps(
        {'foo': sn.defer(['bar']).evaluate()}, separators=(',', ':')
    )
    assert '{"(1, 2, 3)": 1}' == jsonext.dumps({(1, 2, 3): 1})


# Classes to test JSON deserialization

class _D(jsonext.JSONSerializable):
    def __init__(self):
        self.a = 2
        self.b = 'bar'

    def __eq__(self, other):
        if not isinstance(other, _D):
            return NotImplemented

        return self.a == other.a and self.b == other.b


class _Z(_D):
    pass


class _T(jsonext.JSONSerializable):
    __slots__ = ('t',)

    def __eq__(self, other):
        if not isinstance(other, _T):
            return NotImplemented

        return self.t == other.t


class _C(jsonext.JSONSerializable):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.z = None
        self.w = {1, 2}
        self.t = None

        # Dump dict with tuples as keys
        self.v = {(1, 2): 1}

    def __rfm_json_decode__(self, json):
        # Sets are converted to lists when encoding, we need to manually
        # change them back to sets
        self.w = set(json['w'])

    def __eq__(self, other):
        if not isinstance(other, _C):
            return NotImplemented

        return (self.x == other.x and
                self.y == other.y and
                self.z == other.z and
                self.w == other.w and
                self.t == other.t)


def test_jsonext_load(tmp_path):
    c = _C(1, 'foo')
    c.x += 1
    c.y = 'foobar'
    c.z = _Z()
    c.z.a += 1
    c.z.b = 'barfoo'
    c.t = _T()
    c.t.t = 5

    json_dump = tmp_path / 'test.json'
    with open(json_dump, 'w') as fp:
        jsonext.dump(c, fp, indent=2)

    with open(json_dump, 'r') as fp:
        print(fp.read())

    with open(json_dump, 'r') as fp:
        c_restored = jsonext.load(fp)

    assert c == c_restored
    assert c is not c_restored

    # Do the same with dumps() and loads()
    c_restored = jsonext.loads(jsonext.dumps(c))
    assert c == c_restored
    assert c is not c_restored


def test_attr_validator():
    class C:
        def __init__(self):
            self.x = 3
            self.y = [1, 2, 3]
            self.z = {'a': 1, 'b': 2}

    class D:
        def __init__(self):
            self.x = 1
            self.y = C()

    has_no_str = util.attr_validator(lambda x: not isinstance(x, str))

    d = D()
    assert has_no_str(d)[0]

    # Check when a list element does not validate
    d.y.y[1] = 'foo'
    assert has_no_str(d) == (False, 'D.y.y[1]')
    d.y.y[1] = 2

    # Check when a dict element does not validate
    d.y.z['a'] = 'b'
    assert has_no_str(d) == (False, "D.y.z['a']")
    d.y.z['a'] = 1

    # Check when an attribute does not validate
    d.x = 'foo'
    assert has_no_str(d) == (False, 'D.x')
    d.x = 1

    # Check when an attribute does not validate
    d.y.x = 'foo'
    assert has_no_str(d) == (False, 'D.y.x')
    d.y.x = 3

    # Check when an attribute does not validate against a custom type
    has_no_c = util.attr_validator(lambda x: not isinstance(x, C))
    assert has_no_c(d) == (False, 'D.y')


def test_is_picklable():
    class X:
        pass

    x = X()
    assert util.is_picklable(x)
    assert not util.is_picklable(X)

    assert util.is_picklable(1)
    assert util.is_picklable([1, 2])
    assert util.is_picklable((1, 2))
    assert util.is_picklable({1, 2})
    assert util.is_picklable({'a': 1, 'b': 2})

    class Y:
        def __reduce_ex__(self, proto):
            raise TypeError

    y = Y()
    assert not util.is_picklable(y)

    class Z:
        def __reduce__(self):
            return TypeError

    # This is still picklable, because __reduce_ex__() is preferred
    z = Z()
    assert util.is_picklable(z)

    def foo():
        yield

    assert not util.is_picklable(foo)
    assert not util.is_picklable(foo())


def test_is_copyable():
    class X:
        pass

    x = X()
    assert util.is_copyable(x)

    class Y:
        def __copy__(self):
            pass

    y = Y()
    assert util.is_copyable(y)

    class Z:
        def __deepcopy__(self, memo):
            pass

    z = Z()
    assert util.is_copyable(z)

    def foo():
        yield

    assert util.is_copyable(foo)
    assert util.is_copyable(len)
    assert util.is_copyable(int)
    assert not util.is_copyable(foo())


def test_is_trivially_callable():
    def foo():
        pass

    def bar(x, y):
        pass

    assert util.is_trivially_callable(foo)
    assert util.is_trivially_callable(bar, non_def_args=2)
    with pytest.raises(TypeError):
        util.is_trivially_callable(1)


def test_nodelist_abbrev():
    nid_nodes = [f'nid{n:03}' for n in range(5, 20)]
    cid_nodes = [f'cid{n:03}' for n in range(20)]

    random.shuffle(nid_nodes)
    random.shuffle(cid_nodes)
    nid_nodes.insert(0, 'nid002')
    nid_nodes.insert(0, 'nid001')
    nid_nodes.append('nid125')
    cid_nodes += ['cid055', 'cid056']

    all_nodes = nid_nodes + cid_nodes
    random.shuffle(all_nodes)

    nodelist = util.nodelist_abbrev
    assert nodelist(nid_nodes) == 'nid00[1-2],nid0[05-19],nid125'
    assert nodelist(cid_nodes) == 'cid0[00-19],cid05[5-6]'
    assert nodelist(all_nodes) == (
        'cid0[00-19],cid05[5-6],nid00[1-2],nid0[05-19],nid125'
    )

    # Test non-contiguous nodes
    nid_nodes = []
    for i in range(3):
        nid_nodes += [f'nid{n:03}' for n in range(10*i, 10*i+5)]

    random.shuffle(nid_nodes)
    assert nodelist(nid_nodes) == 'nid00[0-4],nid01[0-4],nid02[0-4]'
    assert nodelist(['nid01', 'nid10', 'nid20']) == 'nid01,nid10,nid20'
    assert nodelist([]) == ''
    assert nodelist(['nid001']) == 'nid001'

    # Test host names with numbers in their basename (see GH #2357)
    nodes = [f'c2-01-{n:02}' for n in range(100)]
    assert nodelist(nodes) == 'c2-01-[00-99]'

    # Test node duplicates
    assert nodelist(['nid001', 'nid001', 'nid002']) == 'nid001,nid00[1-2]'

    with pytest.raises(TypeError, match='nodes argument must be a Sequence'):
        nodelist(1)

    with pytest.raises(TypeError, match='nodes argument cannot be a string'):
        nodelist('foo')
