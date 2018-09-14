import os
import shutil
import sys
import tempfile
import unittest

import reframe
import reframe.core.debug as debug
import reframe.core.fields as fields
import reframe.utility as util
import reframe.utility.os_ext as os_ext
from reframe.core.exceptions import (SpawnedProcessError,
                                     SpawnedProcessTimeout)


class TestOSTools(unittest.TestCase):
    def test_command_success(self):
        completed = os_ext.run_command('echo foobar')
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout, 'foobar\n')

    def test_command_error(self):
        self.assertRaises(SpawnedProcessError, os_ext.run_command,
                          'false', 'check=True')

    def test_command_timeout(self):
        try:
            os_ext.run_command('sleep 3', timeout=2)
            self.fail('Expected timeout')
        except SpawnedProcessTimeout as e:
            self.assertEqual(e.timeout, 2)

    def test_command_async(self):
        from datetime import datetime

        t_launch = datetime.now()
        t_sleep  = t_launch
        proc = os_ext.run_command_async('sleep 1')
        t_launch = datetime.now() - t_launch

        proc.wait()
        t_sleep = datetime.now() - t_sleep

        # Now check the timings
        self.assertLess(t_launch.seconds, 1)
        self.assertGreaterEqual(t_sleep.seconds, 1)

    def test_grep(self):
        self.assertTrue(os_ext.grep_command_output(cmd='echo hello',
                                                   pattern='hello'))
        self.assertFalse(os_ext.grep_command_output(cmd='echo hello',
                                                    pattern='foo'))

    def test_copytree(self):
        dir_src = tempfile.mkdtemp()
        dir_dst = tempfile.mkdtemp()

        self.assertRaises(OSError, shutil.copytree, dir_src, dir_dst)
        try:
            os_ext.copytree(dir_src, dir_dst)
        except Exception as e:
            self.fail('custom copytree failed: %s' % e)

        shutil.rmtree(dir_src)
        shutil.rmtree(dir_dst)

    def test_copytree_src_parent_of_dst(self):
        dst_path = tempfile.mkdtemp()
        src_path = os.path.abspath(os.path.join(dst_path, '..'))

        self.assertRaises(ValueError, os_ext.copytree,
                          src_path, dst_path)

        shutil.rmtree(dst_path)

    def _test_rmtree(self, *args, **kwargs):
        testdir = tempfile.mkdtemp()
        with open(os.path.join(testdir, 'foo.txt'), 'w') as fp:
            fp.write('hello\n')

        os_ext.rmtree(testdir, *args, **kwargs)
        self.assertFalse(os.path.exists(testdir))

    def test_rmtree(self):
        self._test_rmtree()

    def test_rmtree_onerror(self):
        self._test_rmtree(onerror=lambda *args: None)

    def test_rmtree_error(self):
        # Try to remove an inexistent directory
        testdir = tempfile.mkdtemp()
        os.rmdir(testdir)
        self.assertRaises(OSError, os_ext.rmtree, testdir)

    def test_inpath(self):
        self.assertTrue(os_ext.inpath('/foo/bin', '/bin:/foo/bin:/usr/bin'))
        self.assertFalse(os_ext.inpath('/foo/bin', '/bin:/usr/local/bin'))

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
        self.assertEqual([prefix], returned_subdirs)

        returned_subdirs = os_ext.subdirs(prefix, recurse=True)
        self.assertEqual(expected_subdirs, set(returned_subdirs))
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

        self.assertTrue(os_ext.samefile('/foo', '/foo'))
        self.assertTrue(os_ext.samefile('/foo', '/foo/'))
        self.assertTrue(os_ext.samefile('/foo/bar', '/foo//bar/'))
        self.assertTrue(os_ext.samefile(os.path.join(prefix, 'foo'),
                                        os.path.join(prefix, 'foolnk')))
        self.assertTrue(os_ext.samefile(os.path.join(prefix, 'foo'),
                                        os.path.join(prefix, 'foolnk1')))
        self.assertFalse(os_ext.samefile('/foo', '/bar'))
        self.assertTrue(os_ext.samefile(
            '/foo', os.path.join(prefix, 'broken')))
        self.assertTrue(os_ext.samefile(os.path.join(prefix, 'broken'),
                                        os.path.join(prefix, 'broken1')))
        shutil.rmtree(prefix)

    def test_is_url(self):
        repo_https = 'https://github.com/eth-cscs/reframe.git'
        repo_ssh = 'git@github.com:eth-cscs/reframe.git'
        self.assertTrue(os_ext.is_url(repo_https))
        self.assertFalse(os_ext.is_url(repo_ssh))

    def test_git_repo_exists(self):
        self.assertTrue(os_ext.git_repo_exists(
            'https://github.com/eth-cscs/reframe.git', timeout=3))
        self.assertFalse(os_ext.git_repo_exists('reframe.git', timeout=3))
        self.assertFalse(os_ext.git_repo_exists(
            'https://github.com/eth-cscs/xxx', timeout=3))

    def test_force_remove_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as fp:
            pass

        self.assertTrue(os.path.exists(fp.name))
        os_ext.force_remove_file(fp.name)
        self.assertFalse(os.path.exists(fp.name))

        # Try to remove a non-existent file
        os_ext.force_remove_file(fp.name)


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

    def verify_target_directory(self, file_links=[]):
        """Verify the directory structure"""
        self.assertTrue(
            os.path.exists(os.path.join(self.target, 'bar', 'bar.txt')))
        self.assertTrue(
            os.path.exists(os.path.join(self.target, 'bar', 'foo.txt')))
        self.assertTrue(
            os.path.exists(os.path.join(self.target, 'bar', 'foobar.txt')))
        self.assertTrue(
            os.path.exists(os.path.join(self.target, 'foo', 'bar.txt')))
        self.assertTrue(os.path.exists(os.path.join(self.target, 'bar.txt')))
        self.assertTrue(os.path.exists(os.path.join(self.target, 'foo.txt')))

        # Verify the symlinks
        for lf in file_links:
            target_name = os.path.abspath(os.path.join(self.prefix, lf))
            link_name = os.path.abspath(os.path.join(self.target, lf))
            self.assertTrue(os.path.islink(link_name))
            self.assertEqual(target_name, os.readlink(link_name))

    def test_virtual_copy_nolinks(self):
        os_ext.copytree_virtual(self.prefix, self.target)
        self.verify_target_directory()

    def test_virtual_copy_valid_links(self):
        file_links = ['bar/', 'foo/bar.txt', 'foo.txt']
        os_ext.copytree_virtual(self.prefix, self.target, file_links)
        self.verify_target_directory(file_links)

    def test_virtual_copy_inexistent_links(self):
        file_links = ['foobar/', 'foo/bar.txt', 'foo.txt']
        self.assertRaises(ValueError, os_ext.copytree_virtual,
                          self.prefix, self.target, file_links)

    def test_virtual_copy_absolute_paths(self):
        file_links = [os.path.join(self.prefix, 'bar'),
                      'foo/bar.txt', 'foo.txt']
        self.assertRaises(ValueError, os_ext.copytree_virtual,
                          self.prefix, self.target, file_links)

    def test_virtual_copy_irrelevenant_paths(self):
        file_links = ['/bin', 'foo/bar.txt', 'foo.txt']
        self.assertRaises(ValueError, os_ext.copytree_virtual,
                          self.prefix, self.target, file_links)

        file_links = [os.path.dirname(self.prefix), 'foo/bar.txt', 'foo.txt']
        self.assertRaises(ValueError, os_ext.copytree_virtual,
                          self.prefix, self.target, file_links)

    def test_virtual_copy_linkself(self):
        file_links = ['.']
        self.assertRaises(OSError, os_ext.copytree_virtual,
                          self.prefix, self.target, file_links)

    def tearDown(self):
        shutil.rmtree(self.prefix)
        shutil.rmtree(self.target)


class TestImportFromFile(unittest.TestCase):
    def test_load_relpath(self):
        module = util.import_module_from_file('reframe/__init__.py')
        self.assertEqual(reframe.VERSION, module.VERSION)
        self.assertEqual('reframe', module.__name__)
        self.assertIs(module, sys.modules.get('reframe'))

    def test_load_directory(self):
        module = util.import_module_from_file('reframe')
        self.assertEqual(reframe.VERSION, module.VERSION)
        self.assertEqual('reframe', module.__name__)
        self.assertIs(module, sys.modules.get('reframe'))

    def test_load_abspath(self):
        filename = os.path.abspath('reframe/__init__.py')
        module = util.import_module_from_file(filename)
        self.assertEqual(reframe.VERSION, module.VERSION)
        self.assertEqual('reframe', module.__name__)
        self.assertIs(module, sys.modules.get('reframe'))

    def test_load_unknown_path(self):
        try:
            util.import_module_from_file('/foo')
            self.fail()
        except ImportError as e:
            self.assertEqual('foo', e.name)
            self.assertEqual('/foo', e.path)

    def test_load_directory_relative(self):
        with os_ext.change_dir('reframe'):
            module = util.import_module_from_file('../reframe')
            self.assertEqual(reframe.VERSION, module.VERSION)
            self.assertEqual('reframe', module.__name__)
            self.assertIs(module, sys.modules.get('reframe'))

    def test_load_relative(self):
        with os_ext.change_dir('reframe'):
            # Load a module from a directory up
            module = util.import_module_from_file('../reframe/__init__.py')
            self.assertEqual(reframe.VERSION, module.VERSION)
            self.assertEqual('reframe', module.__name__)
            self.assertIs(module, sys.modules.get('reframe'))

            # Load a module from the current directory
            module = util.import_module_from_file('utility/os_ext.py')
            self.assertEqual('reframe.utility.os_ext', module.__name__)
            self.assertIs(module, sys.modules.get('reframe.utility.os_ext'))

    def test_load_outside_pkg(self):
        module = util.import_module_from_file(os.path.__file__)

        # os imports the OS-specific path libraries under the name `path`. Our
        # importer will import the actual file, thus the module name should be
        # the real one.
        self.assertTrue(module is sys.modules.get('posixpath') or
                        module is sys.modules.get('ntpath') or
                        module is sys.modules.get('macpath'))

    def test_load_twice(self):
        filename = os.path.abspath('reframe')
        module1 = util.import_module_from_file(filename)
        module2 = util.import_module_from_file(filename)
        self.assertIs(module1, module2)

    def test_load_namespace_package(self):
        module = util.import_module_from_file('unittests/resources')
        self.assertIn('unittests', sys.modules)
        self.assertIn('unittests.resources', sys.modules)


class TestDebugRepr(unittest.TestCase):
    def test_builtin_types(self):
        # builtin types must use the default repr()
        self.assertEqual(repr(1), debug.repr(1))
        self.assertEqual(repr(1.2), debug.repr(1.2))
        self.assertEqual(repr([1, 2, 3]), debug.repr([1, 2, 3]))
        self.assertEqual(repr({1, 2, 3}), debug.repr({1, 2, 3}))
        self.assertEqual(repr({1, 2, 3}), debug.repr({1, 2, 3}))
        self.assertEqual(repr({'a': 1, 'b': {2, 3}}),
                         debug.repr({'a': 1, 'b': {2, 3}}))

    def test_obj_repr(self):
        class C:
            def __repr__(self):
                return debug.repr(self)

        class D:
            def __repr__(self):
                return debug.repr(self)

        c = C()
        c._a = -1
        c.a = 1
        c.b = {1, 2, 3}
        c.d = D()
        c.d.a = 2
        c.d.b = 3

        rep = repr(c)
        self.assertIn('unittests.test_utility', rep)
        self.assertIn('_a=%r' % c._a, rep)
        self.assertIn('b=%r' % c.b, rep)
        self.assertIn('D(...)', rep)


class TestChangeDirCtxManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.wd_save = os.getcwd()

    def test_change_dir_working(self):
        with os_ext.change_dir(self.temp_dir):
            self.assertTrue(os.getcwd(), self.temp_dir)
        self.assertEqual(os.getcwd(), self.wd_save)

    def test_exception_propagation(self):
        try:
            with os_ext.change_dir(self.temp_dir):
                raise RuntimeError
        except RuntimeError:
            self.assertEqual(os.getcwd(), self.wd_save)
        else:
            self.fail('exception not propagated by the ctx manager')

    def tearDown(self):
        os.rmdir(self.temp_dir)


class TestMiscUtilities(unittest.TestCase):
    def test_allx(self):
        l1 = [1, 1, 1]
        l2 = [True, False]
        self.assertTrue(all(l1), util.allx(l1))
        self.assertFalse(all(l2), util.allx(l2))
        self.assertFalse(util.allx([]))
        self.assertTrue(util.allx(i for i in [1, 1, 1]))
        self.assertTrue(util.allx(i for i in range(1, 2)))
        self.assertFalse(util.allx(i for i in range(1)))
        self.assertFalse(util.allx(i for i in range(0)))
        with self.assertRaises(TypeError):
            util.allx(None)

    def test_decamelize(self):
        self.assertEqual('', util.decamelize(''))
        self.assertEqual('my_base_class', util.decamelize('MyBaseClass'))
        self.assertEqual('my_base_class12', util.decamelize('MyBaseClass12'))
        self.assertEqual('my_class_a', util.decamelize('MyClass_A'))
        self.assertEqual('my_class', util.decamelize('my_class'))
        self.assertRaises(TypeError, util.decamelize, None)
        self.assertRaises(TypeError, util.decamelize, 12)

    def test_sanitize(self):
        self.assertEqual('', util.toalphanum(''))
        self.assertEqual('ab12', util.toalphanum('ab12'))
        self.assertEqual('ab1_2', util.toalphanum('ab1_2'))
        self.assertEqual('ab1__2', util.toalphanum('ab1**2'))
        self.assertEqual('ab__12_', util.toalphanum('ab (12)'))
        self.assertRaises(TypeError, util.toalphanum, None)
        self.assertRaises(TypeError, util.toalphanum, 12)


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
        self.assertEqual(3, namespace_dict['a:k1'])
        self.assertEqual(5, namespace_dict['b:k3'])
        del d['b']
        self.assertIn('b:k3', namespace_dict)

        self.assertRaises(TypeError, reframe.utility.ScopedDict, 1)
        self.assertRaises(TypeError, reframe.utility.ScopedDict,
                          {'a': 1, 'b': 2})
        self.assertRaises(TypeError, reframe.utility.ScopedDict,
                          [('a', 1), ('b', 2)])
        self.assertRaises(TypeError, reframe.utility.ScopedDict,
                          {'a': {1: 'k1'}, 'b': {2: 'k2'}})

    def test_contains(self):
        scoped_dict = reframe.utility.ScopedDict({
            'a': {'k1': 1, 'k2': 2},
            'a:b': {'k1': 3, 'k3': 4},
            'a:b:c': {'k2': 5, 'k3': 6},
            '*': {'k1': 7, 'k3': 9, 'k4': 10}
        })

        # Test simple lookup
        self.assertIn('a:k1', scoped_dict)
        self.assertIn('a:k2', scoped_dict)
        self.assertIn('a:k3', scoped_dict)
        self.assertIn('a:k4', scoped_dict)

        self.assertIn('a:b:k1', scoped_dict)
        self.assertIn('a:b:k2', scoped_dict)
        self.assertIn('a:b:k3', scoped_dict)
        self.assertIn('a:b:k4', scoped_dict)

        self.assertIn('a:b:c:k1', scoped_dict)
        self.assertIn('a:b:c:k1', scoped_dict)
        self.assertIn('a:b:c:k1', scoped_dict)
        self.assertIn('a:b:c:k1', scoped_dict)

        # Test global scope
        self.assertIn('k1', scoped_dict)
        self.assertNotIn('k2', scoped_dict)
        self.assertIn('k3', scoped_dict)
        self.assertIn('k4', scoped_dict)

        self.assertIn(':k1', scoped_dict)
        self.assertNotIn(':k2', scoped_dict)
        self.assertIn(':k3', scoped_dict)
        self.assertIn(':k4', scoped_dict)

        self.assertIn('*:k1', scoped_dict)
        self.assertNotIn('*:k2', scoped_dict)
        self.assertIn('*:k3', scoped_dict)
        self.assertIn('*:k4', scoped_dict)

        # Try to get full scopes as keys
        self.assertNotIn('a', scoped_dict)
        self.assertNotIn('a:b', scoped_dict)
        self.assertNotIn('a:b:c', scoped_dict)
        self.assertNotIn('a:b:c:d', scoped_dict)
        self.assertNotIn('*', scoped_dict)
        self.assertNotIn('', scoped_dict)

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
        self.assertEqual(sorted(expected_keys),
                         sorted(k for k in scoped_dict.keys()))

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
        self.assertEqual(sorted(expected_items),
                         sorted(item for item in scoped_dict.items()))

    def test_iter_values(self):
        scoped_dict = reframe.utility.ScopedDict({
            'a': {'k1': 1, 'k2': 2},
            'a:b': {'k1': 3, 'k3': 4},
            'a:b:c': {'k2': 5, 'k3': 6},
            '*': {'k1': 7, 'k3': 9, 'k4': 10}
        })

        expected_values = [1, 2, 3, 4, 5, 6, 7, 9, 10]
        self.assertEqual(expected_values,
                         sorted(v for v in scoped_dict.values()))

    def test_key_resolution(self):
        scoped_dict = reframe.utility.ScopedDict({
            'a': {'k1': 1, 'k2': 2},
            'a:b': {'k1': 3, 'k3': 4},
            'a:b:c': {'k2': 5, 'k3': 6},
            '*': {'k1': 7, 'k3': 9, 'k4': 10}
        })

        self.assertEqual(1, scoped_dict['a:k1'])
        self.assertEqual(2, scoped_dict['a:k2'])
        self.assertEqual(9, scoped_dict['a:k3'])
        self.assertEqual(10, scoped_dict['a:k4'])

        self.assertEqual(3, scoped_dict['a:b:k1'])
        self.assertEqual(2, scoped_dict['a:b:k2'])
        self.assertEqual(4, scoped_dict['a:b:k3'])
        self.assertEqual(10, scoped_dict['a:b:k4'])

        self.assertEqual(3, scoped_dict['a:b:c:k1'])
        self.assertEqual(5, scoped_dict['a:b:c:k2'])
        self.assertEqual(6, scoped_dict['a:b:c:k3'])
        self.assertEqual(10, scoped_dict['a:b:c:k4'])

        # Test global scope
        self.assertEqual(7, scoped_dict['k1'])
        self.assertRaises(
            KeyError, exec, "scoped_dict['k2']", globals(), locals()
        )
        self.assertEqual(9, scoped_dict['k3'])
        self.assertEqual(10, scoped_dict['k4'])

        self.assertEqual(7, scoped_dict[':k1'])
        self.assertRaises(
            KeyError, exec, "scoped_dict[':k2']", globals(), locals()
        )
        self.assertEqual(9, scoped_dict[':k3'])
        self.assertEqual(10, scoped_dict[':k4'])

        self.assertEqual(7, scoped_dict['*:k1'])
        self.assertRaises(
            KeyError, exec, "scoped_dict['*:k2']", globals(), locals()
        )
        self.assertEqual(9, scoped_dict['*:k3'])
        self.assertEqual(10, scoped_dict['*:k4'])

        # Try to fool it, by requesting keys with scope names
        self.assertRaises(
            KeyError, exec, "scoped_dict['a']", globals(), locals()
        )
        self.assertRaises(
            KeyError, exec, "scoped_dict['a:b']", globals(), locals()
        )
        self.assertRaises(
            KeyError, exec, "scoped_dict['a:b:c']", globals(), locals()
        )
        self.assertRaises(
            KeyError, exec, "scoped_dict['a:b:c:d']", globals(), locals()
        )
        self.assertRaises(
            KeyError, exec, "scoped_dict['*']", globals(), locals()
        )
        self.assertRaises(
            KeyError, exec, "scoped_dict['']", globals(), locals()
        )

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
        self.assertEqual(20, scoped_dict['a:k2'])
        self.assertEqual(30, scoped_dict['c:k2'])
        self.assertEqual(40, scoped_dict[':k4'])
        self.assertEqual(50, scoped_dict['k5'])
        self.assertEqual(60, scoped_dict['k6'])

    def test_delitem(self):
        scoped_dict = reframe.utility.ScopedDict({
            'a': {'k1': 1, 'k2': 2},
            'a:b': {'k1': 3, 'k3': 4},
            'a:b:c': {'k2': 5, 'k3': 6},
            '*': {'k1': 7, 'k3': 9, 'k4': 10}
        })

        # delete key
        del scoped_dict['a:k1']
        self.assertEqual(7, scoped_dict['a:k1'])

        # delete key from global scope
        del scoped_dict['k1']
        self.assertEqual(9, scoped_dict['k3'])
        self.assertEqual(10, scoped_dict['k4'])
        self.assertRaises(
            KeyError, exec, "scoped_dict['k1']", globals(), locals()
        )

        # delete a whole scope
        del scoped_dict['*']
        self.assertRaises(
            KeyError, exec, "scoped_dict[':k4']", globals(), locals()
        )
        self.assertRaises(
            KeyError, exec, "scoped_dict['a:k3']", globals(), locals()
        )

        # try to delete a non-existent key
        self.assertRaises(
            KeyError, exec, "del scoped_dict['a:k4']", globals(), locals()
        )

        # test deletion of parent scope keeping a nested one
        scoped_dict = reframe.utility.ScopedDict()
        scoped_dict['s0:k0'] = 1
        scoped_dict['s0:s1:k0'] = 2
        scoped_dict['*:k0'] = 3
        del scoped_dict['s0']
        self.assertEqual(3, scoped_dict['s0:k0'])
        self.assertEqual(2, scoped_dict['s0:s1:k0'])

    def test_scope_key_name_pseudoconflict(self):
        scoped_dict = reframe.utility.ScopedDict({
            's0': {'s1': 1},
            's0:s1': {'k0': 2}
        })

        self.assertEqual(1, scoped_dict['s0:s1'])
        self.assertEqual(2, scoped_dict['s0:s1:k0'])

        del scoped_dict['s0:s1']
        self.assertEqual(2, scoped_dict['s0:s1:k0'])
        self.assertRaises(
            KeyError, exec, "scoped_dict['s0:s1']", globals(), locals()
        )

    def test_update(self):
        scoped_dict = reframe.utility.ScopedDict({
            'a': {'k1': 1, 'k2': 2},
            'a:b': {'k1': 3, 'k3': 4},
            'a:b:c': {'k2': 5, 'k3': 6},
            '*': {'k1': 7, 'k3': 9, 'k4': 10}
        })

        scoped_dict_alt = reframe.utility.ScopedDict({'a': {'k1': 3, 'k2': 5}})
        scoped_dict_alt.update({
            'a': {'k1': 1, 'k2': 2},
            'a:b': {'k1': 3, 'k3': 4},
            'a:b:c': {'k2': 5, 'k3': 6},
            '*': {'k1': 7, 'k3': 9, 'k4': 10}
        })
        self.assertEqual(scoped_dict, scoped_dict_alt)
