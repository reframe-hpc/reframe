import abc
import os
import unittest
import reframe.core.modules as modules

from reframe.core.environments import EnvironmentSnapshot
from reframe.core.exceptions import ConfigError, EnvironError
from unittests.fixtures import TEST_MODULES, has_sane_modules_system


class _TestModulesSystem(unittest.TestCase):
    def setUp(self):
        self.modules_system = modules.get_modules_system()
        self.environ_save = EnvironmentSnapshot()
        self.modules_system.searchpath_add(TEST_MODULES)

    def tearDown(self):
        self.environ_save.load()

    def test_searchpath(self):
        self.assertIn(TEST_MODULES, self.modules_system.searchpath)

        self.modules_system.searchpath_remove(TEST_MODULES)
        self.assertNotIn(TEST_MODULES, self.modules_system.searchpath)

    def test_module_load(self):
        self.assertRaises(EnvironError, self.modules_system.load_module, 'foo')
        self.assertFalse(self.modules_system.is_module_loaded('foo'))
        self.assertNotIn('foo', self.modules_system.loaded_modules())

        self.modules_system.load_module('testmod_foo')
        self.assertTrue(self.modules_system.is_module_loaded('testmod_foo'))
        self.assertIn('testmod_foo', self.modules_system.loaded_modules())
        self.assertIn('TESTMOD_FOO', os.environ)

        self.modules_system.unload_module('testmod_foo')
        self.assertFalse(self.modules_system.is_module_loaded('testmod_foo'))
        self.assertNotIn('testmod_foo', self.modules_system.loaded_modules())
        self.assertNotIn('TESTMOD_FOO', os.environ)

    def test_module_load_force(self):
        self.modules_system.load_module('testmod_foo')

        unloaded = self.modules_system.load_module('testmod_foo', force=True)
        self.assertEqual(0, len(unloaded))
        self.assertTrue(self.modules_system.is_module_loaded('testmod_foo'))

        unloaded = self.modules_system.load_module('testmod_bar', True)
        self.assertTrue(self.modules_system.is_module_loaded('testmod_bar'))
        self.assertFalse(self.modules_system.is_module_loaded('testmod_foo'))
        self.assertIn('testmod_foo', unloaded)
        self.assertIn('TESTMOD_BAR', os.environ)

    def test_module_unload_all(self):
        self.modules_system.load_module('testmod_base')
        self.modules_system.unload_all()
        self.assertEqual(0, len(self.modules_system.loaded_modules()))

    def test_module_list(self):
        self.modules_system.load_module('testmod_foo')
        self.assertIn('testmod_foo', self.modules_system.loaded_modules())
        self.modules_system.unload_module('testmod_foo')

    def test_module_conflict_list(self):
        conflict_list = self.modules_system.conflicted_modules('testmod_bar')
        self.assertIn('testmod_foo', conflict_list)
        self.assertIn('testmod_boo', conflict_list)


class TestTModModulesSystem(_TestModulesSystem):
    def setUp(self):
        try:
            modules.init_modules_system('tmod')
        except ConfigError:
            self.skipTest('tmod not supported')
        else:
            super().setUp()


class TestNoModModulesSystem(_TestModulesSystem):
    def setUp(self):
        try:
            modules.init_modules_system()
        except ConfigError:
            self.skipTest('nomod not supported')
        else:
            super().setUp()

    # Simply test that no exceptions are thrown
    def test_searchpath(self):
        self.modules_system.searchpath_remove(TEST_MODULES)

    def test_module_load(self):
        self.modules_system.load_module('foo')
        self.modules_system.unload_module('foo')

    def test_module_load_force(self):
        self.modules_system.load_module('foo', force=True)

    def test_module_unload_all(self):
        self.modules_system.unload_all()

    def test_module_list(self):
        self.assertEqual(0, len(self.modules_system.loaded_modules()))

    def test_module_conflict_list(self):
        self.assertEqual(0, len(self.modules_system.conflicted_modules('foo')))


class TestModule(unittest.TestCase):
    def setUp(self):
        self.module = modules.Module('foo/1.2')

    def test_invalid_initialization(self):
        self.assertRaises(ValueError, modules.Module, '')
        self.assertRaises(ValueError, modules.Module, ' ')
        self.assertRaises(TypeError, modules.Module, None)
        self.assertRaises(TypeError, modules.Module, 23)

    def test_name_version(self):
        self.assertEqual(self.module.name, 'foo')
        self.assertEqual(self.module.version, '1.2')

    def test_equal(self):
        self.assertEqual(modules.Module('foo'), modules.Module('foo'))
        self.assertEqual(modules.Module('foo/1.2'), modules.Module('foo/1.2'))
        self.assertEqual(modules.Module('foo'), modules.Module('foo/1.2'))
        self.assertEqual(hash(modules.Module('foo')),
                         hash(modules.Module('foo')))
        self.assertEqual(hash(modules.Module('foo/1.2')),
                         hash(modules.Module('foo/1.2')))
        self.assertEqual(hash(modules.Module('foo')),
                         hash(modules.Module('foo/1.2')))
        self.assertNotEqual(modules.Module('foo/1.2'),
                            modules.Module('foo/1.3'))
        self.assertNotEqual(modules.Module('foo'), modules.Module('bar'))
        self.assertNotEqual(modules.Module('foo'), modules.Module('foobar'))
