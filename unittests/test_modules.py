import abc
import os
import unittest
from tempfile import NamedTemporaryFile

import reframe.core.modules as modules
from reframe.core.environments import EnvironmentSnapshot
from reframe.core.exceptions import ConfigError, EnvironError
from reframe.core.runtime import runtime
from unittests.fixtures import TEST_MODULES


class _TestModulesSystem:
    def setUp(self):
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

        unloaded = self.modules_system.load_module('testmod_bar', force=True)
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

    @abc.abstractmethod
    def expected_load_instr(self, module):
        """Expected load instruction."""

    @abc.abstractmethod
    def expected_unload_instr(self, module):
        """Expected unload instruction."""

    def test_emit_load_commands(self):
        self.modules_system.module_map = {
            'm0': ['m1', 'm2']
        }
        self.assertEqual([self.expected_load_instr('foo')],
                         self.modules_system.emit_load_commands('foo'))
        self.assertEqual([self.expected_load_instr('foo/1.2')],
                         self.modules_system.emit_load_commands('foo/1.2'))
        self.assertEqual([self.expected_load_instr('m1'),
                          self.expected_load_instr('m2')],
                         self.modules_system.emit_load_commands('m0'))

    def test_emit_unload_commands(self):
        self.modules_system.module_map = {
            'm0': ['m1', 'm2']
        }
        self.assertEqual([self.expected_unload_instr('foo')],
                         self.modules_system.emit_unload_commands('foo'))
        self.assertEqual([self.expected_unload_instr('foo/1.2')],
                         self.modules_system.emit_unload_commands('foo/1.2'))
        self.assertEqual([self.expected_unload_instr('m2'),
                          self.expected_unload_instr('m1')],
                         self.modules_system.emit_unload_commands('m0'))


class TestTModModulesSystem(_TestModulesSystem, unittest.TestCase):
    def setUp(self):
        try:
            self.modules_system = modules.ModulesSystem.create('tmod')
        except ConfigError:
            self.skipTest('tmod not supported')
        else:
            super().setUp()

    def expected_load_instr(self, module):
        return 'module load %s' % module

    def expected_unload_instr(self, module):
        return 'module unload %s' % module


class TestTMod4ModulesSystem(_TestModulesSystem, unittest.TestCase):
    def setUp(self):
        try:
            self.modules_system = modules.ModulesSystem.create('tmod4')
        except ConfigError:
            self.skipTest('tmod4 not supported')
        else:
            super().setUp()

    def expected_load_instr(self, module):
        return 'module load %s' % module

    def expected_unload_instr(self, module):
        return 'module unload %s' % module


class TestLModModulesSystem(_TestModulesSystem, unittest.TestCase):
    def setUp(self):
        try:
            self.modules_system = modules.ModulesSystem.create('lmod')
        except ConfigError:
            self.skipTest('lmod not supported')
        else:
            super().setUp()

    def expected_load_instr(self, module):
        return 'module load %s' % module

    def expected_unload_instr(self, module):
        return 'module unload %s' % module


class TestNoModModulesSystem(_TestModulesSystem, unittest.TestCase):
    def setUp(self):
        try:
            self.modules_system = modules.ModulesSystem.create()
        except ConfigError:
            self.skipTest('nomod not supported')
        else:
            super().setUp()

    def expected_load_instr(self, module):
        return ''

    def expected_unload_instr(self, module):
        return ''

    def test_searchpath(self):
        # Simply test that no exceptions are thrown
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
        self.assertEqual(modules.Module('foo/1.2'),
                         modules.Module('foo/1.2'))
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


class ModulesSystemEmulator(modules.ModulesSystemImpl):
    """A convenience class that simulates a modules system."""

    def __init__(self):
        self._loaded_modules = set()

        # The following two variables record the sequence of loads and unloads
        self.load_seq = []
        self.unload_seq = []

    def loaded_modules(self):
        return list(self._loaded_modules)

    def conflicted_modules(self, module):
        return []

    def load_module(self, module):
        self.load_seq.append(module.name)
        self._loaded_modules.add(module.name)

    def unload_module(self, module):
        self.unload_seq.append(module.name)
        try:
            self._loaded_modules.remove(module.name)
        except KeyError:
            pass

    def is_module_loaded(self, module):
        return module.name in self._loaded_modules

    def name(self):
        return 'nomod_debug'

    def version(self):
        return '1.0'

    def unload_all(self):
        self._loaded_modules.clear()

    def searchpath(self):
        return []

    def searchpath_add(self, *dirs):
        pass

    def searchpath_remove(self, *dirs):
        pass

    def emit_load_instr(self, module):
        return ''

    def emit_unload_instr(self, module):
        return ''


class TestModuleMapping(unittest.TestCase):
    def setUp(self):
        self.modules_activity = ModulesSystemEmulator()
        self.modules_system = modules.ModulesSystem(self.modules_activity)
        self.mapping_file = NamedTemporaryFile('wt', delete=False)

    def tearDown(self):
        self.mapping_file.close()
        os.remove(self.mapping_file.name)

    def test_mapping_simple(self):
        #
        # m0 -> m1
        #
        self.modules_system.module_map = {'m0': ['m1']}
        self.modules_system.load_module('m0')
        self.assertTrue(self.modules_system.is_module_loaded('m0'))
        self.assertTrue(self.modules_system.is_module_loaded('m1'))
        self.assertEqual(['m1'], self.modules_system.backend.load_seq)

        # Unload module
        self.modules_system.unload_module('m1')
        self.assertFalse(self.modules_system.is_module_loaded('m0'))
        self.assertFalse(self.modules_system.is_module_loaded('m1'))

    def test_mapping_chain(self):
        #
        # m0 -> m1 -> m2
        #
        self.modules_system.module_map = {
            'm0': ['m1'],
            'm1': ['m2']
        }
        self.modules_system.load_module('m0')
        self.assertTrue(self.modules_system.is_module_loaded('m0'))
        self.assertTrue(self.modules_system.is_module_loaded('m1'))
        self.assertTrue(self.modules_system.is_module_loaded('m2'))
        self.assertEqual(['m2'], self.modules_system.backend.load_seq)

        # Unload module
        self.modules_system.unload_module('m1')
        self.assertFalse(self.modules_system.is_module_loaded('m0'))
        self.assertFalse(self.modules_system.is_module_loaded('m1'))
        self.assertFalse(self.modules_system.is_module_loaded('m2'))

    def test_mapping_n_to_one(self):
        #
        # m0 -> m2 <- m1
        #
        self.modules_system.module_map = {
            'm0': ['m2'],
            'm1': ['m2']
        }
        self.modules_system.load_module('m0')
        self.assertTrue(self.modules_system.is_module_loaded('m0'))
        self.assertTrue(self.modules_system.is_module_loaded('m1'))
        self.assertTrue(self.modules_system.is_module_loaded('m2'))
        self.assertEqual(['m2'], self.modules_system.backend.load_seq)

        # Unload module
        self.modules_system.unload_module('m0')
        self.assertFalse(self.modules_system.is_module_loaded('m0'))
        self.assertFalse(self.modules_system.is_module_loaded('m1'))
        self.assertFalse(self.modules_system.is_module_loaded('m2'))

    def test_mapping_one_to_n(self):
        #
        # m2 <- m0 -> m1
        #
        self.modules_system.module_map = {
            'm0': ['m1', 'm2'],
        }
        self.modules_system.load_module('m0')
        self.assertTrue(self.modules_system.is_module_loaded('m0'))
        self.assertTrue(self.modules_system.is_module_loaded('m1'))
        self.assertTrue(self.modules_system.is_module_loaded('m2'))
        self.assertEqual(['m1', 'm2'], self.modules_system.backend.load_seq)

        # m0 is loaded only if m1 and m2 are.
        self.modules_system.unload_module('m2')
        self.assertFalse(self.modules_system.is_module_loaded('m0'))
        self.assertTrue(self.modules_system.is_module_loaded('m1'))

    def test_mapping_deep_dfs_order(self):
        #
        #    -- > m1 ---- > m3
        #   /       \
        # m0         \
        #   \         \
        #    -- > m2   -- > m4
        #
        self.modules_system.module_map = {
            'm0': ['m1', 'm2'],
            'm1': ['m3', 'm4']
        }
        self.modules_system.load_module('m0')
        self.assertTrue(self.modules_system.is_module_loaded('m0'))
        self.assertTrue(self.modules_system.is_module_loaded('m1'))
        self.assertTrue(self.modules_system.is_module_loaded('m2'))
        self.assertTrue(self.modules_system.is_module_loaded('m3'))
        self.assertTrue(self.modules_system.is_module_loaded('m4'))
        self.assertEqual(['m3', 'm4', 'm2'],
                         self.modules_system.backend.load_seq)

        # Test unloading
        self.modules_system.unload_module('m2')
        self.assertFalse(self.modules_system.is_module_loaded('m0'))
        self.assertTrue(self.modules_system.is_module_loaded('m1'))
        self.assertFalse(self.modules_system.is_module_loaded('m2'))
        self.assertTrue(self.modules_system.is_module_loaded('m3'))
        self.assertTrue(self.modules_system.is_module_loaded('m4'))

    def test_mapping_deep_dfs_unload_order(self):
        #
        #    -- > m1 ---- > m3
        #   /       \
        # m0         \
        #   \         \
        #    -- > m2   -- > m4
        #
        self.modules_system.module_map = {
            'm0': ['m1', 'm2'],
            'm1': ['m3', 'm4']
        }
        self.modules_system.load_module('m0')
        self.modules_system.unload_module('m0')
        self.assertEqual(['m2', 'm4', 'm3'],
                         self.modules_system.backend.unload_seq)

    def test_mapping_multiple_paths(self):
        #
        #    -- > m1
        #   /     ^
        # m0      |
        #   \     |
        #    -- > m2
        #
        self.modules_system.module_map = {
            'm0': ['m1', 'm2'],
            'm2': ['m1'],
        }
        self.modules_system.load_module('m0')
        self.assertTrue(self.modules_system.is_module_loaded('m0'))
        self.assertTrue(self.modules_system.is_module_loaded('m1'))
        self.assertTrue(self.modules_system.is_module_loaded('m2'))
        self.assertEqual(['m1'], self.modules_system.backend.load_seq)

        # Test unloading
        self.modules_system.unload_module('m2')
        self.assertFalse(self.modules_system.is_module_loaded('m0'))
        self.assertFalse(self.modules_system.is_module_loaded('m1'))
        self.assertFalse(self.modules_system.is_module_loaded('m2'))

    def test_mapping_deep_multiple_paths(self):
        #
        #    -- > m1 ---- > m3
        #   /     ^ \
        # m0      |  \
        #   \     |   \
        #    -- > m2   -- > m4
        #
        self.modules_system.module_map = {
            'm0': ['m1', 'm2'],
            'm1': ['m3', 'm4'],
            'm2': ['m1']
        }
        self.modules_system.load_module('m0')
        self.assertTrue(self.modules_system.is_module_loaded('m0'))
        self.assertTrue(self.modules_system.is_module_loaded('m1'))
        self.assertTrue(self.modules_system.is_module_loaded('m2'))
        self.assertTrue(self.modules_system.is_module_loaded('m3'))
        self.assertTrue(self.modules_system.is_module_loaded('m4'))
        self.assertEqual(['m3', 'm4'], self.modules_system.backend.load_seq)

    def test_mapping_cycle_simple(self):
        #
        # m0 -> m1 -> m0
        #
        self.modules_system.module_map = {
            'm0': ['m1'],
            'm1': ['m0'],
        }
        self.assertRaises(EnvironError, self.modules_system.load_module, 'm0')
        self.assertRaises(EnvironError, self.modules_system.load_module, 'm1')

    def test_mapping_cycle_self(self):
        #
        # m0 -> m0
        #
        self.modules_system.module_map = {
            'm0': ['m0'],
        }
        self.assertRaises(EnvironError, self.modules_system.load_module, 'm0')

    def test_mapping_deep_cycle(self):
        #
        #    -- > m1 ---- > m3
        #   /     ^         |
        # m0      |         |
        #   \     |         .
        #    -- > m2 < ---- m4
        #
        self.modules_system.module_map = {
            'm0': ['m1', 'm2'],
            'm1': ['m3'],
            'm2': ['m1'],
            'm3': ['m4'],
            'm4': ['m2']
        }
        self.assertRaisesRegex(EnvironError, 'm0->m1->m3->m4->m2->m1',
                               self.modules_system.load_module, 'm0')

    def test_mapping_from_file_simple(self):
        with self.mapping_file:
            self.mapping_file.write('m1:m2  m3   m3\n'
                                    'm2 : m4 \n'
                                    ' #m5: m6\n'
                                    '\n'
                                    ' m2: m7 m8\n'
                                    'm9: m10 # Inline comment')

        reference_map = {
            'm1': ['m2', 'm3'],
            'm2': ['m7', 'm8'],
            'm9': ['m10']
        }
        self.modules_system.load_mapping_from_file(self.mapping_file.name)
        self.assertEqual(reference_map, self.modules_system.module_map)

    def test_maping_from_file_missing_key_separator(self):
        with self.mapping_file:
            self.mapping_file.write('m1 m2')

        self.assertRaises(ConfigError,
                          self.modules_system.load_mapping_from_file,
                          self.mapping_file.name)

    def test_maping_from_file_empty_value(self):
        with self.mapping_file:
            self.mapping_file.write('m1: # m2')

        self.assertRaises(ConfigError,
                          self.modules_system.load_mapping_from_file,
                          self.mapping_file.name)

    def test_maping_from_file_multiple_key_separators(self):
        with self.mapping_file:
            self.mapping_file.write('m1 : m2 : m3')

        self.assertRaises(ConfigError,
                          self.modules_system.load_mapping_from_file,
                          self.mapping_file.name)

    def test_maping_from_file_empty_key(self):
        with self.mapping_file:
            self.mapping_file.write(' :  m2')

        self.assertRaises(ConfigError,
                          self.modules_system.load_mapping_from_file,
                          self.mapping_file.name)

    def test_maping_from_file_missing_file(self):
        self.assertRaises(OSError,
                          self.modules_system.load_mapping_from_file,
                          'foo')
