import os
import unittest

import reframe.core.environments as renv
import reframe.utility.os_ext as os_ext
import unittests.fixtures as fixtures
from reframe.core.runtime import runtime
from reframe.core.exceptions import EnvironError


class TestEnvironment(unittest.TestCase):
    def assertModulesLoaded(self, modules):
        for m in modules:
            self.assertTrue(self.modules_system.is_module_loaded(m))

    def assertModulesNotLoaded(self, modules):
        for m in modules:
            self.assertFalse(self.modules_system.is_module_loaded(m))

    def setup_modules_system(self):
        if not fixtures.has_sane_modules_system():
            self.skipTest('no modules system configured')

        self.modules_system = runtime().modules_system
        self.modules_system.searchpath_add(fixtures.TEST_MODULES)

        # Always add a base module; this is a workaround for the modules
        # environment's inconsistent behaviour, that starts with an empty
        # LOADEDMODULES variable and ends up removing it completely if all
        # present modules are removed.
        self.modules_system.load_module('testmod_base')

    def setUp(self):
        self.modules_system = None
        os.environ['_var0'] = 'val0'
        os.environ['_var1'] = 'val1'
        self.environ_save = renv.EnvironmentSnapshot()
        self.environ = renv.Environment(name='TestEnv1',
                                        modules=['testmod_foo'],
                                        variables=[('_var0', 'val1'),
                                                   ('_var2', '$_var0'),
                                                   ('_var3', '${_var1}')])
        self.environ_other = renv.Environment(name='TestEnv2',
                                              modules=['testmod_boo'],
                                              variables={'_var4': 'val4'})

    def tearDown(self):
        if self.modules_system is not None:
            self.modules_system.searchpath_remove(fixtures.TEST_MODULES)

        self.environ_save.load()

    def test_setup(self):
        if fixtures.has_sane_modules_system():
            self.assertEqual(len(self.environ.modules), 1)
            self.assertIn('testmod_foo', self.environ.modules)

        self.assertEqual(len(self.environ.variables.keys()), 3)
        self.assertEqual(self.environ.variables['_var0'], 'val1')

        # No variable expansion, if environment is not loaded
        self.assertEqual(self.environ.variables['_var2'], '$_var0')
        self.assertEqual(self.environ.variables['_var3'], '${_var1}')

    def test_environ_snapshot(self):
        self.assertRaises(NotImplementedError, self.environ_save.unload)
        self.environ.load()
        self.environ_other.load()
        self.environ_save.load()
        self.assertEqual(self.environ_save, renv.EnvironmentSnapshot())

    def test_environ_snapshot_context_mgr(self):
        with renv.save_environment() as env:
            self.assertIsInstance(env, renv.EnvironmentSnapshot)
            del os.environ['_var0']
            os.environ['_var1'] = 'valX'
            os.environ['_var2'] = 'var3'

        self.assertEqual('val0', os.environ['_var0'])
        self.assertEqual('val1', os.environ['_var1'])
        self.assertNotIn('_var2', os.environ)

    def test_load_restore(self):
        self.environ.load()
        self.assertEqual(os.environ['_var0'], 'val1')
        self.assertEqual(os.environ['_var1'], 'val1')
        self.assertEqual(os.environ['_var2'], 'val1')
        self.assertEqual(os.environ['_var3'], 'val1')
        if fixtures.has_sane_modules_system():
            self.assertModulesLoaded(self.environ.modules)

        self.assertTrue(self.environ.is_loaded)
        self.environ.unload()
        self.assertEqual(self.environ_save, renv.EnvironmentSnapshot())
        self.assertEqual(os.environ['_var0'], 'val0')
        if fixtures.has_sane_modules_system():
            self.assertFalse(
                self.modules_system.is_module_loaded('testmod_foo'))

        self.assertFalse(self.environ.is_loaded)

    @fixtures.switch_to_user_runtime
    def test_load_already_present(self):
        self.setup_modules_system()
        self.modules_system.load_module('testmod_boo')
        self.environ.load()
        self.environ.unload()
        self.assertTrue(self.modules_system.is_module_loaded('testmod_boo'))

    def test_equal(self):
        env1 = renv.Environment('env1', modules=['foo', 'bar'],
                                variables={'a': 1, 'b': 2})
        env2 = renv.Environment('env1', modules=['bar', 'foo'],
                                variables={'b': 2, 'a': 1})
        self.assertEqual(env1, env2)

    def test_not_equal(self):
        env1 = renv.Environment('env1', modules=['foo', 'bar'])
        env2 = renv.Environment('env2', modules=['foo', 'bar'])
        self.assertNotEqual(env1, env2)

    @fixtures.switch_to_user_runtime
    def test_conflicting_environments(self):
        self.setup_modules_system()
        envfoo = renv.Environment(name='envfoo',
                                  modules=['testmod_foo', 'testmod_boo'])
        envbar = renv.Environment(name='envbar', modules=['testmod_bar'])
        envfoo.load()
        envbar.load()
        for m in envbar.modules:
            self.assertTrue(self.modules_system.is_module_loaded(m))

        for m in envfoo.modules:
            self.assertFalse(self.modules_system.is_module_loaded(m))

    @fixtures.switch_to_user_runtime
    def test_conflict_environ_after_module_load(self):
        self.setup_modules_system()
        self.modules_system.load_module('testmod_foo')
        envfoo = renv.Environment(name='envfoo', modules=['testmod_foo'])
        envfoo.load()
        envfoo.unload()
        self.assertTrue(self.modules_system.is_module_loaded('testmod_foo'))

    @fixtures.switch_to_user_runtime
    def test_conflict_environ_after_module_force_load(self):
        self.setup_modules_system()
        self.modules_system.load_module('testmod_foo')
        envbar = renv.Environment(name='envbar', modules=['testmod_bar'])
        envbar.load()
        envbar.unload()
        self.assertTrue(self.modules_system.is_module_loaded('testmod_foo'))

    def test_swap(self):
        from reframe.core.environments import swap_environments

        self.environ.load()
        swap_environments(self.environ, self.environ_other)
        self.assertFalse(self.environ.is_loaded)
        self.assertTrue(self.environ_other.is_loaded)

    @fixtures.switch_to_user_runtime
    def test_emit_load_commands(self):
        self.setup_modules_system()
        rt = runtime()
        expected_commands = [
            rt.modules_system.emit_load_commands('testmod_foo')[0],
            'export _var0=val1',
            'export _var2=$_var0',
            'export _var3=${_var1}',
        ]
        self.assertEqual(expected_commands, self.environ.emit_load_commands())

    @fixtures.switch_to_user_runtime
    def test_emit_load_commands_with_confict(self):
        self.setup_modules_system()

        # Load a conflicting module
        self.modules_system.load_module('testmod_bar')

        # When the environment is not loaded, the conflicting module does not
        # make a difference
        self.test_emit_load_commands()

        self.environ.load()
        rt = runtime()
        expected_commands = [
            rt.modules_system.emit_unload_commands('testmod_bar')[0],
            rt.modules_system.emit_load_commands('testmod_foo')[0],
            'export _var0=val1',
            'export _var2=$_var0',
            'export _var3=${_var1}',
        ]
        self.assertEqual(expected_commands, self.environ.emit_load_commands())

    @fixtures.switch_to_user_runtime
    def test_emit_unload_commands(self):
        self.setup_modules_system()
        rt = runtime()
        expected_commands = [
            'unset _var0',
            'unset _var2',
            'unset _var3',
            rt.modules_system.emit_unload_commands('testmod_foo')[0],
        ]
        self.assertEqual(expected_commands,
                         self.environ.emit_unload_commands())

    @fixtures.switch_to_user_runtime
    def test_emit_unload_commands_with_confict(self):
        self.setup_modules_system()

        # Load a conflicting module
        self.modules_system.load_module('testmod_bar')

        # When the environment is not loaded, the conflicting module does not
        # make a difference
        self.test_emit_unload_commands()

        self.environ.load()
        rt = runtime()
        expected_commands = [
            'unset _var0',
            'unset _var2',
            'unset _var3',
            rt.modules_system.emit_unload_commands('testmod_foo')[0],
            rt.modules_system.emit_load_commands('testmod_bar')[0],
        ]
        self.assertEqual(expected_commands,
                         self.environ.emit_unload_commands())
