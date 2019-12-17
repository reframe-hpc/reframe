import os
import pytest
import unittest

import reframe.core.environments as env
import reframe.utility.os_ext as os_ext
import unittests.fixtures as fixtures
from reframe.core.runtime import runtime
from reframe.core.exceptions import EnvironError


class TestEnvironment(unittest.TestCase):
    def assertModulesLoaded(self, modules):
        for m in modules:
            assert self.modules_system.is_module_loaded(m)

    def assertModulesNotLoaded(self, modules):
        for m in modules:
            assert not self.modules_system.is_module_loaded(m)

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
        self.environ_save = env.snapshot()
        self.environ = env.Environment(name='TestEnv1',
                                       modules=['testmod_foo'],
                                       variables=[('_var0', 'val1'),
                                                  ('_var2', '$_var0'),
                                                  ('_var3', '${_var1}')])
        self.environ_other = env.Environment(name='TestEnv2',
                                             modules=['testmod_boo'],
                                             variables={'_var4': 'val4'})

    def tearDown(self):
        if self.modules_system is not None:
            self.modules_system.searchpath_remove(fixtures.TEST_MODULES)

        self.environ_save.restore()

    def test_setup(self):
        if fixtures.has_sane_modules_system():
            assert len(self.environ.modules) == 1
            assert 'testmod_foo' in self.environ.modules

        assert len(self.environ.variables.keys()) == 3
        assert self.environ.variables['_var0'] == 'val1'

        # No variable expansion, if environment is not loaded
        self.environ.variables['_var2'] == '$_var0'
        self.environ.variables['_var3'] == '${_var1}'

    def test_environ_snapshot(self):
        env.load(self.environ, self.environ_other)
        self.environ_save.restore()
        assert self.environ_save == env.snapshot()
        assert not self.environ.is_loaded
        assert not self.environ_other.is_loaded

    def test_load_restore(self):
        snapshot, _ = env.load(self.environ)
        os.environ['_var0'] == 'val1'
        os.environ['_var1'] == 'val1'
        os.environ['_var2'] == 'val1'
        os.environ['_var3'] == 'val1'
        if fixtures.has_sane_modules_system():
            self.assertModulesLoaded(self.environ.modules)

        assert self.environ.is_loaded
        snapshot.restore()
        self.environ_save == env.snapshot()
        os.environ['_var0'], 'val0'
        if fixtures.has_sane_modules_system():
            assert not self.modules_system.is_module_loaded('testmod_foo')

        assert not self.environ.is_loaded

    @fixtures.switch_to_user_runtime
    def test_temp_environment_context_manager(self):
        with env.temp_environment(['testmod_foo'],
            [('_var0', 'val2'), ('_var3', 'val3')]) as environ:
            assert environ.is_loaded

        assert not environ.is_loaded

    @fixtures.switch_to_user_runtime
    def test_load_already_present(self):
        self.setup_modules_system()
        self.modules_system.load_module('testmod_boo')
        snapshot, _ = env.load(self.environ)
        snapshot.restore()
        assert self.modules_system.is_module_loaded('testmod_boo')

    def test_load_non_overlapping(self):
        e0 = env.Environment(name='e0', variables=[('a', '1'), ('b', '2')])
        e1 = env.Environment(name='e1', variables=[('c', '3'), ('d', '4')])
        env.load(e0, e1)
        assert e0.is_loaded
        assert e1.is_loaded

    def test_load_overlapping(self):
        e0 = env.Environment(name='e0', variables=[('a', '1'), ('b', '2')])
        e1 = env.Environment(name='e1', variables=[('b', '3'), ('c', '4')])
        env.load(e0, e1)
        assert not e0.is_loaded
        assert e1.is_loaded

    def test_equal(self):
        env1 = env.Environment('env1', modules=['foo', 'bar'])
        env2 = env.Environment('env1', modules=['bar', 'foo'])
        assert env1 == env2
        assert env2 == env1

    def test_not_equal(self):
        env1 = env.Environment('env1', modules=['foo', 'bar'])
        env2 = env.Environment('env2', modules=['foo', 'bar'])
        assert env1 != env2

        # Variables are ordered, because they might depend on each other
        env1 = env.Environment('env1', variables=[('a', 1), ('b', 2)])
        env2 = env.Environment('env1', variables=[('b', 2), ('a', 1)])
        assert env1 != env2

    @fixtures.switch_to_user_runtime
    def test_conflicting_environments(self):
        self.setup_modules_system()
        envfoo = env.Environment(name='envfoo',
                                 modules=['testmod_foo', 'testmod_boo'])
        envbar = env.Environment(name='envbar', modules=['testmod_bar'])
        env.load(envfoo, envbar)
        for m in envbar.modules:
            assert self.modules_system.is_module_loaded(m)

        for m in envfoo.modules:
            assert not self.modules_system.is_module_loaded(m)

    @fixtures.switch_to_user_runtime
    def test_conflict_environ_after_module_load(self):
        self.setup_modules_system()
        self.modules_system.load_module('testmod_foo')
        envfoo = env.Environment(name='envfoo', modules=['testmod_foo'])
        snapshot, _ = env.load(envfoo)
        snapshot.restore()
        assert self.modules_system.is_module_loaded('testmod_foo')

    @fixtures.switch_to_user_runtime
    def test_conflict_environ_after_module_force_load(self):
        self.setup_modules_system()
        self.modules_system.load_module('testmod_foo')
        envbar = env.Environment(name='envbar', modules=['testmod_bar'])
        snapshot, _ = env.load(envbar)
        snapshot.restore()
        assert self.modules_system.is_module_loaded('testmod_foo')

    def test_immutability(self):
        # Check emit_load_commands()
        _, commands = env.load(self.environ)

        # Try to modify the returned list of commands
        commands.append('foo')
        assert 'foo' not in env.load(self.environ)[1]

        # Test ProgEnvironment
        prgenv = env.ProgEnvironment('foo_prgenv')
        assert isinstance(prgenv, env.Environment)
        with pytest.raises(AttributeError):
            prgenv.cc = 'gcc'

        with pytest.raises(AttributeError):
            prgenv.cxx = 'g++'

        with pytest.raises(AttributeError):
            prgenv.ftn = 'gfortran'

        with pytest.raises(AttributeError):
            prgenv.nvcc = 'clang'

        with pytest.raises(AttributeError):
            prgenv.cppflags = ['-DFOO']

        with pytest.raises(AttributeError):
            prgenv.cflags = ['-O1']

        with pytest.raises(AttributeError):
            prgenv.cxxflags = ['-O1']

        with pytest.raises(AttributeError):
            prgenv.fflags = ['-O1']

        with pytest.raises(AttributeError):
            prgenv.ldflags = ['-lm']

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
        assert expected_commands == env.emit_load_commands(self.environ)

    @fixtures.switch_to_user_runtime
    def test_emit_load_commands_with_confict(self):
        self.setup_modules_system()

        # Load a conflicting module
        self.modules_system.load_module('testmod_bar')
        rt = runtime()
        expected_commands = [
            rt.modules_system.emit_unload_commands('testmod_bar')[0],
            rt.modules_system.emit_load_commands('testmod_foo')[0],
            'export _var0=val1',
            'export _var2=$_var0',
            'export _var3=${_var1}',
        ]
        assert expected_commands == env.emit_load_commands(self.environ)
