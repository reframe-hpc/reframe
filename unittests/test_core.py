import os
import tempfile
import stat
import unittest
import reframe.core.debug as debug
import reframe.utility.os as os_ext

from reframe.core.environments import Environment, EnvironmentSnapshot, \
    ProgEnvironment
from reframe.core.modules import *
from reframe.core.exceptions import CompilationError
from reframe.core.modules import *
from reframe.core.shell import BashScriptBuilder
from unittests.fixtures import TEST_RESOURCES, TEST_MODULES, force_remove_file


class TestEnvironment(unittest.TestCase):
    def assertEnvironmentVariable(self, name, value):
        if name not in os.environ:
            self.fail('environment variable %s not set' % name)

        self.assertEqual(os.environ[name], value)

    def assertModulesLoaded(self, modules):
        for m in modules:
            self.assertTrue(module_present(m))

    def assertModulesNotLoaded(self, modules):
        for m in modules:
            self.assertFalse(module_present(m))

    def setUp(self):
        module_path_add([TEST_MODULES])

        # Always add a base module; this is a workaround for the modules
        # environment's inconsistent behaviour, that starts with an empty
        # LOADEDMODULES variable and ends up removing it completely if all
        # present modules are removed.
        module_load('testmod_base')

        os.environ['_fookey1'] = 'origfoo'
        os.environ['_fookey1b'] = 'foovalue1'
        os.environ['_fookey2b'] = 'foovalue2'
        self.environ_save = EnvironmentSnapshot()
        self.environ = Environment(name='TestEnv1', modules=['testmod_foo'])
        self.environ.set_variable(name='_fookey1', value='value1')
        self.environ.set_variable(name='_fookey2', value='value2')
        self.environ.set_variable(name='_fookey1', value='value3')
        self.environ.set_variable(name='_fookey3b', value='$_fookey1b')
        self.environ.set_variable(name='_fookey4b', value='${_fookey2b}')
        self.environ_other = Environment(name='TestEnv2',
                                         modules=['testmod_boo'])
        self.environ_other.set_variable(name='_fookey11', value='value11')

    def tearDown(self):
        module_path_remove([TEST_MODULES])
        self.environ_save.load()

    def test_setup(self):
        self.assertEqual(len(self.environ.modules), 1)
        self.assertEqual(len(self.environ.variables.keys()), 4)
        self.assertEqual(self.environ.variables['_fookey1'], 'value3')
        self.assertEqual(self.environ.variables['_fookey2'], 'value2')
        self.assertIn('testmod_foo', self.environ.modules)

    def test_environment_snapshot(self):
        self.assertRaises(RuntimeError,
                          self.environ_save.add_module, 'testmod_foo')
        self.assertRaises(RuntimeError, self.environ_save.set_variable,
                          'foo', 'bar')
        self.assertRaises(RuntimeError, self.environ_save.unload)
        self.environ.load()
        self.environ_other.load()
        self.environ_save.load()
        self.assertEqual(self.environ_save, EnvironmentSnapshot())

    def test_load_restore(self):
        self.environ.load()
        self.assertEnvironmentVariable(name='_fookey1', value='value3')
        self.assertEnvironmentVariable(name='_fookey2', value='value2')
        self.assertEnvironmentVariable(name='_fookey3b', value='foovalue1')
        self.assertEnvironmentVariable(name='_fookey4b', value='foovalue2')
        self.assertModulesLoaded(self.environ.modules)
        self.assertTrue(self.environ.loaded)

        self.environ.unload()
        self.assertEqual(self.environ_save, EnvironmentSnapshot())
        self.assertFalse(module_present('testmod_foo'))
        self.assertEnvironmentVariable(name='_fookey1', value='origfoo')

    def test_load_present(self):
        module_load('testmod_boo')
        self.environ.load()
        self.environ.unload()
        self.assertTrue(module_present('testmod_boo'))

    def test_equal(self):
        env1 = Environment('env1', modules=['foo', 'bar'])
        env2 = Environment('env1', modules=['bar', 'foo'])
        self.assertEqual(env1, env2)

    def test_not_equal(self):
        env1 = Environment('env1', modules=['foo', 'bar'])
        env2 = Environment('env2', modules=['foo', 'bar'])
        self.assertNotEqual(env1, env2)

    def test_conflicting_environments(self):
        envfoo = Environment(name='envfoo',
                             modules=['testmod_foo', 'testmod_boo'])
        envbar = Environment(name='envbar', modules=['testmod_bar'])
        envfoo.load()
        envbar.load()
        for m in envbar.modules:
            self.assertTrue(module_present(m))

        for m in envfoo.modules:
            self.assertFalse(module_present(m))

    def test_conflict_environ_after_module_load(self):
        module_load('testmod_foo')
        envfoo = Environment(name='envfoo', modules=['testmod_foo'])
        envfoo.load()
        envfoo.unload()
        self.assertTrue(module_present('testmod_foo'))

    def test_conflict_environ_after_module_force_load(self):
        module_load('testmod_foo')
        envbar = Environment(name='envbar', modules=['testmod_bar'])
        envbar.load()
        envbar.unload()
        self.assertTrue(module_present('testmod_foo'))

    def test_swap(self):
        from reframe.core.environments import swap_environments

        self.environ.load()
        swap_environments(self.environ, self.environ_other)
        self.assertFalse(self.environ.loaded)
        self.assertTrue(self.environ_other.loaded)


class TestProgEnvironment(unittest.TestCase):
    def setUp(self):
        self.environ_save = EnvironmentSnapshot()
        self.executable = os.path.join(TEST_RESOURCES, 'hello')

    def tearDown(self):
        # Remove generated executable ingoring file-not-found errors
        force_remove_file(self.executable)
        self.environ_save.load()

    def assertHelloMessage(self, executable=None):
        if not executable:
            executable = self.executable

        self.assertTrue(os_ext.grep_command_output(cmd=executable,
                                                   pattern='Hello, World\!'))
        force_remove_file(executable)

    def compile_with_env(self, env, skip_fortran=False):
        srcdir = os.path.join(TEST_RESOURCES, 'src')
        env.cxxflags = '-O2'
        env.load()
        env.compile(sourcepath=os.path.join(srcdir, 'hello.c'),
                    executable=self.executable)
        self.assertHelloMessage()

        env.compile(sourcepath=os.path.join(srcdir, 'hello.cpp'),
                    executable=self.executable)
        self.assertHelloMessage()

        if not skip_fortran:
            env.compile(sourcepath=os.path.join(srcdir, 'hello.f90'),
                        executable=self.executable)
            self.assertHelloMessage()

        env.unload()

    def compile_dir_with_env(self, env, skip_fortran=False):
        srcdir = os.path.join(TEST_RESOURCES, 'src')
        env.cxxflags = '-O3'
        env.load()

        executables = ['hello_c', 'hello_cpp']
        if skip_fortran:
            env.compile(srcdir, makefile='Makefile.nofort')
        else:
            env.compile(srcdir)
            executables.append('hello_fort')

        for e in executables:
            self.assertHelloMessage(os.path.join(srcdir, e))

        env.compile(sourcepath=srcdir, options='clean')
        env.unload()

    def test_compile(self):
        # Compile a 'Hello, World' with the builtin gcc/g++
        env = ProgEnvironment(name='builtin-gcc',
                              cc='gcc', cxx='g++', ftn=None)
        try:
            self.compile_with_env(env, skip_fortran=True)
            self.compile_dir_with_env(env, skip_fortran=True)
        except CompilationError as e:
            self.fail("Compilation failed\n")


class TestShellScriptBuilder(unittest.TestCase):
    def setUp(self):
        self.script_file = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        os.chmod(self.script_file.name,
                 os.stat(self.script_file.name).st_mode | stat.S_IEXEC)

    def tearDown(self):
        os.remove(self.script_file.name)

    def test_bash_builder(self):
        builder = BashScriptBuilder()
        builder.set_variable('var1', '13')
        builder.set_variable('var2', '2')
        builder.set_variable('foo', '33', suppress=True)
        builder.verbatim('((var3 = var1 + var2)); echo hello $var3')
        self.script_file.write(builder.finalise())
        self.script_file.close()
        self.assertTrue(
            os_ext.grep_command_output(self.script_file.name, 'hello 15'))
        self.assertFalse(
            os_ext.grep_command_output(self.script_file.name, 'foo'))


class TestModules(unittest.TestCase):
    def setUp(self):
        self.environ_save = EnvironmentSnapshot()
        module_path_add([TEST_MODULES])

    def tearDown(self):
        self.environ_save.load()
        module_unload('testmod_foo')
        module_unload('testmod_bar')
        module_path_remove([TEST_MODULES])

    def test_module_path(self):
        self.assertTrue(os_ext.inpath(TEST_MODULES, os.environ['MODULEPATH']))

        module_path_remove([TEST_MODULES])
        self.assertFalse(os_ext.inpath(TEST_MODULES, os.environ['MODULEPATH']))

    def test_module_equal(self):
        self.assertTrue(module_equal('foo', 'foo'))
        self.assertTrue(module_equal('foo/1.2', 'foo/1.2'))
        self.assertTrue(module_equal('foo', 'foo/1.2'))
        self.assertFalse(module_equal('foo/1.2', 'foo/1.3'))
        self.assertFalse(module_equal('foo', 'bar'))
        self.assertFalse(module_equal('foo', 'foobar'))

    def test_module_load(self):
        self.assertRaises(ModuleError, module_load, 'foo')
        self.assertFalse(module_present('foo'))

        module_load('testmod_foo')
        self.assertTrue(module_present('testmod_foo'))
        self.assertIn('TESTMOD_FOO', os.environ)

        module_unload('testmod_foo')
        self.assertFalse(module_present('testmod_foo'))
        self.assertNotIn('TESTMOD_FOO', os.environ)

    def test_module_force_load(self):
        module_load('testmod_foo')

        unloaded = module_force_load('testmod_foo')
        self.assertEqual(len(unloaded), 0)
        self.assertTrue(module_present('testmod_foo'))

        unloaded = module_force_load('testmod_bar')
        self.assertTrue(module_present('testmod_bar'))
        self.assertFalse(module_present('testmod_foo'))
        self.assertIn('testmod_foo', unloaded)
        self.assertIn('TESTMOD_BAR', os.environ)

    def test_module_purge(self):
        module_load('testmod_base')
        module_purge()
        self.assertNotIn('LOADEDMODULES', os.environ)


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
        self.assertIn('unittests.test_core', rep)
        self.assertIn('_a=%r' % c._a, rep)
        self.assertIn('b=%r' % c.b, rep)
        self.assertIn('D(...)', rep)
