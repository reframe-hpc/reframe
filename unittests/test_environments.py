# Copyright 2016-2023 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import os
import pytest

import reframe.core.environments as env
import reframe.core.runtime as rt
import unittests.utility as test_util
from reframe.core.exceptions import EnvironError


@pytest.fixture
def base_environ(monkeypatch):
    monkeypatch.setenv('_var0', 'val0')
    monkeypatch.setenv('_var1', 'val1')
    environ_save = env.snapshot()
    yield environ_save
    environ_save.restore()


@pytest.fixture
def modules_system():
    if not test_util.has_sane_modules_system():
        pytest.skip('no modules system configured')

    modsys = rt.runtime().modules_system
    modsys.searchpath_add(test_util.TEST_MODULES)

    # Always add a base module; this is a workaround for the modules
    # environment's inconsistent behaviour, that starts with an empty
    # LOADEDMODULES variable and ends up removing it completely if all
    # present modules are removed.
    modsys.load_module('testmod_base')
    yield modsys
    modsys.searchpath_remove(test_util.TEST_MODULES)


@pytest.fixture
def user_runtime(make_exec_ctx_g):
    if test_util.USER_CONFIG_FILE:
        yield from make_exec_ctx_g(test_util.USER_CONFIG_FILE,
                                   test_util.USER_SYSTEM)
    else:
        yield from make_exec_ctx_g()


@pytest.fixture
def env0():
    return env.Environment(
        'TestEnv1', ['testmod_foo'],
        [('_var0', 'val1'), ('_var2', '$_var0'), ('_var3', '${_var1}')],
        {'foo': 1, 'bar': 2},
        ['feat1', 'feat2'],
        ['echo prep1', 'echo prep2']
    )


@pytest.fixture
def env1():
    return env.Environment('TestEnv2', ['testmod_boo'], {'_var4': 'val4'})


def assert_modules_loaded(modules):
    modsys = rt.runtime().modules_system
    for m in modules:
        assert modsys.is_module_loaded(m)


def test_env_construction(base_environ, env0):
    assert len(env0.modules) == 1
    assert 'testmod_foo' in env0.modules
    assert len(env0.env_vars.keys()) == 3
    assert env0.env_vars['_var0'] == 'val1'

    # No variable expansion, if environment is not loaded
    assert env0.env_vars['_var2'] == '$_var0'
    assert env0.env_vars['_var3'] == '${_var1}'

    # Assert extras
    assert env0.extras == {'foo': 1, 'bar': 2}

    # Assert features
    assert env0.features == ['feat1', 'feat2']

    # Assert prepare_cmds
    assert env0.prepare_cmds == ['echo prep1', 'echo prep2']


def test_progenv_construction():
    environ = env.ProgEnvironment('myenv',
                                  modules=['modfoo'],
                                  env_vars=[('var', 'val')],
                                  extras={'foo': 'bar'},
                                  prepare_cmds=['echo prep1', 'echo prep2'])
    assert environ.name == 'myenv'
    assert environ.modules == ['modfoo']
    assert environ.env_vars == {'var': 'val'}
    assert environ.extras == {'foo': 'bar'}
    assert environ.prepare_cmds == ['echo prep1', 'echo prep2']


def test_env_snapshot(base_environ, env0, env1):
    rt.loadenv(env0, env1)
    base_environ.restore()
    assert base_environ == env.snapshot()
    assert not rt.is_env_loaded(env0)
    assert not rt.is_env_loaded(env1)


def test_env_load_restore(base_environ, env0):
    snapshot, _ = rt.loadenv(env0)
    assert os.environ['_var0'] == 'val1'
    assert os.environ['_var1'] == 'val1'
    assert os.environ['_var2'] == 'val1'
    assert os.environ['_var3'] == 'val1'
    if test_util.has_sane_modules_system():
        assert_modules_loaded(env0.modules)

    assert rt.is_env_loaded(env0)
    snapshot.restore()
    base_environ == env.snapshot()
    assert os.environ['_var0'] == 'val0'
    if test_util.has_sane_modules_system():
        assert not rt.runtime().modules_system.is_module_loaded('testmod_foo')

    assert not rt.is_env_loaded(env0)


def test_temp_environment(base_environ, user_runtime, modules_system):
    with rt.temp_environment(
            ['testmod_foo'], {'_var0': 'val2', '_var3': 'val3'}
    ) as environ:
        assert rt.is_env_loaded(environ)

    assert not rt.is_env_loaded(environ)


def test_env_load_already_present(base_environ, user_runtime,
                                  modules_system, env0):
    modules_system.load_module('testmod_boo')
    snapshot, _ = rt.loadenv(env0)
    snapshot.restore()
    assert modules_system.is_module_loaded('testmod_boo')


def test_env_load_non_overlapping(base_environ):
    e0 = env.Environment(name='e0', env_vars=[('a', '1'), ('b', '2')])
    e1 = env.Environment(name='e1', env_vars=[('c', '3'), ('d', '4')])
    rt.loadenv(e0, e1)
    assert rt.is_env_loaded(e0)
    assert rt.is_env_loaded(e1)


def test_load_overlapping(base_environ):
    e0 = env.Environment(name='e0', env_vars=[('a', '1'), ('b', '2')])
    e1 = env.Environment(name='e1', env_vars=[('b', '3'), ('c', '4')])
    rt.loadenv(e0, e1)
    assert not rt.is_env_loaded(e0)
    assert rt.is_env_loaded(e1)


def test_env_equal(base_environ):
    env1 = env.Environment('env1', modules=['foo', 'bar'],
                           env_vars=[('a', '1')])
    env2 = env.Environment('env1', modules=['bar', 'foo'],
                           env_vars=[('a', 1)])

    # Environments variables must be checked against their string
    # representation
    assert env1 == env2
    assert env2 == env1


def test_env_not_equal(base_environ):
    env1 = env.Environment('env1', modules=['foo', 'bar'])
    env2 = env.Environment('env2', modules=['foo', 'bar'])
    assert env1 != env2

    # Variables are ordered, because they might depend on each other
    env1 = env.Environment('env1', env_vars=[('a', 1), ('b', 2)])
    env2 = env.Environment('env1', env_vars=[('b', 2), ('a', 1)])
    assert env1 != env2


def test_env_conflict(base_environ, user_runtime, modules_system):
    env0 = env.Environment('env0', ['testmod_foo', 'testmod_boo'])
    env1 = env.Environment('env1', ['testmod_bar'])
    rt.loadenv(env0, env1)
    for m in env1.modules:
        assert modules_system.is_module_loaded(m)

    for m in env0.modules:
        assert not modules_system.is_module_loaded(m)


def test_env_conflict_after_module_load(base_environ,
                                        user_runtime, modules_system):
    modules_system.load_module('testmod_foo')
    env0 = env.Environment('env0', ['testmod_foo'])
    snapshot, _ = rt.loadenv(env0)
    snapshot.restore()
    assert modules_system.is_module_loaded('testmod_foo')


def test_env_conflict_after_module_load_force(base_environ,
                                              user_runtime, modules_system):
    modules_system.load_module('testmod_foo')
    env0 = env.Environment(name='env0', modules=['testmod_bar'])
    snapshot, _ = rt.loadenv(env0)
    snapshot.restore()
    assert modules_system.is_module_loaded('testmod_foo')


def test_env_immutability(base_environ, env0):
    # Check emitted commands
    _, commands = rt.loadenv(env0)

    # Try to modify the returned list of commands
    commands.append('foo')
    assert 'foo' not in rt.loadenv(env0)[1]

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


def test_emit_loadenv_commands(base_environ, user_runtime,
                               modules_system, env0):
    ms = rt.runtime().modules_system
    expected_commands = [
        *env0.prepare_cmds,
        ms.emit_load_commands('testmod_foo')[0],
        'export _var0=val1',
        'export _var2=$_var0',
        'export _var3=${_var1}',
    ]
    assert expected_commands == rt.emit_loadenv_commands(env0)


def test_emit_loadenv_nomod_commands(base_environ, env0):
    expected_commands = [
        *env0.prepare_cmds,
        'export _var0=val1',
        'export _var2=$_var0',
        'export _var3=${_var1}',
    ]
    assert expected_commands == rt.emit_loadenv_commands(env0)


def test_emit_loadenv_commands_ignore_confict(base_environ,
                                              make_exec_ctx, env0):
    if not test_util.has_sane_modules_system():
        pytest.skip('no modules system configured')

    if test_util.USER_CONFIG_FILE:
        make_exec_ctx(test_util.USER_CONFIG_FILE, test_util.USER_SYSTEM,
                      options={'general/resolve_module_conflicts': False})
    else:
        make_exec_ctx(options={'general/resolve_module_conflicts': False})

    # Load a conflicting module
    ms = rt.runtime().modules_system
    with ms.change_module_path(test_util.TEST_MODULES):
        ms.load_module('testmod_bar')
        expected_commands = [
            *env0.prepare_cmds,
            ms.emit_load_commands('testmod_foo')[0],
            'export _var0=val1',
            'export _var2=$_var0',
            'export _var3=${_var1}',
        ]
        assert expected_commands == rt.emit_loadenv_commands(env0)


def test_emit_loadenv_commands_with_confict(base_environ, user_runtime,
                                            modules_system, env0):
    # Load a conflicting module
    modules_system.load_module('testmod_bar')
    ms = rt.runtime().modules_system
    expected_commands = [
        *env0.prepare_cmds,
        ms.emit_unload_commands('testmod_bar')[0],
        ms.emit_load_commands('testmod_foo')[0],
        'export _var0=val1',
        'export _var2=$_var0',
        'export _var3=${_var1}',
    ]
    assert expected_commands == rt.emit_loadenv_commands(env0)


def test_emit_loadenv_commands_mapping_with_conflict(base_environ,
                                                     user_runtime,
                                                     modules_system):
    if modules_system.name == 'tmod4':
        pytest.skip('test scenario not valid for tmod4')

    e0 = env.Environment(name='e0', modules=['testmod_ext'])
    ms = rt.runtime().modules_system
    ms.load_mapping('testmod_ext: testmod_ext testmod_foo')
    expected_commands = [
        ms.emit_load_commands('testmod_ext')[0],
        ms.emit_unload_commands('testmod_bar')[0],
        ms.emit_load_commands('testmod_foo')[0],
    ]
    assert expected_commands == rt.emit_loadenv_commands(e0)


def test_emit_loadenv_failure(user_runtime):
    snap = rt.snapshot()
    environ = env.Environment('test', modules=['testmod_foo', 'testmod_xxx'])

    # Suppress the module load error and verify that the original environment
    # is preserved
    with contextlib.suppress(EnvironError):
        rt.emit_loadenv_commands(environ)

    assert rt.snapshot() == snap
