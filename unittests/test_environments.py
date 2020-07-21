# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import pytest

import reframe.core.environments as env
import reframe.core.runtime as rt
import reframe.utility.os_ext as os_ext
import unittests.fixtures as fixtures
from reframe.core.exceptions import EnvironError


@pytest.fixture
def environ_save(monkeypatch):
    monkeypatch.setenv('_var0', 'val0')
    monkeypatch.setenv('_var1', 'val1')
    environ_save = env.snapshot()
    yield environ_save
    environ_save.restore()


@pytest.fixture
def setup_modules_system(environ_save):
    modules_system = None

    def _setup_modules_system():
        if not fixtures.has_sane_modules_system():
            pytest.skip('no modules system configured')

        modules_system = rt.runtime().modules_system
        modules_system.searchpath_add(fixtures.TEST_MODULES)

        # Always add a base module; this is a workaround for the modules
        # environment's inconsistent behaviour, that starts with an empty
        # LOADEDMODULES variable and ends up removing it completely if all
        # present modules are removed.
        modules_system.load_module('testmod_base')

        return modules_system

    yield _setup_modules_system

    if modules_system:
        modules_system.searchpath_remove(fixtures.TEST_MODULES)


@pytest.fixture
def environ(environ_save):
    return env.Environment(name='TestEnv1',
                           modules=['testmod_foo'],
                           variables=[('_var0', 'val1'),
                                      ('_var2', '$_var0'),
                                      ('_var3', '${_var1}')])


@pytest.fixture
def environ_other(environ_save):
    return env.Environment(name='TestEnv2',
                           modules=['testmod_boo'],
                           variables={'_var4': 'val4'})


def test_setup(environ):
    if fixtures.has_sane_modules_system():
        assert len(environ.modules) == 1
        assert 'testmod_foo' in environ.modules

    assert len(environ.variables.keys()) == 3
    assert environ.variables['_var0'] == 'val1'

    # No variable expansion, if environment is not loaded
    assert environ.variables['_var2'] == '$_var0'
    assert environ.variables['_var3'] == '${_var1}'


def test_environ_snapshot(environ_save, environ, environ_other):
    rt.loadenv(environ, environ_other)
    environ_save.restore()
    assert environ_save == env.snapshot()
    assert not rt.is_env_loaded(environ)
    assert not rt.is_env_loaded(environ_other)


def test_load_restore(environ_save, environ):
    snapshot, _ = rt.loadenv(environ)
    os.environ['_var0'] == 'val1'
    os.environ['_var1'] == 'val1'
    os.environ['_var2'] == 'val1'
    os.environ['_var3'] == 'val1'
    if fixtures.has_sane_modules_system():
        for m in environ.modules:
            assert rt.runtime().modules_system.is_module_loaded(m)

    assert rt.is_env_loaded(environ)
    snapshot.restore()
    environ_save == env.snapshot()
    os.environ['_var0'], 'val0'
    if fixtures.has_sane_modules_system():
        assert rt.runtime().modules_system.is_module_loaded('testmod_foo')

    assert not rt.is_env_loaded(environ)


@fixtures.switch_to_user_runtime
def test_temp_environment(setup_modules_system):
    setup_modules_system()
    with rt.temp_environment(
            ['testmod_foo'], {'_var0': 'val2', '_var3': 'val3'}
    ) as environ:
        assert rt.is_env_loaded(environ)

    assert not rt.is_env_loaded(environ)


@fixtures.switch_to_user_runtime
def test_load_already_present(setup_modules_system, environ):
    modules_system = setup_modules_system()
    modules_system.load_module('testmod_boo')
    snapshot, _ = rt.loadenv(environ)
    snapshot.restore()
    assert modules_system.is_module_loaded('testmod_boo')


def test_load_non_overlapping():
    e0 = env.Environment(name='e0', variables=[('a', '1'), ('b', '2')])
    e1 = env.Environment(name='e1', variables=[('c', '3'), ('d', '4')])
    rt.loadenv(e0, e1)
    assert rt.is_env_loaded(e0)
    assert rt.is_env_loaded(e1)


def test_load_overlapping():
    e0 = env.Environment(name='e0', variables=[('a', '1'), ('b', '2')])
    e1 = env.Environment(name='e1', variables=[('b', '3'), ('c', '4')])
    rt.loadenv(e0, e1)
    assert not rt.is_env_loaded(e0)
    assert rt.is_env_loaded(e1)


def test_equal():
    env1 = env.Environment('env1', modules=['foo', 'bar'])
    env2 = env.Environment('env1', modules=['bar', 'foo'])
    assert env1 == env2
    assert env2 == env1


def test_not_equal():
    env1 = env.Environment('env1', modules=['foo', 'bar'])
    env2 = env.Environment('env2', modules=['foo', 'bar'])
    assert env1 != env2

    # Variables are ordered, because they might depend on each other
    env1 = env.Environment('env1', variables=[('a', 1), ('b', 2)])
    env2 = env.Environment('env1', variables=[('b', 2), ('a', 1)])
    assert env1 != env2


@fixtures.switch_to_user_runtime
def test_conflicting_environments(setup_modules_system):
    modules_system = setup_modules_system()
    envfoo = env.Environment(name='envfoo',
                             modules=['testmod_foo', 'testmod_boo'])
    envbar = env.Environment(name='envbar', modules=['testmod_bar'])
    rt.loadenv(envfoo, envbar)
    for m in envbar.modules:
        assert modules_system.is_module_loaded(m)

    for m in envfoo.modules:
        assert not modules_system.is_module_loaded(m)


@fixtures.switch_to_user_runtime
def test_conflict_environ_after_module_load(setup_modules_system):
    modules_system = setup_modules_system()
    modules_system.load_module('testmod_foo')
    envfoo = env.Environment(name='envfoo', modules=['testmod_foo'])
    snapshot, _ = rt.loadenv(envfoo)
    snapshot.restore()
    assert modules_system.is_module_loaded('testmod_foo')


@fixtures.switch_to_user_runtime
def test_conflict_environ_after_module_force_load(setup_modules_system):
    modules_system = setup_modules_system()
    modules_system.load_module('testmod_foo')
    envbar = env.Environment(name='envbar', modules=['testmod_bar'])
    snapshot, _ = rt.loadenv(envbar)
    snapshot.restore()
    assert modules_system.is_module_loaded('testmod_foo')


def test_immutability(environ):
    # Check emit_load_commands()
    _, commands = rt.loadenv(environ)

    # Try to modify the returned list of commands
    commands.append('foo')
    assert 'foo' not in rt.loadenv(environ)[1]

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
def test_emit_load_commands(environ, setup_modules_system):
    setup_modules_system()
    ms = rt.runtime().modules_system
    expected_commands = [
        ms.emit_load_commands('testmod_foo')[0],
        'export _var0=val1',
        'export _var2=$_var0',
        'export _var3=${_var1}',
    ]
    assert expected_commands == rt.emit_loadenv_commands(environ)


@fixtures.switch_to_user_runtime
def test_emit_load_commands_with_confict(setup_modules_system, environ):
    modules_system = setup_modules_system()

    # Load a conflicting module
    modules_system.load_module('testmod_bar')
    ms = rt.runtime().modules_system
    expected_commands = [
        ms.emit_unload_commands('testmod_bar')[0],
        ms.emit_load_commands('testmod_foo')[0],
        'export _var0=val1',
        'export _var2=$_var0',
        'export _var3=${_var1}',
    ]
    assert expected_commands == rt.emit_loadenv_commands(environ)
