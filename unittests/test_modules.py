# Copyright 2016-2023 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import pytest

import reframe.core.environments as env
import reframe.core.modules as modules
import unittests.utility as test_util
from reframe.core.exceptions import ConfigError, EnvironError


@pytest.fixture(params=['tmod', 'tmod4', 'lmod', 'spack', 'nomod'])
def modules_system_nopath(request, monkeypatch):
    # Always pretend to be on a clean modules environment
    monkeypatch.setenv('MODULEPATH', '')
    monkeypatch.setenv('LOADEDMODULES', '')
    monkeypatch.setenv('_LMFILES_', '')
    args = [request.param] if request.param != 'nomod' else []
    try:
        m = modules.ModulesSystem.create(*args)
    except ConfigError:
        pytest.skip(f'{request.param} not supported')

    environ_save = env.snapshot()
    yield m
    environ_save.restore()


@pytest.fixture
def modules_system(modules_system_nopath):
    m = modules_system_nopath
    environ_save = env.snapshot()
    with m.change_module_path(test_util.TEST_MODULES):
        yield m

    environ_save.restore()


def test_searchpath(modules_system):
    if modules_system.name in ['nomod', 'spack']:
        # Simply test that no exceptions are thrown
        modules_system.searchpath_remove(test_util.TEST_MODULES)
    else:
        assert test_util.TEST_MODULES in modules_system.searchpath

        modules_system.searchpath_remove(test_util.TEST_MODULES)
        assert test_util.TEST_MODULES not in modules_system.searchpath


@pytest.fixture
def module_collection(modules_system, tmp_path, monkeypatch):
    if modules_system.name not in ['tmod4', 'lmod']:
        pytest.skip(f'test unsupported with {modules_system.name!r}')

    # Both Lmod and Tmod4 place collections under the user's HOME directory,
    # that's why we monkeypatch it
    monkeypatch.setenv('HOME', str(tmp_path))
    coll_name = 'test_collection'

    # Lmod uses LMOD_SYSTEM_NAME to differentiate between collections of
    # different systems, which is added as a postfix to the collection name.
    # We monkeypatch it since the test directory is temporary
    if modules_system.name == 'lmod':
        monkeypatch.delenv('LMOD_SYSTEM_NAME', raising=False)

    # Create modules collections with conflicting modules
    modules_system.load_module('testmod_base')
    modules_system.load_module('testmod_foo')
    modules_system.execute('save', coll_name)
    modules_system.unload_module('testmod_foo')
    modules_system.unload_module('testmod_base')

    yield coll_name

    # Remove the temporary collection
    if modules_system.name == 'lmod':
        prefix = os.path.join(os.environ['HOME'], '.lmod.d')
    else:
        prefix = os.path.join(os.environ['HOME'], '.module')

    os.remove(os.path.join(prefix, coll_name))


def test_module_load(modules_system):
    if modules_system.name in ['nomod', 'spack']:
        modules_system.load_module('foo')
        modules_system.unload_module('foo')
    else:
        with pytest.raises(EnvironError):
            modules_system.load_module('foo')

        assert not modules_system.is_module_loaded('foo')
        assert 'foo' not in modules_system.loaded_modules()

        modules_system.load_module('testmod_foo')
        assert modules_system.is_module_loaded('testmod_foo')
        assert 'testmod_foo' in modules_system.loaded_modules()
        assert 'TESTMOD_FOO' in os.environ

        modules_system.unload_module('testmod_foo')
        assert not modules_system.is_module_loaded('testmod_foo')
        assert 'testmod_foo' not in modules_system.loaded_modules()
        assert 'TESTMOD_FOO' not in os.environ


def test_module_load_with_path(modules_system_nopath):
    if modules_system_nopath.name in ['nomod', 'spack']:
        pytest.skip('module load not supported')

    modules_system_nopath.load_module('testmod_foo',
                                      path=test_util.TEST_MODULES)
    with modules_system_nopath.change_module_path(test_util.TEST_MODULES):
        assert modules_system_nopath.is_module_loaded('testmod_foo')
        assert 'testmod_foo' in modules_system_nopath.loaded_modules()
        assert 'TESTMOD_FOO' in os.environ


def test_module_load_collection(modules_system, module_collection):
    # Load a conflicting module first
    modules_system.load_module('testmod_bar')
    modules_system.load_module(module_collection, collection=True)
    assert modules_system.is_module_loaded('testmod_base')
    assert modules_system.is_module_loaded('testmod_foo')


def test_module_load_force(modules_system):
    if modules_system.name in ['nomod', 'spack']:
        modules_system.load_module('foo', force=True)
    else:
        modules_system.load_module('testmod_foo')

        unloaded = modules_system.load_module('testmod_foo', force=True)
        assert unloaded == [('testmod_foo', [])]
        assert modules_system.is_module_loaded('testmod_foo')

        unloaded = modules_system.load_module('testmod_bar', force=True)
        assert modules_system.is_module_loaded('testmod_bar')
        assert not modules_system.is_module_loaded('testmod_foo')
        assert unloaded == [('testmod_bar', ['testmod_foo'])]
        assert 'TESTMOD_BAR' in os.environ


def test_module_load_force_collection(modules_system, module_collection):
    # Load a conflicting module first
    modules_system.load_module('testmod_bar')
    unloaded = modules_system.load_module(module_collection,
                                          collection=True, force=True)
    assert unloaded == [('test_collection', [])]
    assert modules_system.is_module_loaded('testmod_base')
    assert modules_system.is_module_loaded('testmod_foo')


def test_module_load_collection_searchpath(modules_system, tmpdir,
                                           module_collection):
    p1 = str(tmpdir.mkdir('path1'))
    p2 = str(tmpdir.mkdir('path2'))
    modules_system.searchpath_add(p1)
    modules_system.searchpath_add(p2)
    modules_system.searchpath_remove(p1)
    modules_system.load_module(module_collection, collection=True)
    assert p1 not in modules_system.searchpath
    assert p2 in modules_system.searchpath


def test_module_unload_all(modules_system):
    if modules_system.name == 'nomod':
        modules_system.unload_all()
    else:
        modules_system.load_module('testmod_base')
        modules_system.unload_all()
        assert 0 == len(modules_system.loaded_modules())


def test_module_list(modules_system):
    if modules_system.name == 'nomod':
        assert 0 == len(modules_system.loaded_modules())
    elif modules_system.name == 'spack':
        # If Spack is installed, we can't be sure that the user has not loaded
        # any module and we cannot unload them in here, since we don't have
        # Python bindings. So we only check that we get a list back.
        assert isinstance(modules_system.loaded_modules(), list)
    else:
        modules_system.load_module('testmod_foo')
        assert 'testmod_foo' in modules_system.loaded_modules()
        modules_system.unload_module('testmod_foo')


def test_module_conflict_list(modules_system):
    if modules_system.name in ['nomod', 'spack']:
        assert 0 == len(modules_system.conflicted_modules('foo'))
    else:
        conflict_list = modules_system.conflicted_modules('testmod_bar')
        assert 'testmod_foo' in conflict_list
        assert 'testmod_boo' in conflict_list


def test_module_available_all(modules_system):
    modules = sorted(modules_system.available_modules())
    if modules_system.name == 'nomod':
        assert modules == []
    elif modules_system.name == 'spack':
        # If Spack is installed, we can't fool it with environment variables
        # about its installed packages, like we can do with modules, so we
        # simply check that we get a list back.
        assert isinstance(modules, list)
    else:
        assert (modules == ['testmod_bar', 'testmod_base',
                            'testmod_boo', 'testmod_ext', 'testmod_foo'])


def _test_module_available_substr(modules_system):
    modules = sorted(modules_system.available_modules('testmod_b'))
    if modules_system.name == 'nomod':
        assert modules == []
    else:
        assert (modules == ['testmod_bar', 'testmod_base', 'testmod_boo'])


@test_util.dispatch('modules_system', suffix=lambda ms: ms.name)
def test_emit_load_commands(modules_system):
    modules_system.module_map = {
        'm0': ['m1', 'm2']
    }
    yield


def _emit_load_commands_tmod(modules_system):
    emit_cmds = modules_system.emit_load_commands
    assert emit_cmds('foo') == ['module load foo']
    assert emit_cmds('foo/1.2') == ['module load foo/1.2']

    # Module mappings are not taking into account since v3.3
    assert emit_cmds('m0') == ['module load m0']


def _emit_load_commands_tmod4(modules_system):
    emit_cmds = modules_system.emit_load_commands
    assert emit_cmds('foo') == ['module load foo']
    assert emit_cmds('foo', collection=True) == [
        'module restore foo', f'module use {test_util.TEST_MODULES}'
    ]
    assert emit_cmds('foo/1.2') == ['module load foo/1.2']
    if modules_system.name == 'lmod':
        assert emit_cmds('foo', path='/path') == ['module use /path',
                                                  'module load foo']
    else:
        assert emit_cmds('foo', path='/path') == ['module use /path',
                                                  'module load foo',
                                                  'module unuse /path']

    # Module mappings are not taking into account since v3.3
    assert emit_cmds('m0') == ['module load m0']
    assert emit_cmds('m0', collection=True) == [
        'module restore m0', f'module use {test_util.TEST_MODULES}'
    ]


def _emit_load_commands_lmod(modules_system):
    return _emit_load_commands_tmod4(modules_system)


def _emit_load_commands_spack(modules_system):
    emit_cmds = modules_system.emit_load_commands
    assert emit_cmds('foo') == ['spack load foo']
    assert emit_cmds('foo/1.2') == ['spack load foo/1.2']

    # Module mappings are not taking into account since v3.3
    assert emit_cmds('m0') == ['spack load m0']


def _emit_load_commands_nomod(modules_system):
    emit_cmds = modules_system.emit_load_commands
    assert emit_cmds('foo') == []
    assert emit_cmds('foo/1.2') == []
    assert emit_cmds('m0') == []


@test_util.dispatch('modules_system', suffix=lambda ms: ms.name)
def test_emit_unload_commands(modules_system):
    modules_system.module_map = {
        'm0': ['m1', 'm2']
    }
    yield


def _emit_unload_commands_tmod(modules_system):
    emit_cmds = modules_system.emit_unload_commands
    assert emit_cmds('foo') == ['module unload foo']
    assert emit_cmds('foo/1.2') == ['module unload foo/1.2']

    # Module mappings are not taking into account since v3.3
    assert emit_cmds('m0') == ['module unload m0']


def _emit_unload_commands_tmod4(modules_system):
    emit_cmds = modules_system.emit_unload_commands
    assert emit_cmds('foo') == ['module unload foo']
    assert emit_cmds('foo', collection=True) == []
    assert emit_cmds('foo/1.2') == ['module unload foo/1.2']
    assert emit_cmds('m0', collection=True) == []

    # Module mappings are not taking into account since v3.3
    assert emit_cmds('m0') == ['module unload m0']


def _emit_unload_commands_lmod(modules_system):
    emit_cmds = modules_system.emit_unload_commands
    assert emit_cmds('foo') == ['module unload foo']
    assert emit_cmds('foo/1.2') == ['module unload foo/1.2']

    # Module mappings are not taking into account since v3.3
    assert emit_cmds('m0') == ['module unload m0']


def _emit_unload_commands_spack(modules_system):
    emit_cmds = modules_system.emit_unload_commands
    assert emit_cmds('foo') == ['spack unload foo']
    assert emit_cmds('foo/1.2') == ['spack unload foo/1.2']

    # Module mappings are not taking into account since v3.3
    assert emit_cmds('m0') == ['spack unload m0']


def _emit_unload_commands_nomod(modules_system):
    emit_cmds = modules_system.emit_unload_commands
    assert emit_cmds('foo') == []
    assert emit_cmds('foo/1.2') == []
    assert emit_cmds('m0') == []


def test_module_construction():
    m = modules.Module('foo/1.2')
    assert m.name == 'foo'
    assert m.version == '1.2'
    assert m.collection is False
    with pytest.raises(ValueError):
        modules.Module('')

    with pytest.raises(ValueError):
        modules.Module(' ')

    with pytest.raises(TypeError):
        modules.Module(None)

    with pytest.raises(TypeError):
        modules.Module(23)

    m = modules.Module('foo/1.2', collection=True)
    assert m.collection is True

    m = modules.Module('foo/1.2', path='/foo/modules')
    assert m.path == '/foo/modules'
    assert m.collection is False


def test_module_equal():
    assert modules.Module('foo') == modules.Module('foo')
    assert modules.Module('foo/1.2') == modules.Module('foo/1.2')
    assert modules.Module('foo') == modules.Module('foo/1.2')
    assert hash(modules.Module('foo')) == hash(modules.Module('foo'))
    assert (hash(modules.Module('foo/1.2')) ==
            hash(modules.Module('foo/1.2')))
    assert hash(modules.Module('foo')) == hash(modules.Module('foo/1.2'))
    assert modules.Module('foo/1.2') != modules.Module('foo/1.3')
    assert modules.Module('foo') != modules.Module('bar')
    assert modules.Module('foo') != modules.Module('foobar')
    assert modules.Module('foo') != modules.Module('foo', collection=True)
    assert modules.Module('foo') != modules.Module('foo', path='/foo/modules')


@pytest.fixture
def modules_system_emu():
    class ModulesSystemEmulator(modules.ModulesSystemImpl):
        '''A convenience class that simulates a modules system.'''

        def __init__(self):
            self._loaded_modules = set()

            # The following two variables record the sequence of loads and
            # unloads
            self.load_seq = []
            self.unload_seq = []

        def loaded_modules(self):
            return list(self._loaded_modules)

        def conflicted_modules(self, module):
            return []

        def _execute(self, cmd, *args):
            return ''

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

        def available_modules(self, substr):
            return []

        def name(self):
            return 'nomod_debug'

        def version(self):
            return '1.0'

        @property
        def modulecmd(self):
            return ''

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

    return modules.ModulesSystem(ModulesSystemEmulator())


def test_mapping_simple(modules_system_emu):
    #
    # m0 -> m1
    #
    modules_system_emu.module_map = {'m0': ['m1']}
    modules_system_emu.load_module('m0')
    assert modules_system_emu.is_module_loaded('m0')
    assert modules_system_emu.is_module_loaded('m1')
    assert ['m1'] == modules_system_emu.backend.load_seq

    # Unload module
    modules_system_emu.unload_module('m1')
    assert not modules_system_emu.is_module_loaded('m0')
    assert not modules_system_emu.is_module_loaded('m1')


def test_mapping_chain(modules_system_emu):
    #
    # m0 -> m1 -> m2
    #
    modules_system_emu.module_map = {
        'm0': ['m1'],
        'm1': ['m2']
    }
    modules_system_emu.load_module('m0')
    assert modules_system_emu.is_module_loaded('m0')
    assert modules_system_emu.is_module_loaded('m1')
    assert modules_system_emu.is_module_loaded('m2')
    assert ['m2'] == modules_system_emu.backend.load_seq

    # Unload module
    modules_system_emu.unload_module('m1')
    assert not modules_system_emu.is_module_loaded('m0')
    assert not modules_system_emu.is_module_loaded('m1')
    assert not modules_system_emu.is_module_loaded('m2')


def test_mapping_n_to_one(modules_system_emu):
    #
    # m0 -> m2 <- m1
    #
    modules_system_emu.module_map = {
        'm0': ['m2'],
        'm1': ['m2']
    }
    modules_system_emu.load_module('m0')
    assert modules_system_emu.is_module_loaded('m0')
    assert modules_system_emu.is_module_loaded('m1')
    assert modules_system_emu.is_module_loaded('m2')
    assert ['m2'] == modules_system_emu.backend.load_seq

    # Unload module
    modules_system_emu.unload_module('m0')
    assert not modules_system_emu.is_module_loaded('m0')
    assert not modules_system_emu.is_module_loaded('m1')
    assert not modules_system_emu.is_module_loaded('m2')


def test_mapping_one_to_n(modules_system_emu):
    #
    # m2 <- m0 -> m1
    #
    modules_system_emu.module_map = {
        'm0': ['m1', 'm2'],
    }
    modules_system_emu.load_module('m0')
    assert modules_system_emu.is_module_loaded('m0')
    assert modules_system_emu.is_module_loaded('m1')
    assert modules_system_emu.is_module_loaded('m2')
    assert ['m1', 'm2'] == modules_system_emu.backend.load_seq

    # m0 is loaded only if m1 and m2 are.
    modules_system_emu.unload_module('m2')
    assert not modules_system_emu.is_module_loaded('m0')
    assert modules_system_emu.is_module_loaded('m1')


def test_mapping_deep_dfs_order(modules_system_emu):
    #
    #    -- > m1 ---- > m3
    #   /       \
    # m0         \
    #   \         \
    #    -- > m2   -- > m4
    #
    modules_system_emu.module_map = {
        'm0': ['m1', 'm2'],
        'm1': ['m3', 'm4']
    }
    modules_system_emu.load_module('m0')
    assert modules_system_emu.is_module_loaded('m0')
    assert modules_system_emu.is_module_loaded('m1')
    assert modules_system_emu.is_module_loaded('m2')
    assert modules_system_emu.is_module_loaded('m3')
    assert modules_system_emu.is_module_loaded('m4')
    assert ['m3', 'm4', 'm2'] == modules_system_emu.backend.load_seq

    # Test unloading
    modules_system_emu.unload_module('m2')
    assert not modules_system_emu.is_module_loaded('m0')
    assert modules_system_emu.is_module_loaded('m1')
    assert not modules_system_emu.is_module_loaded('m2')
    assert modules_system_emu.is_module_loaded('m3')
    assert modules_system_emu.is_module_loaded('m4')


def test_mapping_deep_dfs_unload_order(modules_system_emu):
    #
    #    -- > m1 ---- > m3
    #   /       \
    # m0         \
    #   \         \
    #    -- > m2   -- > m4
    #
    modules_system_emu.module_map = {
        'm0': ['m1', 'm2'],
        'm1': ['m3', 'm4']
    }
    modules_system_emu.load_module('m0')
    modules_system_emu.unload_module('m0')
    assert ['m2', 'm4', 'm3'] == modules_system_emu.backend.unload_seq


def test_mapping_multiple_paths(modules_system_emu):
    #
    #    -- > m1
    #   /     ^
    # m0      |
    #   \     |
    #    -- > m2
    #
    modules_system_emu.module_map = {
        'm0': ['m1', 'm2'],
        'm2': ['m1'],
    }
    modules_system_emu.load_module('m0')
    assert modules_system_emu.is_module_loaded('m0')
    assert modules_system_emu.is_module_loaded('m1')
    assert modules_system_emu.is_module_loaded('m2')
    assert ['m1'] == modules_system_emu.backend.load_seq

    # Test unloading
    modules_system_emu.unload_module('m2')
    assert not modules_system_emu.is_module_loaded('m0')
    assert not modules_system_emu.is_module_loaded('m1')
    assert not modules_system_emu.is_module_loaded('m2')


def test_mapping_deep_multiple_paths(modules_system_emu):
    #
    #    -- > m1 ---- > m3
    #   /     ^ \
    # m0      |  \
    #   \     |   \
    #    -- > m2   -- > m4
    #
    modules_system_emu.module_map = {
        'm0': ['m1', 'm2'],
        'm1': ['m3', 'm4'],
        'm2': ['m1']
    }
    modules_system_emu.load_module('m0')
    assert modules_system_emu.is_module_loaded('m0')
    assert modules_system_emu.is_module_loaded('m1')
    assert modules_system_emu.is_module_loaded('m2')
    assert modules_system_emu.is_module_loaded('m3')
    assert modules_system_emu.is_module_loaded('m4')
    assert ['m3', 'm4'] == modules_system_emu.backend.load_seq


def test_mapping_cycle_simple(modules_system_emu):
    #
    # m0 -> m1 -> m0
    #
    modules_system_emu.module_map = {
        'm0': ['m1'],
        'm1': ['m0'],
    }
    with pytest.raises(EnvironError):
        modules_system_emu.load_module('m0')

    with pytest.raises(EnvironError):
        modules_system_emu.load_module('m1')


def test_mapping_single_module_self_loop(modules_system_emu):
    #
    # m0 -> m0
    #
    modules_system_emu.module_map = {
        'm0': ['m0'],
    }
    modules_system_emu.load_module('m0')
    assert modules_system_emu.is_module_loaded('m0')
    assert ['m0'] == modules_system_emu.backend.load_seq


def test_mapping_deep_cycle(modules_system_emu):
    #
    #    -- > m1 ---- > m3
    #   /     ^         |
    # m0      |         |
    #   \     |         .
    #    -- > m2 < ---- m4
    #
    modules_system_emu.module_map = {
        'm0': ['m1', 'm2'],
        'm1': ['m3'],
        'm2': ['m1'],
        'm3': ['m4'],
        'm4': ['m2']
    }
    with pytest.raises(EnvironError, match='m0->m1->m3->m4->m2->m1'):
        modules_system_emu.load_module('m0')


@pytest.fixture
def mapping_file(tmp_path):
    tmp_mapping = tmp_path / 'mapping'
    with open(tmp_mapping, 'w') as fp:
        yield fp


def test_mapping_from_file_simple(modules_system_emu, mapping_file):
    with mapping_file:
        mapping_file.write('m1:m2  m3   m3\n'
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
    modules_system_emu.load_mapping_from_file(mapping_file.name)
    assert reference_map == modules_system_emu.module_map


def test_mapping_from_file_missing_key_separator(modules_system_emu,
                                                 mapping_file):
    with mapping_file:
        mapping_file.write('m1 m2')

    with pytest.raises(ConfigError):
        modules_system_emu.load_mapping_from_file(mapping_file.name)


def test_mapping_from_file_empty_value(modules_system_emu, mapping_file):
    with mapping_file:
        mapping_file.write('m1: # m2')

    with pytest.raises(ConfigError):
        modules_system_emu.load_mapping_from_file(mapping_file.name)


def test_mapping_from_file_multiple_key_separators(modules_system_emu,
                                                   mapping_file):
    with mapping_file:
        mapping_file.write('m1 : m2 : m3')

    with pytest.raises(ConfigError):
        modules_system_emu.load_mapping_from_file(mapping_file.name)


def test_mapping_from_file_empty_key(modules_system_emu, mapping_file):
    with mapping_file:
        mapping_file.write(' :  m2')

    with pytest.raises(ConfigError):
        modules_system_emu.load_mapping_from_file(mapping_file.name)


def test_mapping_from_file_missing_file(modules_system_emu):
    with pytest.raises(OSError):
        modules_system_emu.load_mapping_from_file('foo')


def test_mapping_with_self_loop(modules_system_emu):
    modules_system_emu.module_map = {
        'm0': ['m1', 'm0', 'm2'],
        'm1': ['m4', 'm3']
    }
    modules_system_emu.load_module('m0')
    assert modules_system_emu.is_module_loaded('m0')
    assert modules_system_emu.is_module_loaded('m1')
    assert modules_system_emu.is_module_loaded('m2')
    assert modules_system_emu.is_module_loaded('m3')
    assert modules_system_emu.is_module_loaded('m4')
    assert ['m4', 'm3', 'm0', 'm2'] == modules_system_emu.backend.load_seq


def test_mapping_with_self_loop_and_duplicate_modules(modules_system_emu):
    modules_system_emu.module_map = {
        'm0': ['m0', 'm0', 'm1', 'm1'],
        'm1': ['m2', 'm3']
    }
    modules_system_emu.load_module('m0')
    assert modules_system_emu.is_module_loaded('m0')
    assert modules_system_emu.is_module_loaded('m1')
    assert modules_system_emu.is_module_loaded('m2')
    assert modules_system_emu.is_module_loaded('m3')
    assert ['m0', 'm2', 'm3'] == modules_system_emu.backend.load_seq
