# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Utilities for manipulating the modules subsystem
#

import abc
import os
import re
from collections import OrderedDict

import reframe.core.fields as fields
import reframe.utility.os_ext as os_ext
import reframe.utility.typecheck as types
from reframe.core.exceptions import (ConfigError, EnvironError,
                                     SpawnedProcessError)
from reframe.utility import OrderedSet


class Module:
    '''Module wrapper.

    This class represents internally a module. Concrete module system
    implementation should deal only with that.
    '''

    def __init__(self, name):
        if not isinstance(name, str):
            raise TypeError('module name not a string')

        name = name.strip()
        if not name:
            raise ValueError('module name cannot be empty')

        try:
            self._name, self._version = name.split('/', maxsplit=1)
        except ValueError:
            self._name, self._version = name, None

    @property
    def name(self):
        return self._name

    @property
    def version(self):
        return self._version

    @property
    def fullname(self):
        if self.version is not None:
            return '/'.join((self.name, self.version))
        else:
            return self.name

    def __hash__(self):
        # Here we hash only over the name of the module, because foo/1.2 and
        # simply foo compare equal. In case of hash conflicts (e.g., foo/1.2
        # and foo/1.3), the equality operator will resolve it.
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        if not self.version or not other.version:
            return self.name == other.name
        else:
            return self.name == other.name and self.version == other.version

    def __repr__(self):
        return '%s(%s)' % (type(self).__name__, self.fullname)

    def __str__(self):
        return self.fullname


class ModulesSystem:
    '''A modules system abstraction inside ReFrame.

    This class interfaces between the framework internals and the actual
    modules systems implementation.
    '''

    module_map = fields.TypedField('module_map',
                                   types.Dict[str, types.List[str]])

    @classmethod
    def create(cls, modules_kind=None):
        if modules_kind is None:
            return ModulesSystem(NoModImpl())
        elif modules_kind == 'tmod31':
            return ModulesSystem(TMod31Impl())
        elif modules_kind == 'tmod':
            return ModulesSystem(TModImpl())
        elif modules_kind == 'tmod32':
            return ModulesSystem(TModImpl())
        elif modules_kind == 'tmod4':
            return ModulesSystem(TMod4Impl())
        elif modules_kind == 'lmod':
            return ModulesSystem(LModImpl())
        else:
            raise ConfigError('unknown module system: %s' % modules_kind)

    def __init__(self, backend):
        self._backend = backend
        self.module_map = {}

    def resolve_module(self, name):
        '''Resolve module ``name`` in the registered module map.

        :returns: the list of real modules names pointed to by ``name``.
        :raises: :class:`reframe.core.exceptions.ConfigError` if the mapping
            contains a cycle.
        '''
        ret = OrderedSet()
        visited = set()
        unvisited = [(name, None)]
        path = []
        while unvisited:
            node, parent = unvisited.pop()
            # Adjust the path
            while path and path[-1] != parent:
                path.pop()

            # Handle modules mappings with self loops
            if node == parent:
                ret.add(node)
                continue

            try:
                # We insert the adjacent nodes in reverse order, so as to
                # preserve the DFS access order
                adjacent = reversed(self.module_map[node])
            except KeyError:
                # We have reached a terminal node
                ret.add(node)
            else:
                path.append(node)
                for m in adjacent:
                    if m in path and m != node:
                        raise EnvironError('module cyclic dependency: ' +
                                           '->'.join(path + [m]))
                    if m not in visited:
                        unvisited.append((m, node))

            visited.add(node)

        return list(ret)

    @property
    def backend(self):
        return(self._backend)

    def loaded_modules(self):
        '''Return a list of loaded modules.

        This method returns a list of strings.
        '''
        return [str(m) for m in self._backend.loaded_modules()]

    def conflicted_modules(self, name):
        '''Return the list of the modules conflicting with module ``name``.

        If module ``name`` resolves to multiple real modules, then the returned
        list will be the concatenation of the conflict lists of all the real
        modules.

        This method returns a list of strings.
        '''
        ret = []
        for m in self.resolve_module(name):
            ret += self._conflicted_modules(m)

        return ret

    def _conflicted_modules(self, name):
        return [str(m) for m in self._backend.conflicted_modules(Module(name))]

    def load_module(self, name, force=False):
        '''Load the module ``name``.

        If ``force`` is set, forces the loading, unloading first any
        conflicting modules currently loaded. If module ``name`` refers to
        multiple real modules, all of the target modules will be loaded.

        Returns the list of unloaded modules as strings.
        '''
        ret = []
        for m in self.resolve_module(name):
            ret += self._load_module(m, force)

        return ret

    def _load_module(self, name, force=False):
        module = Module(name)
        loaded_modules = self._backend.loaded_modules()
        if module in loaded_modules:
            # Do not try to load the module if it is already present
            return []

        # Get the list of the modules that need to be unloaded
        unload_list = set()
        if force:
            conflict_list = self._backend.conflicted_modules(module)
            unload_list = set(loaded_modules) & set(conflict_list)

        for m in unload_list:
            self._backend.unload_module(m)

        self._backend.load_module(module)
        return [str(m) for m in unload_list]

    def unload_module(self, name):
        '''Unload module ``name``.

        If module ``name`` refers to multiple real modules, all the referred to
        modules will be unloaded in reverse order.
        '''
        for m in reversed(self.resolve_module(name)):
            self._unload_module(m)

    def _unload_module(self, name):
        self._backend.unload_module(Module(name))

    def is_module_loaded(self, name):
        '''Check if module ``name`` is loaded.

        If module ``name`` refers to multiple real modules, this method will
        return :class:`True` only if all the referees are loaded.
        '''
        return all(self._is_module_loaded(m)
                   for m in self.resolve_module(name))

    def _is_module_loaded(self, name):
        return self._backend.is_module_loaded(Module(name))

    def load_mapping(self, mapping):
        '''Update the internal module mappings using a single mapping.

        :arg mapping: a string specifying the module mapping.
            Example syntax: ``'m0: m1 m2'``.

        '''
        key, *rest = mapping.split(':')
        if len(rest) != 1:
            raise ConfigError('invalid mapping syntax: %s' % mapping)

        key = key.strip()
        values = rest[0].split()
        if not key:
            raise ConfigError('no key found in mapping: %s' % mapping)

        if not values:
            raise ConfigError('no mapping defined for module: %s' % key)

        self.module_map[key] = list(OrderedDict.fromkeys(values))

    def load_mapping_from_file(self, filename):
        '''Update the internal module mappings from mappings read from file.'''
        with open(filename) as fp:
            for lineno, line in enumerate(fp, start=1):
                line = line.strip().split('#')[0]
                if not line:
                    continue

                try:
                    self.load_mapping(line)
                except ConfigError as e:
                    raise ConfigError('%s:%s' % (filename, lineno)) from e

    @property
    def name(self):
        '''Return the name of this module system.'''
        return self._backend.name()

    @property
    def version(self):
        '''Return the version of this module system.'''
        return self._backend.version()

    def unload_all(self):
        '''Unload all loaded modules.'''
        return self._backend.unload_all()

    @property
    def searchpath(self):
        '''The module system search path as a list of directories.'''
        return self._backend.searchpath()

    def searchpath_add(self, *dirs):
        '''Add ``dirs`` to the module system search path.'''
        return self._backend.searchpath_add(*dirs)

    def searchpath_remove(self, *dirs):
        '''Remove ``dirs`` from the module system search path.'''
        return self._backend.searchpath_remove(*dirs)

    def emit_load_commands(self, name):
        '''Return the appropriate shell command for loading module ``name``.'''
        return [self._backend.emit_load_instr(Module(name))
                for name in self.resolve_module(name)]

    def emit_unload_commands(self, name):
        '''Return the appropriate shell command for unloading module
        ``name``.'''
        return [self._backend.emit_unload_instr(Module(name))
                for name in reversed(self.resolve_module(name))]

    def __str__(self):
        return str(self._backend)


class ModulesSystemImpl(abc.ABC):
    '''Abstract base class for module systems.'''

    @abc.abstractmethod
    def loaded_modules(self):
        '''Return a list of loaded modules.

        This method returns a list of Module instances.
        '''

    @abc.abstractmethod
    def conflicted_modules(self, module):
        '''Return the list of conflicted modules.

        This method returns a list of Module instances.
        '''

    @abc.abstractmethod
    def load_module(self, module):
        '''Load the module ``name``.

        If ``force`` is set, forces the loading,
        unloading first any conflicting modules currently loaded.

        Returns the unloaded modules as a list of module instances.'''

    @abc.abstractmethod
    def unload_module(self, module):
        '''Unload module ``module``.'''

    @abc.abstractmethod
    def is_module_loaded(self, module):
        '''Check presence of module ``module``.'''

    @abc.abstractmethod
    def name(self):
        '''Return the name of this module system.'''

    @abc.abstractmethod
    def version(self):
        '''Return the version of this module system.'''

    @abc.abstractmethod
    def unload_all(self):
        '''Unload all loaded modules.'''

    @abc.abstractmethod
    def searchpath(self):
        '''The module system search path as a list of directories.'''

    @abc.abstractmethod
    def searchpath_add(self, *dirs):
        '''Add ``dirs`` to the module system search path.'''

    @abc.abstractmethod
    def searchpath_remove(self, *dirs):
        '''Remove ``dirs`` from the module system search path.'''

    @abc.abstractmethod
    def emit_load_instr(self, module):
        '''Emit the instruction that loads module.'''

    @abc.abstractmethod
    def emit_unload_instr(self, module):
        '''Emit the instruction that unloads module.'''

    def __repr__(self):
        return type(self).__name__ + '()'

    def __str__(self):
        return self.name() + ' ' + self.version()


class TModImpl(ModulesSystemImpl):
    '''Base class for TMod Module system (Tcl).'''

    MIN_VERSION = (3, 2)

    def __init__(self):
        # Try to figure out if we are indeed using the TCL version
        try:
            completed = os_ext.run_command('modulecmd -V')
        except OSError as e:
            raise ConfigError(
                'could not find a sane TMod installation: %s' % e) from e

        version_match = re.search(r'^VERSION=(\S+)', completed.stdout,
                                  re.MULTILINE)
        tcl_version_match = re.search(r'^TCL_VERSION=(\S+)', completed.stdout,
                                      re.MULTILINE)

        if version_match is None or tcl_version_match is None:
            raise ConfigError('could not find a sane TMod installation')

        version = version_match.group(1)
        try:
            ver_major, ver_minor, *_ = [int(v) for v in version.split('.')]
        except ValueError:
            raise ConfigError(
                'could not parse TMod version string: ' + version) from None

        if (ver_major, ver_minor) < self.MIN_VERSION:
            raise ConfigError(
                'unsupported TMod version: %s (required >= %s)' %
                (version, self.MIN_VERSION))

        self._version = version
        self._command = 'modulecmd python'
        try:
            # Try the Python bindings now
            completed = os_ext.run_command(self._command)
        except OSError as e:
            raise ConfigError(
                'could not get the Python bindings for TMod: ' % e) from e

        if re.search(r'Unknown shell type', completed.stderr):
            raise ConfigError(
                'Python is not supported by this TMod installation')

    def name(self):
        return 'tmod'

    def version(self):
        return self._version

    def _run_module_command(self, *args, msg=None):
        command = ' '.join([self._command, *args])
        try:
            completed = os_ext.run_command(command, check=True)
        except SpawnedProcessError as e:
            raise EnvironError(msg) from e

        if self._module_command_failed(completed):
            err = SpawnedProcessError(command,
                                      completed.stdout,
                                      completed.stderr,
                                      completed.returncode)
            raise EnvironError(msg) from err

        return completed

    def _module_command_failed(self, completed):
        return re.search(r'ERROR', completed.stderr) is not None

    def _exec_module_command(self, *args, msg=None):
        completed = self._run_module_command(*args, msg=msg)
        exec(completed.stdout)

    def loaded_modules(self):
        try:
            # LOADEDMODULES may be defined but empty
            return [Module(m)
                    for m in os.environ['LOADEDMODULES'].split(':') if m]
        except KeyError:
            return []

    def conflicted_modules(self, module):
        completed = self._run_module_command(
            'show', str(module), msg="could not show module '%s'" % module)
        return [Module(m.group(1))
                for m in re.finditer(r'^conflict\s+(\S+)',
                                     completed.stderr, re.MULTILINE)]

    def is_module_loaded(self, module):
        return module in self.loaded_modules()

    def load_module(self, module):
        self._exec_module_command(
            'load', str(module),
            msg="could not load module '%s' correctly" % module)

    def unload_module(self, module):
        self._exec_module_command(
            'unload', str(module),
            msg="could not unload module '%s' correctly" % module)

    def unload_all(self):
        self._exec_module_command('purge')

    def searchpath(self):
        return os.environ['MODULEPATH'].split(':')

    def searchpath_add(self, *dirs):
        self._exec_module_command('use', *dirs)

    def searchpath_remove(self, *dirs):
        self._exec_module_command('unuse', *dirs)

    def emit_load_instr(self, module):
        return 'module load %s' % module

    def emit_unload_instr(self, module):
        return 'module unload %s' % module


class TMod31Impl(TModImpl):
    '''Module system for TMod (Tcl).'''

    MIN_VERSION = (3, 1)

    def __init__(self):
        # Try to figure out if we are indeed using the TCL version
        try:
            modulecmd = os.getenv('MODULESHOME')
            modulecmd = os.path.join(modulecmd, 'modulecmd.tcl')
            completed = os_ext.run_command(modulecmd)
        except OSError as e:
            raise ConfigError(
                'could not find a sane TMod31 installation: %s' % e) from e

        version_match = re.search(r'Release Tcl (\S+)', completed.stderr,
                                  re.MULTILINE)
        tcl_version_match = version_match

        if version_match is None or tcl_version_match is None:
            raise ConfigError('could not find a sane TMod31 installation')

        version = version_match.group(1)
        try:
            ver_major, ver_minor, *_ = [int(v) for v in version.split('.')]
        except ValueError:
            raise ConfigError(
                'could not parse TMod31 version string: ' + version) from None

        if (ver_major, ver_minor) < self.MIN_VERSION:
            raise ConfigError(
                'unsupported TMod version: %s (required >= %s)' %
                (version, self.MIN_VERSION))

        self._version = version
        self._command = '%s python' % modulecmd

        try:
            # Try the Python bindings now
            completed = os_ext.run_command(self._command)
        except OSError as e:
            raise ConfigError(
                'could not get the Python bindings for TMod31: ' % e) from e

        if re.search(r'Unknown shell type', completed.stderr):
            raise ConfigError(
                'Python is not supported by this TMod installation')

    def name(self):
        return 'tmod31'

    def _exec_module_command(self, *args, msg=None):
        completed = self._run_module_command(*args, msg=msg)
        exec_match = re.search(r'^exec\s\'', completed.stdout)
        if exec_match is None:
            raise ConfigError('could not use the python bindings')
        else:
            cmd = completed.stdout
            exec_match = re.search(r'^exec\s\'(\S+)\'', cmd,
                                   re.MULTILINE)
            if exec_match is None:
                raise ConfigError('could not use the python bindings')
            with open(exec_match.group(1), 'r') as content_file:
                cmd = content_file.read()

        exec(cmd)


class TMod4Impl(TModImpl):
    '''Module system for TMod 4.'''

    MIN_VERSION = (4, 1)

    def __init__(self):
        self._command = 'modulecmd python'
        try:
            completed = os_ext.run_command(self._command + ' -V', check=True)
        except OSError as e:
            raise ConfigError(
                'could not find a sane TMod4 installation') from e
        except SpawnedProcessError as e:
            raise ConfigError(
                'could not get the Python bindings for TMod4') from e

        version_match = re.match(r'^Modules Release (\S+)\s+',
                                 completed.stderr)
        if not version_match:
            raise ConfigError('could not retrieve the TMod4 version')

        version = version_match.group(1)
        try:
            ver_major, ver_minor, *_ = [int(v) for v in version.split('.')]
        except ValueError:
            raise ConfigError(
                'could not parse TMod4 version string: ' + version) from None

        if (ver_major, ver_minor) < self.MIN_VERSION:
            raise ConfigError(
                'unsupported TMod4 version: %s (required >= %s)' %
                (version, self.MIN_VERSION))

        self._version = version

    def name(self):
        return 'tmod4'

    def _exec_module_command(self, *args, msg=None):
        command = ' '.join([self._command, *args])
        completed = os_ext.run_command(command, check=True)
        namespace = {}
        exec(completed.stdout, {}, namespace)
        if not namespace['_mlstatus']:
            # _mlstatus is set by the TMod4 Python bindings
            if msg is None:
                msg = 'modules system command failed: '
                if isinstance(completed.args, str):
                    msg += completed.args
                else:
                    msg += ' '.join(completed.args)

            raise EnvironError(msg)


class LModImpl(TModImpl):
    '''Module system for Lmod (Tcl/Lua).'''

    def __init__(self):
        # Try to figure out if we are indeed using LMOD
        lmod_cmd = os.getenv('LMOD_CMD')
        if lmod_cmd is None:
            raise ConfigError('could not find a sane Lmod installation: '
                              'environment variable LMOD_CMD is not defined')

        try:
            completed = os_ext.run_command('%s --version' % lmod_cmd)
        except OSError as e:
            raise ConfigError(
                'could not find a sane Lmod installation: %s' % e)

        version_match = re.search(r'.*Version\s*(\S+)', completed.stderr,
                                  re.MULTILINE)
        if version_match is None:
            raise ConfigError('could not retrieve Lmod version')

        self._version = version_match.group(1)
        self._command = '%s python ' % lmod_cmd
        try:
            # Try the Python bindings now
            completed = os_ext.run_command(self._command)
        except OSError as e:
            raise ConfigError(
                'could not get the Python bindings for Lmod: ' % e)

        if re.search(r'Unknown shell type', completed.stderr):
            raise ConfigError('Python is not supported by '
                              'this Lmod installation')

    def name(self):
        return 'lmod'

    def _module_command_failed(self, completed):
        return completed.stdout.strip() == 'false'

    def conflicted_modules(self, module):
        completed = self._run_module_command(
            'show', str(module), msg="could not show module '%s'" % module)

        # Lmod accepts both Lua and and Tcl syntax
        # The following test allows incorrect syntax, e.g., `conflict
        # ('package"(`, but we expect this to be caught by the Lmod framework
        # in earlier stages.
        ret = []
        for m in re.finditer(r'conflict\s*(\S+)', completed.stderr):
            conflict_arg = m.group(1)
            if conflict_arg.startswith('('):
                # Lua syntax
                ret.append(Module(conflict_arg.strip('\'"()')))
            else:
                # Tmod syntax
                ret.append(Module(conflict_arg))

        return ret

    def unload_all(self):
        # Currently, we don't take any provision for sticky modules in Lmod, so
        # we forcefully unload everything.
        self._exec_module_command('--force', 'purge')


class NoModImpl(ModulesSystemImpl):
    '''A convenience class that implements a no-op a modules system.'''

    def loaded_modules(self):
        return []

    def conflicted_modules(self, module):
        return []

    def load_module(self, module):
        pass

    def unload_module(self, module):
        pass

    def is_module_loaded(self, module):
        #
        # Always return `True`, since this pseudo modules system effectively
        # assumes that everything needed is loaded.
        #
        return True

    def name(self):
        return 'nomod'

    def version(self):
        return '1.0'

    def unload_all(self):
        pass

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
