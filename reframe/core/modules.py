# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
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
import reframe.utility.osext as osext
import reframe.utility.typecheck as types
from reframe.core.exceptions import (ConfigError, EnvironError,
                                     SpawnedProcessError)
from reframe.core.logging import getlogger
from reframe.utility import OrderedSet


class Module:
    '''Module wrapper.

    This class represents internally a module. Concrete module system
    implementation should deal only with that.

    :meta private:
    '''

    def __init__(self, name, collection=False, path=None):
        if not isinstance(name, str):
            raise TypeError('module name not a string')

        name = name.strip()
        if not name:
            raise ValueError('module name cannot be empty')

        try:
            self._name, self._version = name.split('/', maxsplit=1)
        except ValueError:
            self._name, self._version = name, None

        self._path = path

        # This module represents a "module collection" in TMod4
        self._collection = collection

    @property
    def name(self):
        return self._name

    @property
    def version(self):
        return self._version

    @property
    def collection(self):
        return self._collection

    @property
    def path(self):
        return self._path

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

        if self.path != other.path:
            return False

        if self.collection != other.collection:
            return False

        if not self.version or not other.version:
            return self.name == other.name
        else:
            return self.name == other.name and self.version == other.version

    def __repr__(self):
        return (f'{type(self).__name__}({self.fullname}, '
                '{self.collection}, {self.path})')

    def __str__(self):
        return self.fullname


class ModulesSystem:
    '''A modules system.'''

    module_map = fields.TypedField(types.Dict[str, types.List[str]])

    @classmethod
    def create(cls, modules_kind=None):
        getlogger().debug(f'Initializing modules system {modules_kind!r}')
        if modules_kind is None or modules_kind == 'nomod':
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
        elif modules_kind == 'spack':
            return ModulesSystem(SpackImpl())
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

        :meta private:
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

    def available_modules(self, substr=None):
        '''Return a list of available modules that contain ``substr`` in their
        name.

        :rtype: List[str]
        '''
        return [str(m) for m in self._backend.available_modules(substr or '')]

    def loaded_modules(self):
        '''Return a list of loaded modules.

        :rtype: List[str]
        '''
        return [str(m) for m in self._backend.loaded_modules()]

    def conflicted_modules(self, name, collection=False, path=None):
        '''Return the list of the modules conflicting with module ``name``.

        If module ``name`` resolves to multiple real modules, then the returned
        list will be the concatenation of the conflict lists of all the real
        modules.

        :arg name: The name of the module.
        :arg collection: The module is a "module collection" (TMod4/LMod only).
        :arg path: The path where the module resides if not in the default
            ``MODULEPATH``.
        :returns: A list of conflicting module names.

        .. versionchanged:: 3.3
           The ``collection`` argument is added.

        .. versionchanged:: 3.5.0
           The ``path`` argument is added.

        '''
        ret = []
        for m in self.resolve_module(name):
            ret += self._conflicted_modules(m, collection, path)

        return ret

    def _conflicted_modules(self, name, collection=False, path=None):
        return [
            str(m)
            for m in self._backend.conflicted_modules(
                Module(name, collection, path)
            )
        ]

    def execute(self, cmd, *args):
        '''Execute an arbitrary module command.

        :arg cmd: The command to execute, e.g., ``load``, ``restore`` etc.
        :arg args: The arguments to pass to the command.
        :returns: The command output.
        '''
        return self._backend.execute(cmd, *args)

    def load_module(self, name, collection=False, path=None, force=False):
        '''Load the module ``name``.

        :arg collection: The module is a "module collection" (TMod4/Lmod only)
        :arg path: The path where the module resides if not in the default
            ``MODULEPATH``.
        :arg force: If set, forces the loading, unloading first any
            conflicting modules currently loaded. If module ``name`` refers to
            multiple real modules, all of the target modules will be loaded.
        :returns: A list of two-element tuples, where each tuple contains the
            module that was loaded and the list of modules that had to be
            unloaded first due to conflicts. This list will be normally of
            size one, but it can be longer if there is mapping that maps
            module ``name`` to multiple other modules.

        .. versionchanged:: 3.3
           - The ``collection`` argument is added.
           - This function now returns a list of tuples.

        .. versionchanged:: 3.5.0
           - The ``path`` argument is added.
           - The ``force`` argument is now the last argument.

        '''
        ret = []
        for m in self.resolve_module(name):
            ret.append((m, self._load_module(m, collection, path, force)))

        return ret

    def _load_module(self, name, collection=False, path=None, force=False):
        module = Module(name, collection, path)
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

    def unload_module(self, name, collection=False, path=None):
        '''Unload module ``name``.

        :arg name: The name of the module to unload. If module ``name`` is
            resolved to multiple real modules, all the referred to modules
            will be unloaded in reverse order.
        :arg collection: The module is a "module collection" (TMod4 only)
        :arg path: The path where the module resides if not in the default
            ``MODULEPATH``.

        .. versionchanged:: 3.3
           The ``collection`` argument was added.

        .. versionchanged:: 3.5.0
           The ``path`` argument is added.

        '''
        for m in reversed(self.resolve_module(name)):
            self._unload_module(m, collection, path)

    def _unload_module(self, name, collection=False, path=None):
        self._backend.unload_module(Module(name, collection, path))

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

        :meta private:
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
        '''Update the internal module mappings from mappings read from file.

        :meta private:
        '''
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
        '''The name of this module system.'''
        return self._backend.name()

    @property
    def version(self):
        '''The version of this module system.'''
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

    def change_module_path(self, *dirs):
        return self._backend.change_module_path(*dirs)

    def emit_load_commands(self, name, collection=False, path=None):
        '''Return the appropriate shell commands for loading a module.

        Module mappings are not taken into account by this function.

        :arg name: The name of the module to load.
        :arg collection: The module is a "module collection" (TMod4/LMod only)
        :arg path: The path where the module resides if not in the default
            ``MODULEPATH``.
        :returns: A list of shell commands.

        .. versionchanged:: 3.3
           The ``collection`` argument was added and module mappings are no
           more taken into account by this function.

        .. versionchanged:: 3.5.0
           The ``path`` argument is added.

        '''

        # We don't consider module mappings here, because we cannot treat
        # correctly possible conflicts
        return self._backend.emit_load_instr(Module(name, collection, path))

    def emit_unload_commands(self, name, collection=False, path=None):
        '''Return the appropriate shell commands for unloading a module.

        Module mappings are not taken into account by this function.

        :arg name: The name of the module to unload.
        :arg collection: The module is a "module collection" (TMod4/LMod only)
        :arg path: The path where the module resides if not in the default
            ``MODULEPATH``.
        :returns: A list of shell commands.

        .. versionchanged:: 3.3
           The ``collection`` argument was added and module mappings are no
           more taken into account by this function.

        .. versionchanged:: 3.5.0
           The ``path`` argument is added.

        '''

        # See comment in emit_load_commands()
        return self._backend.emit_unload_instr(Module(name, collection, path))

    def __str__(self):
        return str(self._backend)


class ModulesSystemImpl(abc.ABC):
    '''Abstract base class for module systems.

    :meta private:
    '''

    def execute(self, cmd, *args):
        '''Execute an arbitrary module command using the modules backend.

        :arg cmd: The command to execute, e.g., ``load``, ``restore`` etc.
        :arg args: The arguments to pass to the command.
        :returns: The command output.
        '''
        try:
            exec_output = self._execute(cmd, *args)
        except SpawnedProcessError as e:
            raise EnvironError('could not execute module operation') from e

        return exec_output

    def execute_with_path(self, cmd, *args, path=None):
        with self.change_module_path(path):
            return self.execute(cmd, *args)

    @abc.abstractmethod
    def _execute(self, cmd, *args):
        '''Execute an arbitrary command of the module system.'''

    @abc.abstractmethod
    def available_modules(self, substr):
        '''Return a list of available modules, whose name contains ``substr``.

        This method returns a list of Module instances.
        '''

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
        '''Load module ``module``.'''

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
    def modulecmd(self, *args):
        '''The low level command to use for issuing module commads'''

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

    def process(self, source):
        '''Process the Python source emitted by the Python bindings of the
        different backends.

        Backends should call this before executing any Python commands.

        :arg source: The Python source code to be executed.
        :returns: The modified Python source code to be executed. By default
            ``source`` is returned unchanged.

        .. versionadded:: 3.4

        '''
        return source

    def change_module_path(self, *dirs):
        '''Temporarily change the module path.

        :arg dirs: The directories to add to the module path.
        :returns: a context manager that handles the temporary module path
            change.

        .. versionadded:: 3.5.0
        '''

        class _CtxMgr:
            def __init__(this):
                # Filter out empty paths
                this._paths = [d for d in dirs if d]

            def __enter__(this):
                self.searchpath_add(*this._paths)

            def __exit__(this, exc_type, exc_value, traceback):
                self.searchpath_remove(*this._paths)

        return _CtxMgr()

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
            completed = osext.run_command('modulecmd -V')
        except OSError as e:
            raise ConfigError(
                'could not find a sane TMod installation') from e

        version_match = re.search(r'^VERSION=(\S+)', completed.stdout,
                                  re.MULTILINE)
        tcl_version_match = re.search(r'^TCL_VERSION=(\S+)', completed.stdout,
                                      re.MULTILINE)

        if version_match is None or tcl_version_match is None:
            raise ConfigError('could not find a sane TMod installation')

        version = version_match.group(1)
        try:
            ver_major, ver_minor = [int(v) for v in version.split('.')[:2]]
        except ValueError:
            raise ConfigError(
                'could not parse TMod version string: ' + version) from None

        if (ver_major, ver_minor) < self.MIN_VERSION:
            raise ConfigError(
                'unsupported TMod version: %s (required >= %s)' %
                (version, self.MIN_VERSION))

        self._version = version
        try:
            # Try the Python bindings now
            completed = osext.run_command(self.modulecmd())
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

    def modulecmd(self, *args):
        return ' '.join(['modulecmd', 'python', *args])

    def _execute(self, cmd, *args):
        modulecmd = self.modulecmd(cmd, *args)
        completed = osext.run_command(modulecmd)
        if re.search(r'\bERROR\b', completed.stderr) is not None:
            raise SpawnedProcessError(modulecmd,
                                      completed.stdout,
                                      completed.stderr,
                                      completed.returncode)

        exec(self.process(completed.stdout))
        return completed.stderr

    def available_modules(self, substr):
        output = self.execute('avail', '-t', substr)
        ret = []
        for line in output.split('\n'):
            if not line or line[-1] == ':':
                # Ignore empty lines and path entries
                continue

            module = re.sub(r'\(default\)', '', line)
            ret.append(Module(module))

        return ret

    def loaded_modules(self):
        try:
            # LOADEDMODULES may be defined but empty
            return [Module(m)
                    for m in os.environ['LOADEDMODULES'].split(':') if m]
        except KeyError:
            return []

    def conflicted_modules(self, module):
        output = self.execute_with_path('show', str(module), path=module.path)
        return [Module(m.group(1))
                for m in re.finditer(r'^conflict\s+(\S+)',
                                     output, re.MULTILINE)]

    def is_module_loaded(self, module):
        return module in self.loaded_modules()

    def load_module(self, module):
        self.execute_with_path('load', str(module), path=module.path)

    def unload_module(self, module):
        self.execute('unload', str(module))

    def unload_all(self):
        self.execute('purge')

    def searchpath(self):
        path = os.getenv('MODULEPATH', '')
        return path.split(':')

    def searchpath_add(self, *dirs):
        if dirs:
            self.execute('use', *dirs)

    def searchpath_remove(self, *dirs):
        if dirs:
            self.execute('unuse', *dirs)

    def emit_load_instr(self, module):
        commands = []
        if module.path:
            commands.append(f'module use {module.path}')

        commands.append(f'module load {module.fullname}')
        if module.path:
            commands.append(f'module unuse {module.path}')

        return commands

    def emit_unload_instr(self, module):
        return [f'module unload {module}']


class TMod31Impl(TModImpl):
    '''Module system for TMod (Tcl).'''

    MIN_VERSION = (3, 1)

    def __init__(self):
        # Try to figure out if we are indeed using the TCL version
        try:
            modulecmd = os.getenv('MODULESHOME')
            modulecmd = os.path.join(modulecmd, 'modulecmd.tcl')
            completed = osext.run_command(modulecmd)
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
            ver_major, ver_minor = [int(v) for v in version.split('.')[:2]]
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
            completed = osext.run_command(self._command)
        except OSError as e:
            raise ConfigError(
                'could not get the Python bindings for TMod31: ' % e) from e

        if re.search(r'Unknown shell type', completed.stderr):
            raise ConfigError(
                'Python is not supported by this TMod installation')

    def name(self):
        return 'tmod31'

    def modulecmd(self, *args):
        return ' '.join([self._command, *args])

    def _execute(self, cmd, *args):
        modulecmd = self.modulecmd(cmd, *args)
        completed = osext.run_command(modulecmd)
        if re.search(r'\bERROR\b', completed.stderr) is not None:
            raise SpawnedProcessError(modulecmd,
                                      completed.stdout,
                                      completed.stderr,
                                      completed.returncode)

        exec_match = re.search(r"^exec\s'(\S+)'", completed.stdout,
                               re.MULTILINE)
        if exec_match is None:
            raise ConfigError('could not use the python bindings')

        with open(exec_match.group(1), 'r') as content_file:
            cmd = content_file.read()

        exec(self.process(cmd))
        return completed.stderr


class TMod4Impl(TModImpl):
    '''Module system for TMod 4.'''

    MIN_VERSION = (4, 1)

    def __init__(self):
        try:
            completed = osext.run_command(self.modulecmd('-V'), check=True)
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
            ver_major, ver_minor = [int(v) for v in version.split('.')[:2]]
        except ValueError:
            raise ConfigError(
                'could not parse TMod4 version string: ' + version) from None

        if (ver_major, ver_minor) < self.MIN_VERSION:
            raise ConfigError(
                'unsupported TMod4 version: %s (required >= %s)' %
                (version, self.MIN_VERSION))

        self._version = version
        self._extra_module_paths = []

    def name(self):
        return 'tmod4'

    def modulecmd(self, *args):
        return ' '.join(['modulecmd', 'python', *args])

    def _execute(self, cmd, *args):
        modulecmd = self.modulecmd(cmd, *args)
        completed = osext.run_command(modulecmd, check=False)
        namespace = {}
        exec(self.process(completed.stdout), {}, namespace)

        # _mlstatus is set by the TMod4 only if the command was unsuccessful,
        # but Lmod sets it always
        if not namespace.get('_mlstatus', True):
            raise SpawnedProcessError(modulecmd,
                                      completed.stdout,
                                      completed.stderr,
                                      completed.returncode)

        return completed.stderr

    def load_module(self, module):
        if module.collection:
            self.execute('restore', str(module))

            # Here the module search path removal/addition is repeated since
            # 'restore' discards previous module path manipulations
            for op, mp in self._extra_module_paths:
                if op == '+':
                    super().searchpath_add(mp)
                else:
                    super().searchpath_remove(mp)

            return []
        else:
            return super().load_module(module)

    def unload_module(self, module):
        if module.collection:
            # Module collection are not unloaded
            return

        super().unload_module(module)

    def conflicted_modules(self, module):
        if module.collection:
            # Conflicts have no meaning in module collection. The modules
            # system will take care of these when restoring a module
            # collection
            return []

        return super().conflicted_modules(module)

    def _emit_restore_instr(self, module):
        cmds = [f'module restore {module}']

        # Here we append module searchpath removal/addition commands
        # since 'restore' discards previous module path manipulations
        for op, mp in self._extra_module_paths:
            operation = 'use' if op == '+' else 'unuse'
            cmds += [f'module {operation} {mp}']

        return cmds

    def emit_load_instr(self, module):
        if module.collection:
            return self._emit_restore_instr(module)

        return super().emit_load_instr(module)

    def emit_unload_instr(self, module):
        if module.collection:
            return []

        return super().emit_unload_instr(module)

    def searchpath_add(self, *dirs):
        if dirs:
            self._extra_module_paths += [('+', mp) for mp in dirs]

        super().searchpath_add(*dirs)

    def searchpath_remove(self, *dirs):
        if dirs:
            self._extra_module_paths += [('-', mp) for mp in dirs]

        super().searchpath_remove(*dirs)


class LModImpl(TMod4Impl):
    '''Module system for Lmod (Tcl/Lua).'''

    def __init__(self):
        # Try to figure out if we are indeed using LMOD
        self._lmod_cmd = os.getenv('LMOD_CMD')
        if self._lmod_cmd is None:
            raise ConfigError('could not find a sane Lmod installation: '
                              'environment variable LMOD_CMD is not defined')

        try:
            completed = osext.run_command(f'{self._lmod_cmd} --version')
        except OSError as e:
            raise ConfigError(
                'could not find a sane Lmod installation: %s' % e)

        version_match = re.search(r'.*Version\s*(\S+)', completed.stderr,
                                  re.MULTILINE)
        if version_match is None:
            raise ConfigError('could not retrieve Lmod version')

        self._version = version_match.group(1)
        try:
            # Try the Python bindings now
            completed = osext.run_command(self.modulecmd())
        except OSError as e:
            raise ConfigError(
                'could not get the Python bindings for Lmod: ' % e)

        if re.search(r'Unknown shell type', completed.stderr):
            raise ConfigError('Python is not supported by '
                              'this Lmod installation')

        self._extra_module_paths = []

    def name(self):
        return 'lmod'

    def process(self, source):
        major, minor, *_ = self.version().split('.')
        major, minor = int(major), int(minor)
        if (major, minor) < (8, 2):
            # Older Lmod versions do not emit an `import os` and emit an
            # invalid `false` statement in case of errors; we fix these here
            return 'import os\n\n' + source.replace('false',
                                                    '_mlstatus = False')

        return source

    def modulecmd(self, *args):
        return ' '.join([self._lmod_cmd, 'python', *args])

    def available_modules(self, substr):
        output = self.execute('-t', 'avail', substr)
        ret = []
        for line in output.split('\n'):
            if not line or line[-1] == ':':
                # Ignore empty lines and path entries
                continue

            module = re.sub(r'\(\S+\)', '', line)
            ret.append(Module(module))

        return ret

    def conflicted_modules(self, module):
        if module.collection:
            # Conflicts have no meaning in module collection. The modules
            # system will take care of these when restoring a module
            # collection
            return []

        output = self.execute_with_path('show', str(module), path=module.path)

        # Lmod accepts both Lua and and Tcl syntax
        # The following test allows incorrect syntax, e.g., `conflict
        # ('package"(`, but we expect this to be caught by the Lmod framework
        # in earlier stages.
        ret = []
        for m in re.finditer(r'conflict\s*(\S+)', output):
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
        self.execute('--force', 'purge')

    def emit_load_instr(self, module):
        if module.collection:
            return self._emit_restore_instr(module)

        cmds = []
        if module.path:
            cmds.append(f'module use {module.path}')

        cmds.append(f'module load {module.fullname}')
        return cmds


class NoModImpl(ModulesSystemImpl):
    '''A convenience class that implements a no-op a modules system.'''

    def _warn(self, msg):
        getlogger().warning(
            f"no modules system is set: {msg}: "
            f"check the 'modules_system' configuration "
            f"parameter for your system",
            cache=True
        )

    def available_modules(self, substr):
        return []

    def loaded_modules(self):
        return []

    def conflicted_modules(self, module):
        return []

    def _execute(self, cmd, *args):
        return ''

    def load_module(self, module):
        self._warn(f'module {module.name!r} will not be loaded')

    def unload_module(self, module):
        self._warn(f'module {module.name!r} will not be unloaded')

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

    def modulecmd(self, *args):
        return ''

    def unload_all(self):
        self._warn('no module will be purged')

    def searchpath(self):
        return []

    def searchpath_add(self, *dirs):
        self._warn('MODULEPATH will not be modified')

    def searchpath_remove(self, *dirs):
        self._warn('MODULEPATH will not be modified')

    def emit_load_instr(self, module):
        self._warn(f'module {module.name!r} will not be loaded')
        return []

    def emit_unload_instr(self, module):
        self._warn(f'module {module.name!r} will not be unloaded')
        return []


class SpackImpl(ModulesSystemImpl):
    '''Backend for Spack's modules system emulation.

    This backend implements :func:`load_module`, :func:`unload_module` as well
    as the searchpath methods as no-ops, since Spack does not offer any Python
    bindings for its emulation.

    '''

    def __init__(self):
        # Try to figure out if we are indeed using the TCL version
        try:
            completed = osext.run_command('spack -V')
        except OSError as e:
            raise ConfigError(
                'could not find a sane Spack installation') from e

        self._version = completed.stdout.strip()
        self._name_format = '{name}/{version}-{hash}'

    def name(self):
        return 'spack'

    def version(self):
        return self._version

    def modulecmd(self, *args):
        return ' '.join(['spack', *args])

    def _execute(self, cmd, *args):
        modulecmd = self.modulecmd(cmd, *args)
        completed = osext.run_command(modulecmd, check=True)
        return completed.stdout

    def available_modules(self, substr):
        output = self.execute('find', '--format', self._name_format,
                              substr)
        ret = []
        for line in output.split('\n'):
            if not line or line[-1] == ':':
                # Ignore empty lines and path entries
                continue

            ret.append(Module(line))

        return ret

    def loaded_modules(self):
        output = self.execute('find', '--loaded', '--format',
                              self._name_format)
        return [Module(m) for m in output.split('\n') if m]

    def conflicted_modules(self, module):
        return []

    def is_module_loaded(self, module):
        module = self.execute('find', '--format', self._name_format, name)
        module = Module(module)
        return module in self.loaded_modules()

    def load_module(self, module):
        pass

    def unload_module(self, module):
        pass

    def unload_all(self):
        pass

    def searchpath(self):
        return []

    def searchpath_add(self, *dirs):
        pass

    def searchpath_remove(self, *dirs):
        pass

    def emit_load_instr(self, module):
        return [f'spack load {module.fullname}']

    def emit_unload_instr(self, module):
        return [f'spack unload {module.fullname}']
