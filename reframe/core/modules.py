#
# Utilities for manipulating the modules subsystem
#

import abc
import os
import re
import reframe.utility.os as os_ext
import subprocess

from reframe.core.exceptions import ConfigError, EnvironError


class Module:
    """Module wrapper.

    This class represents internally a module. Concrete module system
    implementation should deal only with that.
    """

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
    """Implements the frontend of the module systems."""

    def __init__(self, backend):
        self._backend = backend

    @property
    def backend(self):
        return(self._backend)

    def loaded_modules(self):
        """Return a list of loaded modules.

        This method returns a list of strings.
        """
        return [str(m) for m in self._backend.loaded_modules()]

    def conflicted_modules(self, name):
        """Return the list of conflicted modules.

        This method returns a list of strings.
        """
        return [str(m) for m in self._backend.conflicted_modules(Module(name))]

    def load_module(self, name, force=False):
        """Load the module `name'.

        If ``force`` is set, forces the loading,
        unloading first any conflicting modules currently loaded.

        Returns the list of unloaded modules as strings."""
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
        """Unload module ``name``."""
        self._backend.unload_module(Module(name))

    def is_module_loaded(self, name):
        """Check presence of module ``name``."""
        return self._backend.is_module_loaded(Module(name))

    @property
    def name(self):
        """Return the name of this module system."""
        return self._backend.name()

    @property
    def version(self):
        """Return the version of this module system."""
        return self._backend.version()

    def unload_all(self):
        """Unload all loaded modules."""
        return self._backend.unload_all()

    @property
    def searchpath(self):
        """The module system search path as a list of directories."""
        return self._backend.searchpath()

    def searchpath_add(self, *dirs):
        """Add ``dirs`` to the module system search path."""
        return self._backend.searchpath_add(*dirs)

    def searchpath_remove(self, *dirs):
        """Remove ``dirs`` from the module system search path."""
        return self._backend.searchpath_remove(*dirs)

    def __str__(self):
        return str(self._backend)


class ModulesSystemImpl(abc.ABC):
    """Abstract base class for module systems."""

    @abc.abstractmethod
    def loaded_modules(self):
        """Return a list of loaded modules.

        This method returns a list of Module instances.
        """

    @abc.abstractmethod
    def conflicted_modules(self, module):
        """Return the list of conflicted modules.

        This method returns a list of Module instances.
        """

    @abc.abstractmethod
    def load_module(self, module):
        """Load the module `name'.

        If ``force`` is set, forces the loading,
        unloading first any conflicting modules currently loaded.

        Returns the unloaded modules as a list of module instances."""

    @abc.abstractmethod
    def unload_module(self, module):
        """Unload module ``module``."""

    @abc.abstractmethod
    def is_module_loaded(self, module):
        """Check presence of module ``module``."""

    @abc.abstractmethod
    def name(self):
        """Return the name of this module system."""

    @abc.abstractmethod
    def version(self):
        """Return the version of this module system."""

    @abc.abstractmethod
    def unload_all(self):
        """Unload all loaded modules."""

    @abc.abstractmethod
    def searchpath(self):
        """The module system search path as a list of directories."""

    @abc.abstractmethod
    def searchpath_add(self, *dirs):
        """Add ``dirs`` to the module system search path."""

    @abc.abstractmethod
    def searchpath_remove(self, *dirs):
        """Remove ``dirs`` from the module system search path."""

    def __repr__(self):
        return type(self).__name__ + '()'

    def __str__(self):
        return self.name() + ' ' + self.version()


class TModImpl(ModulesSystemImpl):
    """Module system for TMod (Tcl)."""

    def __init__(self):
        # Try to figure out if we are indeed using the TCL version
        try:
            completed = os_ext.run_command('modulecmd -V')
        except OSError as e:
            raise ConfigError(
                'could not find a sane Tmod installation: %s' % e) from e

        version_match = re.search(r'^VERSION=(\S+)', completed.stdout,
                                  re.MULTILINE)
        tcl_version_match = re.search(r'^TCL_VERSION=(\S+)', completed.stdout,
                                      re.MULTILINE)

        if version_match is None or tcl_version_match is None:
            raise ConfigError('could not find a sane Tmod installation')

        self._version = version_match.group(1)
        self._command = 'modulecmd python'
        try:
            # Try the Python bindings now
            completed = os_ext.run_command(self._command)
        except OSError as e:
            raise ConfigError(
                'could not get the Python bindings for Tmod: ' % e) from e

        if re.search(r'Unknown shell type', completed.stderr):
            raise ConfigError(
                'Python is not supported by this Tmod installation')

    def name(self):
        return 'tmod'

    def version(self):
        return self._version

    def _run_module_command(self, *args):
        command = [self._command, *args]
        return os_ext.run_command(' '.join(command))

    def _exec_module_command(self, *args):
        completed = self._run_module_command(*args)
        exec(completed.stdout)

    def loaded_modules(self):
        try:
            # LOADEDMODULES may be defined but empty
            return [Module(m)
                    for m in os.environ['LOADEDMODULES'].split(':') if m]
        except KeyError:
            return []

    def conflicted_modules(self, module):
        conflict_list = []
        completed = self._run_module_command('show', str(module))
        return [Module(m.group(1))
                for m in re.finditer(r'^conflict\s+(\S+)',
                                     completed.stderr, re.MULTILINE)]

    def is_module_loaded(self, module):
        return module in self.loaded_modules()

    def load_module(self, module):
        self._exec_module_command('load', str(module))
        if not self.is_module_loaded(module):
            raise EnvironError('could not load module %s' % module)

    def unload_module(self, module):
        self._exec_module_command('unload', str(module))
        if self.is_module_loaded(module):
            raise EnvironError('could not unload module %s' % module)

    def unload_all(self):
        self._exec_module_command('purge')

    def searchpath(self):
        return os.environ['MODULEPATH'].split(':')

    def searchpath_add(self, *dirs):
        self._exec_module_command('use', *dirs)

    def searchpath_remove(self, *dirs):
        self._exec_module_command('unuse', *dirs)


class NoModImpl(ModulesSystemImpl):
    """A convenience class that implements a no-op a modules system."""

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


# The module system used by the framework
_modules_system = None


def init_modules_system(modules_kind=None):
    global _modules_system

    if modules_kind is None:
        _modules_system = ModulesSystem(NoModImpl())
    elif modules_kind == 'tmod':
        _modules_system = ModulesSystem(TModImpl())
    else:
        raise ConfigError('unknown module system')


def get_modules_system():
    if _modules_system is None:
        raise ConfigError('no modules system is configured')

    return _modules_system
