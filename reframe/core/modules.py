#
# Utilities for manipulating the modules subsystem
#

import os
import subprocess
import re
import reframe
import reframe.core.debug as debug
import reframe.utility.os as os_ext

from reframe.core.exceptions import ModuleError


class Module:
    """Module wrapper.

    We basically need it for defining operators for use in standard Python
    algorithms."""

    def __init__(self, name):
        if not name:
            raise ModuleError('no module name specified')

        name_parts = name.split('/')
        self._name = name_parts[0]
        if len(name_parts) > 1:
            self._version = name_parts[1]
        else:
            self._version = None

    @property
    def name(self):
        return self._name

    @property
    def version(self):
        return self._version

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        if self._version is None or other._version is None:
            return self._name == other._name

        return self._name == other._name and self._version == other._version

    def __repr__(self):
        return debug.repr(self)

    def __str__(self):
        if self._version:
            return '%s/%s' % (self._name, self._version)
        else:
            return self._name


def module_equal(rhs, lhs):
    return Module(rhs) == Module(lhs)


def module_list():
    try:
        # LOADEDMODULES may be defined but empty
        return [m for m in os.environ['LOADEDMODULES'].split(':') if m]
    except KeyError:
        return []


def module_conflict_list(name):
    """Return the list of conflicted packages"""
    conflict_list = []
    completed = os_ext.run_command(
        cmd='%s show %s' % (reframe.MODULECMD_PYTHON, name))

    # Search for lines starting with 'conflict'
    for line in completed.stderr.split('\n'):
        match = re.search('^conflict\s+(?P<module_name>\S+)', line)
        if match:
            conflict_list.append(match.group('module_name'))

    return conflict_list


def module_present(name):
    for m in module_list():
        if module_equal(m, name):
            return True

    return False


def module_load(name):
    completed = os_ext.run_command(
        cmd='%s load %s' % (reframe.MODULECMD_PYTHON, name))
    exec(completed.stdout)

    if not module_present(name):
        raise ModuleError('Could not load module %s' % name)


def module_force_load(name):
    """Forces the loading of package `name', unloading first any conflicting
    currently loaded modules.

    Returns the a list of unloaded packages
    """
    # Do not try to load the module if it is already present
    if module_present(name):
        return []

    # Discard the version information of the loaded modules
    loaded_modules = set([m.split('/')[0] for m in module_list()])
    conflict_list  = set(module_conflict_list(name))
    unload_list    = loaded_modules & conflict_list
    for m in unload_list:
        module_unload(m)

    module_load(name)
    return list(unload_list)


def module_unload(name):
    completed = os_ext.run_command(
        cmd='%s unload %s' % (reframe.MODULECMD_PYTHON, name))
    exec(completed.stdout)

    if module_present(name):
        raise ModuleError('Could not unload module %s' % name)


def module_purge():
    completed = os_ext.run_command(
        cmd='%s purge' % reframe.MODULECMD_PYTHON)
    exec(completed.stdout)


def module_path_add(dirs):
    """
    Adds list of dirs to module path
    """
    args = ' '.join(dirs)
    completed = os_ext.run_command(
        cmd='%s use %s' % (reframe.MODULECMD_PYTHON, args))
    exec(completed.stdout)


def module_path_remove(dirs):
    """
    Removes list of dirs from module path
    """
    args = ' '.join(dirs)
    completed = os_ext.run_command(
        cmd='%s unuse %s' % (reframe.MODULECMD_PYTHON, args))
    exec(completed.stdout)
