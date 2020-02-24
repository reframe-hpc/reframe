# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Handling of the current host context
#

import os
import functools
import re
import socket
from datetime import datetime

import reframe.core.config as config
import reframe.core.fields as fields
import reframe.utility.os_ext as os_ext
from reframe.core.exceptions import (ConfigError,
                                     ReframeFatalError,
                                     SpawnedProcessError,
                                     SystemAutodetectionError,
                                     UnknownSystemError)
from reframe.core.modules import ModulesSystem


class HostSystem:
    '''The host system of the framework.

    The host system is a representation of the system that the framework
    currently runs on.If the framework is properly configured, the host
    system is automatically detected. If not, it may be explicitly set by the
    user.

    This class is mainly a proxy of :class:`reframe.core.systems.System` that
    stores optionally a partition name and provides some additional
    functionality for manipulating system partitions.

    All attributes of the :class:`reframe.core.systems.System` may be accessed
    directly from this proxy.

    .. note::
       .. versionadded:: 2.13
    '''

    def __init__(self, system, partname=None):
        self._system = system
        self._partname = partname

    def __getattr__(self, attr):
        # Delegate any failed attribute lookup to our backend
        return getattr(self._system, attr)

    @property
    def partitions(self):
        '''The partitions of this system.

        :type: :class:`list[reframe.core.systems.SystemPartition]`.
        '''

        if not self._partname:
            return self._system.partitions

        return [p for p in self._system.partitions if p.name == self._partname]

    def partition(self, name):
        '''Return the system partition ``name``.

        :type: :class:`reframe.core.systems.SystemPartition`.
        '''
        for p in self.partitions:
            if p.name == name:
                return p

        return None

    def __str__(self):
        partitions = '\n'.join(re.sub('(?m)^', 6*' ', '- ' + str(p))
                               for p in self.partitions)
        lines = [
            '%s [%s]:' % (self._name, self._descr),
            '    hostnames: ' + ', '.join(self._hostnames),
            '    modules_system: ' + str(self._modules_system),
            '    resourcesdir: ' + self._resourcesdir,
            '    partitions:\n' + partitions,
        ]
        return '\n'.join(lines)

    def __repr__(self):
        return 'HostSystem(%r, %r)' % (self._system, self._partname)


class HostResources:
    '''Resources associated with ReFrame execution on the current host.

    .. note::
       .. versionadded:: 2.13
    '''

    #: The prefix directory of ReFrame execution.
    #: This is always an absolute path.
    #:
    #: :type: :class:`str`
    #:
    #: .. caution::
    #:    Users may not set this field.
    #:
    prefix = fields.AbsolutePathField('prefix')
    outputdir = fields.AbsolutePathField('outputdir', type(None))
    stagedir = fields.AbsolutePathField('stagedir', type(None))
    perflogdir = fields.AbsolutePathField('perflogdir', type(None))

    def __init__(self, prefix=None, stagedir=None,
                 outputdir=None, perflogdir=None, timefmt=None):
        self.prefix = prefix or '.'
        self.stagedir = stagedir
        self.outputdir = outputdir
        self.perflogdir = perflogdir
        self.timefmt = timefmt
        self._timestamp = datetime.now()

    def _makedir(self, *dirs, wipeout=False):
        ret = os.path.join(*dirs)
        if wipeout:
            os_ext.rmtree(ret, ignore_errors=True)

        os.makedirs(ret, exist_ok=True)
        return ret

    def _format_dirs(self, *dirs):
        try:
            last = dirs[-1]
        except IndexError:
            return dirs

        current_run = runtime().current_run
        if current_run == 0:
            return dirs

        last += '_retry%s' % current_run
        return (*dirs[:-1], last)

    @property
    def timestamp(self):
        return self._timestamp.strftime(self.timefmt) if self.timefmt else ''

    @property
    def output_prefix(self):
        '''The output prefix directory of ReFrame.'''
        if self.outputdir is None:
            return os.path.join(self.prefix, 'output', self.timestamp)
        else:
            return os.path.join(self.outputdir, self.timestamp)

    @property
    def stage_prefix(self):
        '''The stage prefix directory of ReFrame.'''
        if self.stagedir is None:
            return os.path.join(self.prefix, 'stage', self.timestamp)
        else:
            return os.path.join(self.stagedir, self.timestamp)

    @property
    def perflog_prefix(self):
        if self.perflogdir is None:
            return os.path.join(self.prefix, 'perflogs')
        else:
            return self.perflogdir

    def make_stagedir(self, *dirs, wipeout=True):
        return self._makedir(self.stage_prefix,
                             *self._format_dirs(*dirs), wipeout=wipeout)

    def make_outputdir(self, *dirs, wipeout=True):
        return self._makedir(self.output_prefix,
                             *self._format_dirs(*dirs), wipeout=wipeout)


class RuntimeContext:
    '''The runtime context of the framework.

    This class essentially groups the current host system and the associated
    resources of the framework on the current system.
    It also encapsulates other runtime parameters that are relevant to the
    framework's execution.

    There is a single instance of this class globally in the framework.

    .. note::
       .. versionadded:: 2.13

    '''

    def __init__(self, dict_config, sysdescr=None, **options):
        self._site_config = config.SiteConfiguration(dict_config)
        if sysdescr is not None:
            sysname, _, partname = sysdescr.partition(':')
            try:
                self._system = HostSystem(
                    self._site_config.systems[sysname], partname)
            except KeyError:
                raise UnknownSystemError('unknown system: %s' %
                                         sysdescr) from None
        else:
            self._system = HostSystem(self._autodetect_system())

        self._resources = HostResources(
            self._system.prefix, self._system.stagedir,
            self._system.outputdir, self._system.perflogdir)
        self._modules_system = ModulesSystem.create(
            self._system.modules_system)
        self._current_run = 0
        self._non_default_craype = options.get('non_default_craype', False)

    def _autodetect_system(self):
        '''Auto-detect system.'''

        # Try to detect directly the cluster name from /etc/xthostname (Cray
        # specific)
        try:
            hostname = os_ext.run_command(
                'cat /etc/xthostname', check=True).stdout
        except SpawnedProcessError:
            # Try to figure it out with the standard method
            hostname = socket.gethostname()

        # Go through the supported systems and try to match the hostname
        for system in self._site_config.systems.values():
            for hostname_patt in system.hostnames:
                if re.match(hostname_patt, hostname):
                    return system

        raise SystemAutodetectionError

    def mode(self, name):
        try:
            return self._site_config.modes[name]
        except KeyError:
            raise ConfigError('unknown execution mode: %s' % name) from None

    def next_run(self):
        self._current_run += 1

    @property
    def current_run(self):
        return self._current_run

    @property
    def system(self):
        '''The current host system.

        :type: :class:`reframe.core.runtime.HostSystem`
        '''
        return self._system

    @property
    def resources(self):
        '''The framework resources.

        :type: :class:`reframe.core.runtime.HostResources`
        '''
        return self._resources

    @property
    def modules_system(self):
        '''The modules system used by the current host system.

        :type: :class:`reframe.core.modules.ModulesSystem`.
        '''
        return self._modules_system

    @property
    def non_default_craype(self):
        '''True if a non-default Cray PE is tested.

        This will cause ReFrame to set the ``LD_LIBRARY_PATH`` as follows after
        all modules have been loaded:

        .. code:: shell

            export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH


        This property is set through the ``--non-default-craype`` command-line
        option.

        :type: :class:`bool` (default: :class:`False`)

        '''
        return self._non_default_craype

    def show_config(self):
        '''Return a textual representation of the current runtime.'''
        return str(self._system)


# Global resources for the current host
_runtime_context = None


def init_runtime(dict_config, sysname=None, **options):
    global _runtime_context

    if _runtime_context is None:
        _runtime_context = RuntimeContext(dict_config, sysname, **options)


def runtime():
    '''Retrieve the framework's runtime context.

    :type: :class:`reframe.core.runtime.RuntimeContext`

    .. note::
       .. versionadded:: 2.13
    '''
    if _runtime_context is None:
        raise ReframeFatalError('no runtime context is configured')

    return _runtime_context


# The following utilities are useful only for the unit tests

class temp_runtime:
    '''Context manager to temporarily switch to another runtime.'''

    def __init__(self, dict_config, sysname=None):
        global _runtime_context
        self._runtime_save = _runtime_context
        if dict_config is None:
            _runtime_context = None
        else:
            _runtime_context = RuntimeContext(dict_config, sysname)

    def __enter__(self):
        return _runtime_context

    def __exit__(self, exc_type, exc_value, traceback):
        global _runtime_context
        _runtime_context = self._runtime_save


def switch_runtime(dict_config, sysname=None):
    '''Function decorator for temporarily changing the runtime for a
    function.'''
    def _runtime_deco(fn):
        @functools.wraps(fn)
        def _fn(*args, **kwargs):
            with temp_runtime(dict_config, sysname):
                ret = fn(*args, **kwargs)

            return ret

        return _fn

    return _runtime_deco


class module_use:
    '''Context manager for temporarily modifying the module path'''

    def __init__(self, *paths):
        self._paths = paths

    def __enter__(self):
        runtime().modules_system.searchpath_add(*self._paths)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        runtime().modules_system.searchpath_remove(*self._paths)
