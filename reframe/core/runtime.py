#
# Handling of the current host context
#

import os
import functools
import re
import shutil
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
    def __init__(self, system, partname=None):
        self._system = system
        self._partname = partname

    def __getattr__(self, attr):
        # Delegate any failed attribute lookup to our backend
        return getattr(self._system, attr)

    # We redefine the following properties just to document them for the users
    @property
    def name(self):
        """The name of this system."""
        return self._system.name

    @property
    def descr(self):
        """The description of this system."""
        return self._system.descr

    @property
    def resourcesdir(self):
        """Global resources directory for this system.

        You may use this directory for storing large resource files of your
        regression tests.
        See `here <configure.html#system-configuration>`__ on how to configure
        this.

        :type: :class:`str`
        """
        return self._system.resourcesdir

    @property
    def partitions(self):
        """The partitions of this system.

        :type: a list of :class:`reframe.core.systems.SystemPartition`.
        """

        if not self._partname:
            return self._system.partitions

        return [p for p in self._system.partitions if p.name == self._partname]

    def partition(self, name):
        for p in self.partitions:
            if p.name == name:
                return p

        return None

    def __str__(self):
        return str(self._system)

    def __repr__(self):
        return 'HostSystem(%r, %r)' % (self._system, self._partname)


class HostResources:
    prefix = fields.AbsolutePathField('prefix')
    outputdir = fields.AbsolutePathField('outputdir', allow_none=True)
    stagedir  = fields.AbsolutePathField('stagedir', allow_none=True)
    perflogdir = fields.AbsolutePathField('perflogdir', allow_none=True)

    def __init__(self, prefix=None, stagedir=None,
                 outputdir=None, perflogdir=None, timefmt=None):
        self.prefix = prefix or '.'
        self.stagedir  = stagedir
        self.outputdir = outputdir
        self.perflogdir = perflogdir
        self._timestamp = datetime.now()
        self.timefmt = timefmt

    def _makedir(self, *dirs, wipeout=False):
        ret = os.path.join(*dirs)
        if wipeout:
            shutil.rmtree(ret, True)

        os.makedirs(ret, exist_ok=True)
        return ret

    @property
    def timestamp(self):
        return self._timestamp.strftime(self.timefmt) if self.timefmt else ''

    @property
    def output_prefix(self):
        if self.outputdir is None:
            return os.path.join(self.prefix, 'output', self.timestamp)
        else:
            return os.path.join(self.outputdir, self.timestamp)

    @property
    def stage_prefix(self):
        if self.stagedir is None:
            return os.path.join(self.prefix, 'stage', self.timestamp)
        else:
            return os.path.join(self.stagedir, self.timestamp)

    @property
    def perflog_prefix(self):
        if self.perflogdir is None:
            return os.path.join(self.prefix, 'logs')
        else:
            return self.perflogdir

    def make_stagedir(self, *dirs, wipeout=True):
        return self._makedir(self.stage_prefix, *dirs, wipeout=wipeout)

    def make_outputdir(self, *dirs, wipeout=True):
        return self._makedir(self.output_prefix, *dirs, wipeout=wipeout)

    def make_perflogdir(self, *dirs, wipeout=False):
        return self._makedir(self.perflog_prefix, *dirs, wipeout=wipeout)


class RuntimeContext:
    def __init__(self, dict_config, sysdescr=None):
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
            self._system.outputdir, self._system.logdir)
        self._modules_system = ModulesSystem.create(
            self._system.modules_system)

    def _autodetect_system(self):
        """Auto-detect system."""

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

    @property
    def system(self):
        return self._system

    @property
    def resources(self):
        return self._resources

    @property
    def modules_system(self):
        return self._modules_system



# Global resources for the current host
_runtime_context = None


def init_runtime(dict_config, sysname=None):
    global _runtime_context

    if _runtime_context is None:
        _runtime_context = RuntimeContext(dict_config, sysname)


def runtime():
    if _runtime_context is None:
        raise ReframeFatalError('no runtime context is configured')

    return _runtime_context


# The following utilities are useful only for the unit tests

class temp_runtime:
    """Context manager to temporarily switch to another runtime."""

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
    """Function decorator for temporarily changing the runtime for a function."""
    def _runtime_deco(fn):
        @functools.wraps(fn)
        def _fn(*args, **kwargs):
            with temp_runtime(dict_config, sysname):
                ret = fn(*args, **kwargs)

            return ret

        return _fn

    return _runtime_deco
