#
# Regression test loader
#

import os
import logging
import sys
import reframe.utility.os as os_ext

from importlib.machinery import SourceFileLoader
from reframe.core.environments import Environment, ProgEnvironment
from reframe.core.exceptions import ConfigurationError, ReframeError
from reframe.core.systems import System, SystemPartition
from reframe.core.fields import ScopedDict
from reframe.settings import settings


class RegressionCheckLoader:
    def __init__(self, load_path, prefix = '', recurse = False):
        self.load_path = load_path
        self.prefix = prefix if prefix != None else ''
        self.recurse = recurse


    def _module_name(self, filename):
        """Figure out a module name from filename.

        If filename is an absolute path, module name will the basename without
        the extension. Otherwise, it will be the same as path with `/' replaced
        by `.' and without the final file extension."""
        if os.path.isabs(filename):
            return os.path.splitext(os.path.basename(filename))[0]
        else:
            return (os.path.splitext(filename)[0]).replace('/', '.')


    def load_from_module(self, module, **check_args):
        """Load user checks from module.

        This method tries to call the `_get_checks()` method of the user check
        and validates its return value."""
        from reframe.core.pipeline import RegressionTest

        if hasattr(module, '_get_checks') and callable(module._get_checks):
            try:
                checks = [ c for c in module._get_checks(**check_args)
                           if isinstance(c, RegressionTest) ]
            except TypeError:
                # Guard against _get_checks() returning a non-iterable
                checks = []
        else:
            checks = []

        return checks


    def load_from_file(self, filename, **check_args):
        module_name = self._module_name(filename)
        try:
            loader = SourceFileLoader(module_name, filename)
            return self.load_from_module(loader.load_module(), **check_args)
        except OSError as e:
            raise ReframeError(
                "Could not load module `%s' from file `%s': %s" % \
                (module_name, filename, e.strerror))


    def load_from_dir(self, dirname, recurse=False, **check_args):
        checks = []
        for entry in os.scandir(dirname):
            if recurse and entry.is_dir():
                checks.extend(
                    self.load_from_dir(entry.path, recurse, **check_args)
                )

            if     entry.name.startswith('.') or \
               not entry.name.endswith('.py') or \
               not entry.is_file():
                continue

            checks.extend(self.load_from_file(entry.path, **check_args))

        return checks


    def load_all(self, **check_args):
        """Load all checks in self.load_path.

        If a self.prefix exists, it will be prepended to each path."""
        checks = []
        for d in self.load_path:
            d = os.path.join(self.prefix, d)
            if not os.path.exists(d):
                continue
            if os.path.isdir(d):
                checks.extend(self.load_from_dir(d, self.recurse,
                                                  **check_args))
            else:
                checks.extend(self.load_from_file(d, **check_args))

        return checks


class SiteConfiguration:
    """Holds the configuration of systems and environments"""
    def __init__(self):
        self.systems = {}


    def load_from_dict(self, site_config):
        if not isinstance(site_config, dict):
            raise ConfigurationError('site configuration is not a dict')

        sysconfig = site_config.get('systems', None)
        envconfig = site_config.get('environments', None)

        if not sysconfig:
            raise ConfigurationError('no entry for systems was found')

        if not envconfig:
            raise ConfigurationError('no entry for environments was found')

        # Convert envconfig to a ScopedDict
        try:
            envconfig = ScopedDict(envconfig)
        except TypeError:
            raise ConfigurationError('environments configuration '
                                     'is not properly formatted')

        def create_env(system, partition, name):
            # Create an environment instance
            try:
                config  = envconfig['%s:%s:%s' % (system, partition, name)]
            except KeyError as e:
                raise ConfigurationError(
                    "could not find a definition for `%s'" % name
                )

            if not isinstance(config, dict):
                raise ConfigurationError(
                    "config for `%s' is not a dictionary" % name
                )

            try:
                envtype = globals()[config['type']]
                return envtype(name, **config)
            except KeyError:
                raise ConfigurationError("no type specified for `%s'" % name)


        # Populate the systems directory
        for sysname, config in sysconfig.items():
            if not isinstance(config, dict):
                raise ConfigurationError(
                    'system configuration is not a dictionary'
                )

            if not isinstance(config['partitions'], dict):
                raise ConfigurationError('partitions must be a dictionary')

            system = System(sysname)
            system.descr     = config.get('descr', sysname)
            system.hostnames = config.get('hostnames', [])

            # prefix must always be defined; the rest default to None, so as to
            # be set from the ResourcesManager
            system.prefix    = config.get('prefix', '.')
            system.stagedir  = config.get('stagedir', None)
            system.outputdir = config.get('outputdir', None)
            system.logdir    = config.get('logdir', None)

            # expand variables
            if system.prefix:
                system.prefix = os.path.expandvars(system.prefix)

            if system.stagedir:
                system.stagedir = os.path.expandvars(system.stagedir)

            if system.outputdir:
                system.outputdir = os.path.expandvars(system.outputdir)

            if system.logdir:
                system.logdir = os.path.expandvars(system.logdir)

            for partname, partconfig in config.get('partitions', {}).items():
                if not isinstance(partconfig, dict):
                    raise ConfigurationError(
                        "partition `%s' not configured "
                        "as a dictionary" % partname
                    )

                partition = SystemPartition(partname)
                partition.descr = partconfig.get('descr', partname)
                partition.scheduler = partconfig.get('scheduler', 'local')
                partition.local_env = Environment(
                    name='__env_%s' % partname,
                    modules=partconfig.get('modules', []),
                    variables=partconfig.get('variables', {})
                )
                partition.environs = [
                    create_env(sysname, partname, e) \
                    for e in partconfig.get('environs', [])
                ]
                partition.access = partconfig.get('access', [])
                partition.resources = partconfig.get('resources', {})
                system.partitions.append(partition)

            self.systems[sysname] = system


def autodetect_system(site_config):
    """Auto-detect system"""
    import re
    import socket

    # Try to detect directly the cluster name from /etc/xthostname (Cray
    # specific)
    try:
        hostname = os_ext.run_command('cat /etc/xthostname', check=True).stdout
    except ReframeError:
        # Try to figure it out with the standard method
        hostname = socket.gethostname()

    # Go through the supported systems and try to match the hostname
    for system in site_config.systems.values():
        for hostname_patt in system.hostnames:
            if re.match(hostname_patt, hostname):
                return system

    return None
