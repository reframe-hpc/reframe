import os
import collections.abc

import reframe.core.debug as debug
import reframe.utility.os_ext as os_ext
from reframe.core.environments import Environment
from reframe.core.exceptions import ConfigError, ReframeError, ReframeFatalError
from reframe.core.fields import ScopedDict, ScopedDictField
from reframe.core.launchers.registry import getlauncher
from reframe.core.schedulers.registry import getscheduler
from reframe.core.systems import System, SystemPartition
from reframe.utility import import_module_from_file

_settings = None

def load_from_file(filename):
    global _settings
    _settings = import_module_from_file(filename)
    return _settings


def settings():
    if _settings is None:
        raise ReframeFatalError('No configuration loaded')

    return _settings


class SiteConfiguration:
    """Holds the configuration of systems and environments"""
    _modes = ScopedDictField('_modes', (list, str))

    def __init__(self):
        self._systems = {}
        self._modes = {}

    def __repr__(self):
        return debug.repr(self)

    @property
    def systems(self):
        return self._systems

    @property
    def modes(self):
        return self._modes

    def get_schedsystem_config(self, descr):
        # Handle the special shortcuts first
        if descr == 'nativeslurm':
            return getscheduler('slurm'), getlauncher('srun')

        if descr == 'local':
            return getscheduler('local'), getlauncher('local')

        try:
            sched_descr, launcher_descr = descr.split('+')
        except ValueError as e:
            raise ValueError('invalid syntax for the '
                             'scheduling system: %s' % descr) from None

        return getscheduler(sched_descr), getlauncher(launcher_descr)

    def load_from_dict(self, site_config):
        if not isinstance(site_config, collections.abc.Mapping):
            raise TypeError('site configuration is not a dict')

        sysconfig = site_config.get('systems', None)
        envconfig = site_config.get('environments', None)
        modes = site_config.get('modes', {})

        if not sysconfig:
            raise ValueError('no entry for systems was found')

        if not envconfig:
            raise ValueError('no entry for environments was found')

        # Convert envconfig to a ScopedDict
        try:
            envconfig = ScopedDict(envconfig)
        except TypeError:
            raise TypeError('environments configuration '
                            'is not a scoped dictionary') from None

        # Convert modes to a `ScopedDict`; note that `modes` will implicitly
        # converted to a scoped dict here, since `self._models` is a
        # `ScopedDictField`.
        try:
            self._modes = modes
        except TypeError:
            raise TypeError('modes configuration '
                            'is not a scoped dictionary') from None

        def create_env(system, partition, name):
            # Create an environment instance
            try:
                config = envconfig['%s:%s:%s' % (system, partition, name)]
            except KeyError as e:
                raise ConfigError(
                    "could not find a definition for `%s'" % name
                ) from None

            if not isinstance(config, collections.abc.Mapping):
                raise TypeError("config for `%s' is not a dictionary" % name)

            try:
                import reframe.core.environments as m_env

                envtype = m_env.__dict__[config['type']]
                return envtype(name, **config)
            except KeyError:
                raise ConfigError("no type specified for environment `%s'" %
                                  name) from None

        # Populate the systems directory
        for sys_name, config in sysconfig.items():
            if not isinstance(config, dict):
                raise TypeError('system configuration is not a dictionary')

            if not isinstance(config['partitions'], collections.abc.Mapping):
                raise TypeError('partitions must be a dictionary')

            sys_descr = config.get('descr', sys_name)
            sys_hostnames = config.get('hostnames', [])

            # The System's constructor provides also reasonable defaults, but
            # since we are going to set them anyway from the values provided by
            # the configuration, we should set default values here. The stage,
            # output and log directories default to None, since they are
            # going to be set dynamically by the ResourcesManager
            sys_prefix = config.get('prefix', '.')
            sys_stagedir = config.get('stagedir', None)
            sys_outputdir = config.get('outputdir', None)
            sys_logdir = config.get('logdir', None)
            sys_resourcesdir = config.get('resourcesdir', '.')
            sys_modules_system = config.get('modules_system', None)

            # Expand variables
            if sys_prefix:
                sys_prefix = os.path.expandvars(sys_prefix)

            if sys_stagedir:
                sys_stagedir = os.path.expandvars(sys_stagedir)

            if sys_outputdir:
                sys_outputdir = os.path.expandvars(sys_outputdir)

            if sys_logdir:
                sys_logdir = os.path.expandvars(sys_logdir)

            if sys_resourcesdir:
                sys_resourcesdir = os.path.expandvars(sys_resourcesdir)

            system = System(name=sys_name,
                            descr=sys_descr,
                            hostnames=sys_hostnames,
                            prefix=sys_prefix,
                            stagedir=sys_stagedir,
                            outputdir=sys_outputdir,
                            logdir=sys_logdir,
                            resourcesdir=sys_resourcesdir,
                            modules_system=sys_modules_system)
            for part_name, partconfig in config.get('partitions', {}).items():
                if not isinstance(partconfig, collections.abc.Mapping):
                    raise TypeError("partition `%s' not configured "
                                    "as a dictionary" % part_name)

                part_descr = partconfig.get('descr', part_name)
                part_scheduler, part_launcher = self.get_schedsystem_config(
                    partconfig.get('scheduler', 'local+local')
                )
                part_local_env = Environment(
                    name='__rfm_env_%s' % part_name,
                    modules=partconfig.get('modules', []),
                    variables=partconfig.get('variables', {})
                )
                part_environs = [
                    create_env(sys_name, part_name, e)
                    for e in partconfig.get('environs', [])
                ]
                part_access = partconfig.get('access', [])
                part_resources = partconfig.get('resources', {})
                part_max_jobs = partconfig.get('max_jobs', 1)
                system.add_partition(SystemPartition(name=part_name,
                                                     descr=part_descr,
                                                     scheduler=part_scheduler,
                                                     launcher=part_launcher,
                                                     access=part_access,
                                                     environs=part_environs,
                                                     resources=part_resources,
                                                     local_env=part_local_env,
                                                     max_jobs=part_max_jobs))

            self._systems[sys_name] = system


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
