# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import collections.abc
import re

import reframe.core.debug as debug
import reframe.core.fields as fields
import reframe.utility as util
import reframe.utility.os_ext as os_ext
import reframe.utility.typecheck as types
from reframe.core.exceptions import (ConfigError, ReframeFatalError)


_settings = None


def load_settings_from_file(filename):
    global _settings
    try:
        _settings = util.import_module_from_file(filename).settings
        return _settings
    except Exception as e:
        raise ConfigError(
            "could not load configuration file `%s'" % filename) from e


def settings():
    if _settings is None:
        raise ReframeFatalError('ReFrame is not configured')

    return _settings


class SiteConfiguration:
    '''Holds the configuration of systems and environments'''
    _modes = fields.ScopedDictField('_modes', types.List[str])

    def __init__(self, dict_config=None):
        self._systems = {}
        self._modes = {}
        if dict_config is not None:
            self.load_from_dict(dict_config)

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
        from reframe.core.launchers.registry import getlauncher
        from reframe.core.schedulers.registry import getscheduler

        if descr == 'nativeslurm':
            return getscheduler('slurm'), getlauncher('srun')

        if descr == 'local':
            return getscheduler('local'), getlauncher('local')

        try:
            sched_descr, launcher_descr = descr.split('+')
        except ValueError:
            raise ValueError('invalid syntax for the '
                             'scheduling system: %s' % descr) from None

        return getscheduler(sched_descr), getlauncher(launcher_descr)

    def load_from_dict(self, site_config):
        if not isinstance(site_config, collections.abc.Mapping):
            raise TypeError('site configuration is not a dict')

        # We do all the necessary imports here and not on the top, because we
        # want to remove import time dependencies
        import reframe.core.environments as m_env
        from reframe.core.systems import System, SystemPartition

        sysconfig = site_config.get('systems', None)
        envconfig = site_config.get('environments', None)
        modes = site_config.get('modes', {})

        if not sysconfig:
            raise ValueError('no entry for systems was found')

        if not envconfig:
            raise ValueError('no entry for environments was found')

        # Convert envconfig to a ScopedDict
        try:
            envconfig = fields.ScopedDict(envconfig)
        except TypeError:
            raise TypeError('environments configuration '
                            'is not a scoped dictionary') from None

        # Convert modes to a `ScopedDict`; note that `modes` will implicitly
        # converted to a scoped dict here, since `self._modes` is a
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
            except KeyError:
                raise ConfigError(
                    "could not find a definition for `%s'" % name
                ) from None

            if not isinstance(config, collections.abc.Mapping):
                raise TypeError("config for `%s' is not a dictionary" % name)

            return m_env.ProgEnvironment(name, **config)

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
            # output and log directories default to None, since they are going
            # to be set dynamically by the runtime.
            sys_prefix = config.get('prefix', '.')
            sys_stagedir = config.get('stagedir', None)
            sys_outputdir = config.get('outputdir', None)
            sys_perflogdir = config.get('perflogdir', None)
            sys_resourcesdir = config.get('resourcesdir', '.')
            sys_modules_system = config.get('modules_system', None)

            # Expand variables
            if sys_prefix:
                sys_prefix = os_ext.expandvars(sys_prefix)

            if sys_stagedir:
                sys_stagedir = os_ext.expandvars(sys_stagedir)

            if sys_outputdir:
                sys_outputdir = os_ext.expandvars(sys_outputdir)

            if sys_perflogdir:
                sys_perflogdir = os_ext.expandvars(sys_perflogdir)

            if sys_resourcesdir:
                sys_resourcesdir = os_ext.expandvars(sys_resourcesdir)

            # Create the preload environment for the system
            sys_preload_env = m_env.Environment(
                name='__rfm_env_%s' % sys_name,
                modules=config.get('modules', []),
                variables=config.get('variables', {})
            )

            system = System(name=sys_name,
                            descr=sys_descr,
                            hostnames=sys_hostnames,
                            preload_env=sys_preload_env,
                            prefix=sys_prefix,
                            stagedir=sys_stagedir,
                            outputdir=sys_outputdir,
                            perflogdir=sys_perflogdir,
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
                part_local_env = m_env.Environment(
                    name='__rfm_env_%s' % part_name,
                    modules=partconfig.get('modules', []),
                    variables=partconfig.get('variables', {}).items()
                )
                part_environs = [
                    create_env(sys_name, part_name, e)
                    for e in partconfig.get('environs', [])
                ]
                part_access = partconfig.get('access', [])
                part_resources = partconfig.get('resources', {})
                part_max_jobs = partconfig.get('max_jobs', 1)
                part = SystemPartition(name=part_name,
                                       descr=part_descr,
                                       scheduler=part_scheduler,
                                       launcher=part_launcher,
                                       access=part_access,
                                       environs=part_environs,
                                       resources=part_resources,
                                       local_env=part_local_env,
                                       max_jobs=part_max_jobs)

                container_platforms = partconfig.get('container_platforms', {})
                for cp, env_spec in container_platforms.items():
                    cp_env = m_env.Environment(
                        name='__rfm_env_%s' % cp,
                        modules=env_spec.get('modules', []),
                        variables=env_spec.get('variables', {})
                    )
                    part.add_container_env(cp, cp_env)

                system.add_partition(part)

            self._systems[sys_name] = system
