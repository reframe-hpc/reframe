# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Handling of the current host context
#

import os
import functools
from datetime import datetime

import reframe.core.config as config
import reframe.utility.osext as osext
from reframe.core.environments import (Environment, snapshot)
from reframe.core.exceptions import ReframeFatalError
from reframe.core.logging import getlogger
from reframe.core.systems import System


class RuntimeContext:
    '''The runtime context of the framework.

    There is a single instance of this class globally in the framework.

    .. versionadded:: 2.13
    '''

    def __init__(self, site_config):
        self._site_config = site_config
        self._system = System.create(site_config)
        self._current_run = 0
        self._timestamp = datetime.now()

    def _makedir(self, *dirs, wipeout=False):
        ret = os.path.join(*dirs)
        if wipeout:
            osext.rmtree(ret, ignore_errors=True)

        os.makedirs(ret, exist_ok=True)
        return ret

    def _format_dirs(self, *dirs):
        if not self.get_option('general/0/clean_stagedir'):
            # If stagedir is to be reused, no new stage directories will be
            # used for retries
            return dirs

        try:
            last = dirs[-1]
        except IndexError:
            return dirs

        current_run = runtime().current_run
        if current_run == 0:
            return dirs

        last += '_retry%s' % current_run
        return (*dirs[:-1], last)

    def next_run(self):
        self._current_run += 1

    @property
    def current_run(self):
        return self._current_run

    @property
    def site_config(self):
        return self._site_config

    @property
    def system(self):
        '''The current host system.

        :type: :class:`reframe.core.systems.System`
        '''
        return self._system

    @property
    def prefix(self):
        return osext.expandvars(
            self.site_config.get('systems/0/prefix')
        )

    @property
    def stagedir(self):
        return osext.expandvars(
            self.site_config.get('systems/0/stagedir')
        )

    @property
    def outputdir(self):
        return osext.expandvars(
            self.site_config.get('systems/0/outputdir')
        )

    @property
    def perflogdir(self):
        # Find the first filelog handler
        handlers = self.site_config.get('logging/0/handlers_perflog')
        for i, h in enumerate(handlers):
            if h['type'] == 'filelog':
                break

        return osext.expandvars(
            self.site_config.get(f'logging/0/handlers_perflog/{i}/basedir')
        )

    @property
    def timestamp(self):
        timefmt = self.site_config.get('general/0/timestamp_dirs')
        return self._timestamp.strftime(timefmt)

    @property
    def output_prefix(self):
        '''The output directory prefix.

        :type: :class:`str`
        '''
        if self.outputdir:
            ret = os.path.join(self.outputdir, self.timestamp)
        else:
            ret = os.path.join(self.prefix, 'output', self.timestamp)

        return os.path.abspath(ret)

    @property
    def stage_prefix(self):
        '''The stage directory prefix.

        :type: :class:`str`
        '''
        if self.stagedir:
            ret = os.path.join(self.stagedir, self.timestamp)
        else:
            ret = os.path.join(self.prefix, 'stage', self.timestamp)

        return os.path.abspath(ret)

    def make_stagedir(self, *dirs):
        wipeout = self.get_option('general/0/clean_stagedir')
        ret = self._makedir(self.stage_prefix,
                            *self._format_dirs(*dirs), wipeout=wipeout)
        getlogger().debug(
            f'Created stage directory {ret!r} [clean_stagedir: {wipeout}]'
        )
        return ret

    def make_outputdir(self, *dirs):
        ret = self._makedir(self.output_prefix,
                            *self._format_dirs(*dirs), wipeout=True)
        getlogger().debug(f'Created output directory {ret!r}')
        return ret

    @property
    def modules_system(self):
        '''The environment modules system used in the current host.

        :type: :class:`reframe.core.modules.ModulesSystem`.
        '''
        return self._system.modules_system

    def get_option(self, option, default=None):
        '''Get a configuration option.

        :arg option: The option to be retrieved.
        :arg default: The value to return if ``option`` cannot be retrieved.
        :returns: The value of the option.

        .. versionchanged:: 3.11.0
          Add ``default`` named argument.
        '''
        return self._site_config.get(option, default=default)


# Global resources for the current host
_runtime_context = None


def init_runtime(site_config):
    global _runtime_context

    if _runtime_context is None:
        _runtime_context = RuntimeContext(site_config)


def runtime():
    '''Get the runtime context of the framework.

    .. versionadded:: 2.13

    :returns: A :class:`reframe.core.runtime.RuntimeContext` object.
    '''
    if _runtime_context is None:
        raise ReframeFatalError('no runtime context is configured')

    return _runtime_context


def loadenv(*environs):
    '''Load environments in the current Python context.

    :arg environs: A list of environments to load.
    :type environs: List[Environment]

    :returns: A tuple containing snapshot of the current environment upon
        entry to this function and a list of shell commands required to load
        the environments.
    :rtype: Tuple[_EnvironmentSnapshot, List[str]]

    '''

    def _load_cmds_tracked(**module):
        commands = []
        load_seq = modules_system.load_module(**module, force=True)
        for m, conflicted in load_seq:
            for c in conflicted:
                commands += modules_system.emit_unload_commands(c)

            commands += modules_system.emit_load_commands(
                m, module.get('collection', False), module.get('path', None)
            )

        return commands

    modules_system = runtime().modules_system
    env_snapshot = snapshot()
    commands = []
    for env in environs:
        for mod in env.modules_detailed:
            if runtime().get_option('general/0/resolve_module_conflicts'):
                commands += _load_cmds_tracked(**mod)
            else:
                commands += modules_system.emit_load_commands(**mod)

        for k, v in env.env_vars.items():
            os.environ[k] = osext.expandvars(v)
            commands.append(f'export {k}={v}')

    return env_snapshot, commands


def emit_loadenv_commands(*environs):
    env_snapshot = snapshot()
    try:
        _, commands = loadenv(*environs)
    finally:
        env_snapshot.restore()

    return commands


def is_env_loaded(environ):
    '''Check if environment is loaded.

    :arg environ: Environment to check for.
    :type environ: Environment

    :returns: :class:`True` if this environment is loaded, :class:`False`
        otherwise.
    '''
    is_module_loaded = runtime().modules_system.is_module_loaded
    return (all(map(is_module_loaded, environ.modules)) and
            all(os.environ.get(k, None) == osext.expandvars(v)
                for k, v in environ.env_vars.items()))


def _is_valid_part(part, valid_systems):
    for spec in valid_systems:
        if spec[0] not in ('+', '-', '%'):
            # This is the standard case
            sysname, partname = part.fullname.split(':')
            valid_matches = ['*', '*:*', sysname, f'{sysname}:*',
                             f'*:{partname}', f'{part.fullname}']
            if spec in valid_matches:
                return True
        else:
            plus_feats = []
            minus_feats = []
            props = {}
            for subspec in spec.split(' '):
                if subspec.startswith('+'):
                    plus_feats.append(subspec[1:])
                elif subspec.startswith('-'):
                    minus_feats.append(subspec[1:])
                elif subspec.startswith('%'):
                    key, val = subspec[1:].split('=')
                    props[key] = val

            have_plus_feats = all(
                ft in part.features or ft in part.resources
                for ft in plus_feats
            )
            have_minus_feats = any(
                ft in part.features or ft in part.resources
                for ft in minus_feats
            )
            try:
                have_props = True
                for k, v in props.items():
                    extra_value = part.extras[k]
                    extra_type  = type(extra_value)
                    if extra_value != extra_type(v):
                        have_props = False
                        break
            except (KeyError, ValueError):
                have_props = False

            if have_plus_feats and not have_minus_feats and have_props:
                return True

    return False


def _is_valid_env(env, valid_prog_environs):
    if '*' in valid_prog_environs:
        return True

    for spec in valid_prog_environs:
        if spec[0] not in ('+', '-', '%'):
            # This is the standard case
            if env.name == spec:
                return True
        else:
            plus_feats = []
            minus_feats = []
            props = {}
            for subspec in spec.split(' '):
                if subspec.startswith('+'):
                    plus_feats.append(subspec[1:])
                elif subspec.startswith('-'):
                    minus_feats.append(subspec[1:])
                elif subspec.startswith('%'):
                    key, val = subspec[1:].split('=')
                    props[key] = val

            have_plus_feats = all(ft in env.features for ft in plus_feats)
            have_minus_feats = any(ft in env.features
                                   for ft in minus_feats)
            try:
                have_props = True
                for k, v in props.items():
                    extra_value = env.extras[k]
                    extra_type  = type(extra_value)
                    if extra_value != extra_type(v):
                        have_props = False
                        break
            except (KeyError, ValueError):
                have_props = False

            if have_plus_feats and not have_minus_feats and have_props:
                return True

    return False


def valid_sysenv_comb(valid_systems, valid_prog_environs,
                      check_systems=True, check_environs=True):
    ret = {}
    curr_sys = runtime().system
    for part in curr_sys.partitions:
        if check_systems and not _is_valid_part(part, valid_systems):
            continue

        ret[part] = []
        for env in part.environs:
            if check_environs and not _is_valid_env(env, valid_prog_environs):
                continue

            ret[part].append(env)

    return ret


class temp_environment:
    '''Context manager to temporarily change the environment.'''

    def __init__(self, modules=[], variables=[]):
        self._modules = modules
        self._variables = variables

    def __enter__(self):
        new_env = Environment('_rfm_temp_env', self._modules,
                              self._variables.items())
        self._environ_save, _ = loadenv(new_env)
        return new_env

    def __exit__(self, exc_type, exc_value, traceback):
        self._environ_save.restore()


class temp_config:
    '''Context manager to temporarily switch to specific configuration.'''

    def __init__(self, system):
        self.__to = system
        self.__from = runtime().system.name

    def __enter__(self):
        runtime().site_config.select_subconfig(self.__to)

    def __exit__(self, exc_type, exc_value, traceback):
        runtime().site_config.select_subconfig(self.__from)


# The following utilities are useful only for the unit tests

class temp_runtime:
    '''Context manager to temporarily switch to another runtime.

    :meta private:
    '''

    def __init__(self, config_file, sysname=None, options=None):
        global _runtime_context

        options = options or {}
        self._runtime_save = _runtime_context
        if config_file is None:
            _runtime_context = None
        else:
            site_config = config.load_config(config_file)
            site_config.select_subconfig(sysname, ignore_resolve_errors=True)
            for opt, value in options.items():
                site_config.add_sticky_option(opt, value)

            _runtime_context = RuntimeContext(site_config)

    def __enter__(self):
        return _runtime_context

    def __exit__(self, exc_type, exc_value, traceback):
        global _runtime_context
        _runtime_context = self._runtime_save


def switch_runtime(config_file, sysname=None, options=None):
    '''Function decorator for temporarily changing the runtime for a
    function.

    :meta private:
    '''
    def _runtime_deco(fn):
        @functools.wraps(fn)
        def _fn(*args, **kwargs):
            with temp_runtime(config_file, sysname, options):
                ret = fn(*args, **kwargs)

            return ret

        return _fn

    return _runtime_deco


class module_use:
    '''Context manager for temporarily modifying the module path.'''

    def __init__(self, *paths):
        self._paths = paths

    def __enter__(self):
        runtime().modules_system.searchpath_add(*self._paths)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        runtime().modules_system.searchpath_remove(*self._paths)
