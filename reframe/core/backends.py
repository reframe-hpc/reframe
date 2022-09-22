# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause


import importlib
import functools

import reframe.core.fields as fields
from reframe.core.exceptions import ConfigError

_launcher_backend_modules = [
    'reframe.core.launchers.local',
    'reframe.core.launchers.mpi',
    'reframe.core.launchers.ssh'
]
_launchers = {}
_scheduler_backend_modules = [
    'reframe.core.schedulers.flux',
    'reframe.core.schedulers.local',
    'reframe.core.schedulers.lsf',
    'reframe.core.schedulers.pbs',
    'reframe.core.schedulers.oar',
    'reframe.core.schedulers.sge',
    'reframe.core.schedulers.slurm'
]
_schedulers = {}


def _register_backend(name, local=False, error=None, *, backend_type):
    def do_register(cls):
        registry = globals()[f'_{backend_type}s']
        if name in registry:
            raise ConfigError(
                f"'{name}' is already registered as a {backend_type}"
            )

        cls.is_local = fields.ConstantField(bool(local))
        cls.registered_name = fields.ConstantField(name)
        registry[name] = (cls, error)
        return cls

    return do_register


def _get_backend(name, *, backend_type):
    backend_modules = globals()[f'_{backend_type}_backend_modules']
    for mod in backend_modules:
        importlib.import_module(mod)

    try:
        cls, error = globals()[f'_{backend_type}s'][name]
        if error:
            raise ConfigError(
                f'could not register {backend_type} backend: {error}'
            )
    except KeyError:
        raise ConfigError(f'no such {backend_type}: {name!r}')
    else:
        return cls


register_scheduler = functools.partial(
    _register_backend, backend_type='scheduler'
)
register_launcher = functools.partial(
    _register_backend, backend_type='launcher'
)
getscheduler = functools.partial(_get_backend, backend_type='scheduler')
getlauncher  = functools.partial(_get_backend, backend_type='launcher')
