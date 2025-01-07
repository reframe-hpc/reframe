# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause


import importlib
import functools

import reframe.core.fields as fields
from reframe.core.exceptions import ConfigError
from reframe.core.modules import ModulesSystem
from reframe.core.logging import getlogger

_launcher_backend_modules = [
    'reframe.core.launchers.local',
    'reframe.core.launchers.rsh',
    'reframe.core.launchers.mpi'
]
_launchers = {}
_scheduler_backend_modules = [
    'reframe.core.schedulers.local',
    'reframe.core.schedulers.ssh',
    'reframe.core.schedulers.flux',
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


def _detect_backend(backend_type: str):
    backend_modules = globals()[f'_{backend_type}_backend_modules']
    backend_found = []
    for mod in backend_modules:
        importlib.import_module(mod)

    for bcknd in globals()[f'_{backend_type}s']:
        bcknd, _ = globals()[f'_{backend_type}s'][bcknd]
        backend = bcknd.validate()
        if not backend:
            pass
        else:
            backend_found.append((bcknd, backend))
            getlogger().info(f'Found {backend_type}: {backend}')
    if len(backend_found) == 1:
        getlogger().warning(f'No remote {backend_type} detected')
    # By default, select the last one detected
    return backend_found[-1]


register_scheduler = functools.partial(
    _register_backend, backend_type='scheduler'
)
register_launcher = functools.partial(
    _register_backend, backend_type='launcher'
)
getscheduler = functools.partial(_get_backend, backend_type='scheduler')
getlauncher  = functools.partial(_get_backend, backend_type='launcher')
detect_scheduler = functools.partial(_detect_backend, backend_type='scheduler')
detect_launcher = functools.partial(_detect_backend, backend_type='launcher')
# TODO find a better place for this function
detect_modules_system = ModulesSystem.detect
