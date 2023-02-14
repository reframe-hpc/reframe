# Copyright 2016-2023 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# unittests/utility.py -- Utilities used in unit tests
#

import functools
import inspect
import os
import sys

import reframe.core.config as config
import reframe.core.modules as modules
import reframe.core.runtime as rt
from reframe.core.meta import make_test


TEST_RESOURCES = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'resources'
)
TEST_RESOURCES_CHECKS = os.path.join(TEST_RESOURCES, 'checks')
TEST_MODULES = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'modules'
)

# Builtin configuration
BUILTIN_CONFIG_FILE = 'reframe/core/settings.py'

# Unit tests site configuration
TEST_CONFIG_FILE = 'unittests/resources/config/settings.py'

# User supplied configuration file and site configuration
USER_CONFIG_FILE = None
USER_SYSTEM = None


def init_runtime():
    site_config = config.load_config('unittests/resources/config/settings.py')
    site_config.select_subconfig('generic')
    rt.init_runtime(site_config)


def partition_by_scheduler(name=None):
    '''Retrieve a system partition from the runtime whose scheduler is
    registered with ``name``.

    If ``name`` is :class:`None`, any partition with a non-local scheduler will
    be returned.
    '''

    system = rt.runtime().system
    for p in system.partitions:
        if name is None and not p.scheduler.is_local:
            return p

        if p.scheduler.registered_name == name:
            return p

    return None


def partition_by_name(name):
    for p in rt.runtime().system.partitions:
        if p.name == name:
            return p

    return None


def environment_by_name(name, partition):
    for e in partition.environs:
        if e.name == name:
            return e

    return None


def has_sane_modules_system():
    return not isinstance(rt.runtime().modules_system.backend,
                          (modules.NoModImpl, modules.SpackImpl))


def custom_prefix(prefix):
    '''Assign a custom prefix to a test.

    This is useful in unit tests when we want to create tests on-the-fly and
    associate them with existing resources.'''

    def _set_prefix(cls):
        cls._rfm_custom_prefix = prefix
        return cls

    return _set_prefix


def dispatch(argname, suffix=None):
    '''Dispatch call to the decorated function to another one based on the type
    of the keyword argument ``argname``.

    The target function has the same name as the decorated one, but without
    the ``test_`` prefix and a suffix ``_{type(kwargs[argname])}`` will be
    appended. If the target function does not exist, the original function
    will be called.

    If the decorated function is a generator, it will be called twice: before
    and after the target function. This allows you to perform actions before
    and after the function call is dispatched.

    :arg argname: The name of the decorated function's keyword argument, based
      on whose type we will dispatch the call.

    :arg suffix: A callable taking a single argument that will produce the
      unique suffix of the function where this call will be dispatched. This
      callable will be passed the value of the ``argname`` argument that was
      passed to the decorated function. If :class:`None` the default suffix
      mentioned above will be used.

    '''

    def _dispatch_deco(fn):
        fnmodule = sys.modules[fn.__module__]

        @functools.wraps(fn)
        def _wrapped(*args, **kwargs):
            target_fn_name = fn.__name__
            if not suffix:
                fn_suffix = f'_{type(kwargs[argname]).__name__}'
            else:
                fn_suffix = '_' + suffix(kwargs[argname])

            if target_fn_name.startswith('test'):
                target_fn_name = target_fn_name[4:]

            if argname not in kwargs:
                raise ValueError(
                    f'cannot dispatch call: no {argname!r} keyword argument '
                    f'was passed to {fn.__name__}()'
                )

            target_fn_name += fn_suffix
            try:
                if inspect.isgeneratorfunction(fn):
                    gen = fn(*args, **kwargs)
                    next(gen)

                fnmodule.__dict__[target_fn_name](*args, **kwargs)
                if inspect.isgeneratorfunction(fn):
                    try:
                        next(gen)
                    except StopIteration:
                        pass

            except KeyError:
                # no target function found; use the original one
                fn(*args, **kwargs)

        return _wrapped

    return _dispatch_deco


def make_check(cls, *, alt_name=None, **vars):
    '''Create a new test from class `cls`.

    :arg cls: the class of the test.
    'arg alt_name: an alternative name to be given to the test class
    :arg vars: variables to set in the test upon creation
    '''

    if alt_name:
        cls = make_test(alt_name, (cls,), {})

    for k, v in vars.items():
        cls.setvar(k, v)

    return cls()
