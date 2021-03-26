# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import functools

import reframe.utility as util


def _attach_hooks(hooks, name=None):
    def _deco(func):
        def select_hooks(obj, kind):
            phase = name
            if phase is None:
                fn_name = func.__name__
                if fn_name == '__init__':
                    fn_name = 'init'

                phase = kind + fn_name
            elif phase is not None and not phase.startswith(kind):
                # Just any phase that does not exist
                phase = 'xxx'

            if phase not in hooks:
                return []

            return [h for h in hooks[phase]
                    if h.name not in obj._disabled_hooks]

        @functools.wraps(func)
        def _fn(obj, *args, **kwargs):
            for h in select_hooks(obj, 'pre_'):
                h(obj)

            func(obj, *args, **kwargs)
            for h in select_hooks(obj, 'post_'):
                h(obj)

        return _fn

    return _deco


class Hook:
    def __init__(self, fn):
        self.__fn = fn

    @property
    def fn(self):
        return self.__fn

    @property
    def name(self):
        return self.__fn.__name__

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        return self.name == other.name

    def __call__(self, *args, **kwargs):
        return self.__fn(*args, **kwargs)

    def __repr__(self):
        return repr(self.__fn)


class HookRegistry:
    '''Global hook registry.'''

    def __init__(self, hooks=None):
        self.__hooks = {}
        if hooks is not None:
            self.update(hooks)

    def __getitem__(self, key):
        return self.__hooks[key]

    def __setitem__(self, key, name):
        self.__hooks[key] = name

    def __getattr__(self, name):
        return getattr(self.__hooks, name)

    def update(self, hooks):
        for phase, hks in hooks.items():
            self.__hooks.setdefault(phase, util.OrderedSet())
            for h in hks:
                self.__hooks[phase].add(h)

    def apply(self, obj):
        cls = type(obj)
        cls.__init__ = _attach_hooks(self.__hooks)(cls.__init__)
        cls.setup = _attach_hooks(self.__hooks)(cls.setup)
        cls.compile = _attach_hooks(self.__hooks, 'pre_compile')(cls.compile)
        cls.compile_wait = _attach_hooks(self.__hooks, 'post_compile')(
            cls.compile_wait
        )
        cls.run = _attach_hooks(self.__hooks, 'pre_run')(cls.run)
        cls.run_wait = _attach_hooks(self.__hooks, 'post_run')(cls.run_wait)
        cls.sanity = _attach_hooks(self.__hooks)(cls.sanity)
        cls.performance = _attach_hooks(self.__hooks)(cls.performance)
        cls.cleanup = _attach_hooks(self.__hooks)(cls.cleanup)
