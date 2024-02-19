# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import functools
import inspect


def is_hook(func):
    return hasattr(func, '_rfm_attach')


def is_dep_hook(func):
    return hasattr(func, '_rfm_resolve_deps')


def attach_to(phase, always_last):
    '''Backend function to attach a hook to a given phase.

    :meta private:
    '''
    def deco(func):
        if is_hook(func):
            func._rfm_attach.append((phase, always_last))
        else:
            func._rfm_attach = [(phase, always_last)]

        try:
            # no need to resolve dependencies independently; this function is
            # already attached to a different phase
            func._rfm_resolve_deps = False
        except AttributeError:
            pass

        @functools.wraps(func)
        def _fn(*args, **kwargs):
            func(*args, **kwargs)

        return _fn

    return deco


def require_deps(func):
    '''Decorator to denote that a function will use the test dependencies.

    The arguments of the decorated function must be named after the
    dependencies that the function intends to use. The decorator will bind the
    arguments to a partial realization of the
    :func:`~reframe.core.pipeline.RegressionTest.getdep` function, such that
    conceptually the new function arguments will be the following:

    .. code-block:: python

       new_arg = functools.partial(getdep, orig_arg_name)

    The converted arguments are essentially functions accepting a single
    argument, which is the target test's programming environment.
    Additionally, this decorator will attach the function to run *after* the
    test's setup phase, but *before* any other "post-setup" pipeline hook.

    .. warning::
       .. versionchanged:: 3.7.0
          Using this functionality from the :py:mod:`reframe` or
          :py:mod:`reframe.core.decorators` modules is now deprecated. You
          should use the built-in function described here.

       .. versionchanged:: 4.0.0
          You may only use this function as framework built-in.

    '''

    tests = inspect.getfullargspec(func).args[1:]
    func._rfm_resolve_deps = True

    @functools.wraps(func)
    def _fn(obj, *args):
        newargs = [functools.partial(obj.getdep, t) for t in tests]
        func(obj, *newargs)

    return _fn


def attach_hooks(hooks):
    '''Attach pipeline hooks to phase ``name''.

    This function returns a decorator for pipeline functions that will run the
    registered hooks before and after the function.

    If ``name'' is :class:`None`, both pre- and post-hooks will run, otherwise
    only the hooks of the phase ``name'' will be executed.
    '''

    def _deco(func):
        def select_hooks(obj, kind):
            phase = kind + func.__name__
            if phase not in hooks:
                return []

            return [h for h in hooks.get(phase, [])
                    if h.__name__ not in getattr(obj, '_disabled_hooks', [])]

        @functools.wraps(func)
        def _fn(obj, *args, **kwargs):
            for h in select_hooks(obj, 'pre_'):
                getattr(obj, h.__name__)()

            func(obj, *args, **kwargs)
            for h in select_hooks(obj, 'post_'):
                getattr(obj, h.__name__)()

        return _fn

    return _deco


class Hook:
    '''A pipeline hook.

    This is essentially a function wrapper that hashes the functions by name,
    since we want hooks to be overriden by name in subclasses.
    '''

    def __init__(self, fn):
        self.__fn = fn
        if not is_hook(fn):
            raise ValueError(f'{fn.__name__} is not a hook')

    @property
    def stages(self):
        return self._rfm_attach
        # return [stage for stage, _ in self._rfm_attach]

    def __getattr__(self, attr):
        return getattr(self.__fn, attr)

    @property
    def fn(self):
        return self.__fn

    def __hash__(self):
        return hash(self.__name__)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        return self.__name__ == other.__name__

    def __call__(self, *args, **kwargs):
        return self.__fn(*args, **kwargs)

    def __repr__(self):
        return repr(self.__fn)


class HookRegistry:
    '''Global hook registry.'''

    def __init__(self, hooks=None):
        self.__hooks = []
        if hooks is not None:
            self.update(hooks)

    def __contains__(self, key):
        return key in self.__hooks

    def __getattr__(self, name):
        return getattr(self.__hooks, name)

    def __iter__(self):
        return iter(self.__hooks)

    def add(self, v):
        '''Add value to the hook registry if it meets the conditions.

        Hook functions have an `_rfm_attach` attribute that specify the stages
        of the pipeline where they must be attached. Dependencies will be
        resolved first in the post-setup phase if not assigned elsewhere.
        '''
        if is_hook(v):
            # Always override hooks with the same name
            h = Hook(v)
            try:
                pos = self.__hooks.index(h)
            except ValueError:
                self.__hooks.append(h)
            else:
                self.__hooks[pos] = h
        elif is_dep_hook(v):
            v._rfm_attach = [('post_setup', None)]
            self.__hooks.append(Hook(v))

    def update(self, other, *, forbidden_names=None):
        '''Update the hook registry with the hooks from another hook
        registry.'''

        assert isinstance(other, HookRegistry)
        forbidden_names = forbidden_names or {}
        for h in other:
            if (h.__name__ in forbidden_names and
                not is_hook(forbidden_names[h.__name__])):
                continue

            try:
                pos = self.__hooks.index(h)
            except ValueError:
                self.__hooks.append(h)
            else:
                self.__hooks[pos] = h

    def __repr__(self):
        return repr(self.__hooks)
