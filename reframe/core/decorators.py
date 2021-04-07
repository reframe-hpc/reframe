# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Decorators used for the definition of tests
#

__all__ = [
    'parameterized_test', 'simple_test', 'required_version',
    'require_deps', 'run_before', 'run_after'
]


import collections
import functools
import inspect
import sys
import traceback

import reframe.utility.osext as osext
from reframe.core.exceptions import (ReframeSyntaxError,
                                     SkipTestError,
                                     user_frame)
from reframe.core.logging import getlogger
from reframe.core.pipeline import RegressionTest
from reframe.utility.versioning import VersionValidator


def _register_test(cls, args=None):
    '''Register the test.

    Register the test with _rfm_use_params=True. This additional argument flags
    this case to consume the parameter space. Otherwise, the regression test
    parameters would simply be initialized to None.
    '''
    def _instantiate(cls, args):
        if isinstance(args, collections.abc.Sequence):
            return cls(*args, _rfm_use_params=True)
        elif isinstance(args, collections.abc.Mapping):
            args['_rfm_use_params'] = True
            return cls(**args)
        elif args is None:
            return cls(_rfm_use_params=True)

    def _instantiate_all():
        ret = []
        for cls, args in mod.__rfm_test_registry:
            try:
                if cls in mod.__rfm_skip_tests:
                    continue

            except AttributeError:
                mod.__rfm_skip_tests = set()

            try:
                ret.append(_instantiate(cls, args))
            except SkipTestError as e:
                getlogger().warning(f'skipping test {cls.__name__!r}: {e}')
            except Exception:
                frame = user_frame(*sys.exc_info())
                getlogger().warning(
                    f"skipping test {cls.__name__!r} due to errors: "
                    f"use `-v' for more information\n"
                    f"    FILE: {frame.filename}:{frame.lineno}"
                )
                getlogger().verbose(traceback.format_exc())

        return ret

    mod = inspect.getmodule(cls)
    if not hasattr(mod, '_rfm_gettests'):
        mod._rfm_gettests = _instantiate_all

    try:
        mod.__rfm_test_registry.append((cls, args))
    except AttributeError:
        mod.__rfm_test_registry = [(cls, args)]


def _validate_test(cls):
    if not issubclass(cls, RegressionTest):
        raise ReframeSyntaxError('the decorated class must be a '
                                 'subclass of RegressionTest')

    if (cls.is_abstract()):
        raise ValueError(f'decorated test ({cls.__qualname__!r}) has one or '
                         f'more undefined parameters')


def simple_test(cls):
    '''Class decorator for registering tests with ReFrame.

    The decorated class must derive from
    :class:`reframe.core.pipeline.RegressionTest`.  This decorator is also
    available directly under the :mod:`reframe` module.

    .. versionadded:: 2.13
    '''
    _validate_test(cls)

    for _ in cls.param_space:
        _register_test(cls)

    return cls


def parameterized_test(*inst):
    '''Class decorator for registering multiple instantiations of a test class.

   The decorated class must derive from
   :class:`reframe.core.pipeline.RegressionTest`. This decorator is also
   available directly under the :mod:`reframe` module.

   :arg inst: The different instantiations of the test. Each instantiation
        argument may be either a sequence or a mapping.

   .. versionadded:: 2.13

   .. note::
      This decorator does not instantiate any test.  It only registers them.
      The actual instantiation happens during the loading phase of the test.
    '''
    def _do_register(cls):
        _validate_test(cls)
        if not cls.param_space.is_empty():
            raise ValueError(
                f'{cls.__qualname__!r} is already a parameterized test'
            )

        for args in inst:
            _register_test(cls, args)

        return cls

    return _do_register


def required_version(*versions):
    '''Class decorator for specifying the required ReFrame versions for the
    following test.

    If the test is not compatible with the current ReFrame version it will be
    skipped.

    :arg versions: A list of ReFrame version specifications that this test is
      allowed to run. A version specification string can have one of the
      following formats:

      1. ``VERSION``: Specifies a single version.
      2. ``{OP}VERSION``, where ``{OP}`` can be any of ``>``, ``>=``, ``<``,
         ``<=``, ``==`` and ``!=``. For example, the version specification
         string ``'>=3.5.0'`` will allow the following test to be loaded only
         by ReFrame 3.5.0 and higher. The ``==VERSION`` specification is the
         equivalent of ``VERSION``.
      3. ``V1..V2``: Specifies a range of versions.

      You can specify multiple versions with this decorator, such as
      ``@required_version('3.5.1', '>=3.5.6')``, in which case the test will be
      selected if *any* of the versions is satisfied, even if the versions
      specifications are conflicting.

    .. versionadded:: 2.13

    .. versionchanged:: 3.5.0

       Passing ReFrame version numbers that do not comply with the `semantic
       versioning <https://semver.org/>`__ specification is deprecated.
       Examples of non-compliant version numbers are ``3.5`` and ``3.5-dev0``.
       These should be written as ``3.5.0`` and ``3.5.0-dev.0``.

    '''
    if not versions:
        raise ValueError('no versions specified')

    conditions = [VersionValidator(v) for v in versions]

    def _skip_tests(cls):
        mod = inspect.getmodule(cls)
        if not hasattr(mod, '__rfm_skip_tests'):
            mod.__rfm_skip_tests = set()

        if not any(c.validate(osext.reframe_version()) for c in conditions):
            getlogger().info('skipping incompatible test defined'
                             ' in class: %s' % cls.__name__)
            mod.__rfm_skip_tests.add(cls)

        return cls

    return _skip_tests


def _runx(phase):
    def deco(func):
        if hasattr(func, '_rfm_attach'):
            func._rfm_attach.append(phase)
        else:
            func._rfm_attach = [phase]

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


# Valid pipeline stages that users can specify in the `run_before()` and
# `run_after()` decorators
_USER_PIPELINE_STAGES = (
    'init', 'setup', 'compile', 'run', 'sanity', 'performance', 'cleanup'
)


def run_before(stage):
    '''Decorator for attaching a test method to a pipeline stage.

    The method will run just before the specified pipeline stage and it should
    not accept any arguments except ``self``.

    This decorator can be stacked, in which case the function will be attached
    to multiple pipeline stages.

    The ``stage`` argument can be any of ``'setup'``, ``'compile'``,
    ``'run'``, ``'sanity'``, ``'performance'`` or ``'cleanup'``.

    '''
    if stage not in _USER_PIPELINE_STAGES:
        raise ValueError(f'invalid pipeline stage specified: {stage!r}')

    if stage == 'init':
        raise ValueError('pre-init hooks are not allowed')

    return _runx('pre_' + stage)


def run_after(stage):
    '''Decorator for attaching a test method to a pipeline stage.

    This is analogous to the :py:attr:`~reframe.core.decorators.run_before`,
    except that ``'init'`` can also be used as the ``stage`` argument. In this
    case, the hook will execute right after the test is initialized (i.e.
    after the :func:`__init__` method is called), before entering the test's
    pipeline. In essence, a post-init hook is equivalent to defining
    additional :func:`__init__` functions in the test. All the other
    properties of pipeline hooks apply equally here. The following code

    .. code-block:: python

       @rfm.run_after('init')
       def foo(self):
           self.x = 1


    is equivalent to

    .. code-block:: python

       def __init__(self):
           self.x = 1

    .. versionchanged:: 3.5.2
       Add the ability to define post-init hooks in tests.

    '''

    if stage not in _USER_PIPELINE_STAGES:
        raise ValueError(f'invalid pipeline stage specified: {stage!r}')

    # Map user stage names to the actual pipeline functions if needed
    if stage == 'init':
        stage = '__init__'
    elif stage == 'compile':
        stage = 'compile_wait'
    elif stage == 'run':
        stage = 'run_wait'

    return _runx('post_' + stage)


def require_deps(func):
    '''Denote that the decorated test method will use the test dependencies.

    The arguments of the decorated function must be named after the
    dependencies that the function intends to use. The decorator will bind the
    arguments to a partial realization of the
    :func:`reframe.core.pipeline.RegressionTest.getdep` function, such that
    conceptually the new function arguments will be the following:

    .. code-block:: python

       new_arg = functools.partial(getdep, orig_arg_name)

    The converted arguments are essentially functions accepting a single
    argument, which is the target test's programming environment.

    Additionally, this decorator will attach the function to run *after* the
    test's setup phase, but *before* any other "post_setup" pipeline hook.

    This decorator is also directly available under the :mod:`reframe` module.

    .. versionadded:: 2.21

    '''
    tests = inspect.getfullargspec(func).args[1:]
    func._rfm_resolve_deps = True

    @functools.wraps(func)
    def _fn(obj, *args):
        newargs = [functools.partial(obj.getdep, t) for t in tests]
        func(obj, *newargs)

    return _fn
