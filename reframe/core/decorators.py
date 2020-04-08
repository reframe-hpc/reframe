# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
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

import reframe
from reframe.core.exceptions import ReframeSyntaxError, user_frame
from reframe.core.logging import getlogger
from reframe.core.pipeline import RegressionTest
from reframe.utility.versioning import VersionValidator


def _register_test(cls, args=None):
    def _instantiate(cls, args):
        if isinstance(args, collections.abc.Sequence):
            return cls(*args)
        elif isinstance(args, collections.abc.Mapping):
            return cls(**args)
        elif args is None:
            return cls()

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
            except Exception:
                frame = user_frame(sys.exc_info()[2])
                msg = "skipping test due to errors: %s: " % cls.__name__
                msg += "use `-v' for more information\n"
                msg += "  FILE: %s:%s" % (frame.filename, frame.lineno)
                getlogger().warning(msg)
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


def simple_test(cls):
    '''Class decorator for registering parameterless tests with ReFrame.

    The decorated class must derive from
    :class:`reframe.core.pipeline.RegressionTest`.  This decorator is also
    available directly under the :mod:`reframe` module.

    .. versionadded:: 2.13

    '''

    _validate_test(cls)
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
      ``<=``, ``==`` and ``!=``. For example, the version specification string
      ``'>=2.15'`` will only allow the following test to be loaded only by
      ReFrame 2.15 and higher. The ``==VERSION`` specification is the
      equivalent of ``VERSION``.

      3. ``V1..V2``: Specifies a range of versions.

      You can specify multiple versions with this decorator, such as
      ``@required_version('2.13', '>=2.16')``, in which case the test will be
      selected if *any* of the versions is satisfied, even if the versions
      specifications are conflicting.

    .. versionadded:: 2.13

    '''
    if not versions:
        raise ValueError('no versions specified')

    conditions = [VersionValidator(v) for v in versions]

    def _skip_tests(cls):
        mod = inspect.getmodule(cls)
        if not hasattr(mod, '__rfm_skip_tests'):
            mod.__rfm_skip_tests = set()

        if not any(c.validate(reframe.VERSION) for c in conditions):
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


def run_before(stage):
    '''Run the decorated function before the specified pipeline stage.

    The decorated function must be a method of a regression test.

    .. versionadded:: 2.20

    '''
    return _runx('pre_' + stage)


def run_after(stage):
    '''Run the decorated function after the specified pipeline stage.

    The decorated function must be a method of a regression test.

    .. versionadded:: 2.20

    '''
    return _runx('post_' + stage)


def require_deps(func):
    '''Denote that the decorated test method will use the test dependencies.

    The arguments of the decorated function must be named after the
    dependencies that the function intends to use. The decorator will bind the
    arguments to a partial realization of the
    :func:`reframe.core.pipeline.RegressionTest.getdep` function, such that
    conceptually the new function arguments will be the following:

    .. code:: python

       new_arg = functools.partial(getdep, orig_arg_name)

    The converted arguments are essentially functions accepting a single
    argument, which is the target test's programming environment.

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
