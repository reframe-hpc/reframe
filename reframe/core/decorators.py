#
# Decorators for registering tests with the framework
#

__all__ = ['parameterized_test', 'simple_test', 'required_version']


import collections
import inspect

import reframe
from reframe.core.exceptions import ReframeSyntaxError
from reframe.core.logging import getlogger
from reframe.core.pipeline import RegressionTest
from reframe.utility.versioning import Version, VersionValidator


def _register_test(cls, args=None):
    def _instantiate():
        ret = []
        for cls, args in mod.__rfm_test_registry:
            try:
                if cls in mod.__rfm_skip_tests:
                    continue

            except AttributeError:
                mod.__rfm_skip_tests = set()

            if isinstance(args, collections.Sequence):
                ret.append(cls(*args))
            elif isinstance(args, collections.Mapping):
                ret.append(cls(**args))
            elif args is None:
                ret.append(cls())

        return ret

    mod = inspect.getmodule(cls)
    if not hasattr(mod, '_rfm_gettests'):
        mod._rfm_gettests = _instantiate

    try:
        mod.__rfm_test_registry.append((cls, args))
    except AttributeError:
        mod.__rfm_test_registry = [(cls, args)]


def _validate_test(cls):
    if not issubclass(cls, RegressionTest):
        raise ReframeSyntaxError('the decorated class must be a '
                                 'subclass of RegressionTest')


def simple_test(cls):
    """Class decorator for registering parameterless tests with ReFrame.

    The decorated class must derive from
    :class:`reframe.core.pipeline.RegressionTest`.  This decorator is also
    available directly under the :mod:`reframe` module.

    .. versionadded:: 2.13

    """

    _validate_test(cls)
    _register_test(cls)
    return cls


def parameterized_test(*inst):
    """Class decorator for registering multiple instantiations of a test class.

   The decorated class must derive from
   :class:`reframe.core.pipeline.RegressionTest`. This decorator is also
   available directly under the :mod:`reframe` module.

   :arg inst: The different instantiations of the test. Each instantiation
        argument may be either a sequence or a mapping.

   .. versionadded:: 2.13

   .. note::

      This decorator does not instantiate any test.  It only registers them.
      The actual instantiation happens during the loading phase of the test.

    """
    def _do_register(cls):
        _validate_test(cls)
        for args in inst:
            _register_test(cls, args)

        return cls

    return _do_register


def required_version(*versions):
    """Class decorator for skipping version-uncompatible tests.

    The decorated class must derive from
    :class:`reframe.core.pipeline.RegressionTest`. This decorator is also
    available directly under the :mod:`reframe` module.

    :arg versions: The versions that are compatible with the test.

    .. versionadded:: 2.13

    """
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
