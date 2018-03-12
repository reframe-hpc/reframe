#
# Core decorators
#

import inspect
import os
import re

from reframe.core.exceptions import ReframeError


_RFM_BASES = tuple()


def abstract_regression_test(cls):
    global _RFM_BASES

    _RFM_BASES += (cls,)
    return cls


def _check_bases(cls):
    """Check if cls is directly derived from one of the framework base
    classes."""

    for base in cls.__bases__:
        if base in _RFM_BASES:
            return True

    return False


def _base_names():
    return (c.__name__ for c in _RFM_BASES)


def _decamelize(s):
    if not s:
        return ''

    return re.sub(r'([a-z])([A-Z])', r'\1_\2', s).lower()


def autoinit_test(cls):
    if not _check_bases(cls):
        raise ReframeError(
            '@autoinit_test decorator can only be used '
            'on direct subclasses of %s' % ', '.join(_base_names()))

    user_init = cls.__init__

    def _rich_init(self, *args, **kwargs):
        super(cls, self).__init__(_decamelize(cls.__name__),
                                  os.path.dirname(inspect.getfile(cls)))

        user_init(self, *args, **kwargs)

    cls.__init__ = _rich_init
    return cls


def _register_test(check):
    def _get_checks(**kwargs):
        return __rfm_checks

    mod = inspect.getmodule(type(cls))
    mod._get_checks = _get_checks
    try:
        mod.__rfm_checks.append(check)
    except NameError:
        mod.__rfm_checks = [check]


def register_singletest(cls):
    _register_test(cls())
    return cls


def register_multitest(inst):
    def _do_register(cls):
        for args in inst:
            _register_test(cls(*args))

        return cls

    return _do_register
