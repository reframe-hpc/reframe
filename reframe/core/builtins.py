# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Regression test class builtins
#

import functools
import reframe.core.parameters as parameters
import reframe.core.variables as variables
import reframe.core.fixtures as fixtures
import reframe.core.hooks as hooks
import reframe.utility as utils
from reframe.core.deferrable import deferrable, _DeferredPerformanceExpression


__all__ = ['deferrable', 'deprecate', 'final', 'fixture', 'loggable',
           'loggable_as', 'parameter', 'performance_function', 'required',
           'require_deps', 'run_before', 'run_after', 'sanity_function',
           'variable']

parameter = parameters.TestParam
variable = variables.TestVar
required = variables.Undefined
deprecate = variables.TestVar.create_deprecated
fixture = fixtures.TestFixture


def final(fn):
    '''Indicate that a function is final and cannot be overridden.'''

    fn._rfm_final = True
    return fn


# Hook-related builtins

def run_before(stage):
    '''Decorator for attaching a test method to a given stage.

    See online docs for more information.
    '''
    return hooks.attach_to('pre_' + stage)


def run_after(stage):
    '''Decorator for attaching a test method to a given stage.

    See online docs for more information.
    '''
    return hooks.attach_to('post_' + stage)


require_deps = hooks.require_deps


# Sanity and performance function builtins

def sanity_function(fn):
    '''Mark a function as the test's sanity function.

    Decorated functions must be unary and they will be converted into
    deferred expressions.
    '''

    _def_fn = deferrable(fn)
    setattr(_def_fn, '_rfm_sanity_fn', True)
    return _def_fn


def performance_function(units, *, perf_key=None):
    '''Decorate a function to extract a performance variable.

    The ``units`` argument indicates the units of the performance
    variable to be extracted.
    The ``perf_key`` optional arg will be used as the name of the
    performance variable. If not provided, the function name will
    be used as the performance variable name.
    '''
    if not isinstance(units, str):
        raise TypeError('performance units must be a string')

    if perf_key and not isinstance(perf_key, str):
        raise TypeError("'perf_key' must be a string")

    def _deco_wrapper(func):
        if not utils.is_trivially_callable(func, non_def_args=1):
            raise TypeError(
                f'performance function {func.__name__!r} has more '
                f'than one argument without a default value'
            )

        @functools.wraps(func)
        def _perf_fn(*args, **kwargs):
            return _DeferredPerformanceExpression(
                func, units, *args, **kwargs
            )

        _perf_key = perf_key if perf_key else func.__name__
        setattr(_perf_fn, '_rfm_perf_key', _perf_key)
        return _perf_fn

    return _deco_wrapper


def loggable_as(name):
    '''Mark a property as loggable.

    :param name: An alternative name that will be used for logging
        this property. If :obj:`None`, the name of the decorated
        property will be used.
    :raises ValueError: if the decorated function is not a property.

    .. versionadded:: 3.10.2

    :meta private:

    '''
    def _loggable(fn):
        if not hasattr(fn, 'fget'):
            raise ValueError('decorated function does not '
                             'look like a property')

        # Mark property as loggable
        #
        # NOTE: Attributes cannot be set on property objects, so we
        # set the attribute on one of its functions
        prop_name = fn.fget.__name__
        fn.fget._rfm_loggable = (prop_name, name)
        return fn

    return _loggable


loggable = loggable_as(None)
