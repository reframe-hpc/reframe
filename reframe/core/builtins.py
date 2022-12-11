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
    '''Attach the decorated function before a certain pipeline stage.

    The function will run just before the specified pipeline stage and it
    cannot accept any arguments except ``self``. This decorator can be
    stacked, in which case the function will be attached to multiple pipeline
    stages. See above for the valid ``stage`` argument values.

    :param stage: The pipeline stage where this function will be attached to.
        See :ref:`pipeline-hooks` for the list of valid stage values.
    '''
    return hooks.attach_to('pre_' + stage)


def run_after(stage):
    '''Attach the decorated function after a certain pipeline stage.

    This is analogous to :func:`~RegressionMixin.run_before`, except that the
    hook will execute right after the stage it was attached to. This decorator
    also supports ``'init'`` as a valid ``stage`` argument, where in this
    case, the hook will execute right after the test is initialized (i.e.
    after the :func:`__init__` method is called) and before entering the
    test's pipeline. In essence, a post-init hook is equivalent to defining
    additional :func:`__init__` functions in the test. The following code

    .. code-block:: python

       class MyTest(rfm.RegressionTest):
           @run_after('init')
           def foo(self):
               self.x = 1

    is equivalent to

    .. code-block:: python

       class MyTest(rfm.RegressionTest):
           def __init__(self):
               self.x = 1

    .. versionchanged:: 3.5.2
       Add support for post-init hooks.

    '''
    return hooks.attach_to('post_' + stage)


require_deps = hooks.require_deps


# Sanity and performance function builtins

def sanity_function(fn):
    '''Decorate a test member function to mark it as a sanity check.

    This decorator will convert the given function into a
    :func:`~RegressionMixin.deferrable` and mark it to be executed during the
    test's sanity stage. When this decorator is used, manually assigning a
    value to :attr:`~RegressionTest.sanity_patterns` in the test is not
    allowed.

    Decorated functions may be overridden by derived classes, and derived
    classes may also decorate a different method as the test's sanity
    function. Decorating multiple member functions in the same class is not
    allowed. However, a :class:`RegressionTest` may inherit from multiple
    :class:`RegressionMixin` classes with their own sanity functions. In this
    case, the derived class will follow Python's `MRO
    <https://docs.python.org/3/library/stdtypes.html#class.__mro__>`_ to find
    a suitable sanity function.

    .. versionadded:: 3.7.0
    '''

    _def_fn = deferrable(fn)
    setattr(_def_fn, '_rfm_sanity_fn', True)
    return _def_fn


def performance_function(unit, *, perf_key=None):
    '''Decorate a test member function to mark it as a performance metric
    function.

    This decorator converts the decorated method into a performance deferrable
    function (see ":ref:`deferrable-performance-functions`" for more details)
    whose evaluation is deferred to the performance stage of the regression
    test. The decorated function must take a single argument without a default
    value (i.e. ``self``) and any number of arguments with default values. A
    test may decorate multiple member functions as performance functions,
    where each of the decorated functions must be provided with the unit of
    the performance quantity to be extracted from the test. Any performance
    function may be overridden in a derived class and multiple bases may
    define their own performance functions. In the event of a name conflict,
    the derived class will follow Python's `MRO
    <https://docs.python.org/3/library/stdtypes.html#class.__mro__>`_ to
    choose the appropriate performance function. However, defining more than
    one performance function with the same name in the same class is
    disallowed.

    The full set of performance functions of a regression test is stored under
    :attr:`~reframe.core.pipeline.RegressionTest.perf_variables` as key-value
    pairs, where, by default, the key is the name of the decorated member
    function, and the value is the deferred performance function itself.
    Optionally, the key under which a performance function is stored in
    :attr:`~reframe.core.pipeline.RegressionTest.perf_variables` can be
    customised by passing the desired key as the ``perf_key`` argument to this
    decorator.

    :param unit: A string representing the measurement unit of this metric.

    .. versionadded:: 3.8.0

    '''

    if not isinstance(unit, str):
        raise TypeError('performance unit must be a string')

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
                func, unit, *args, **kwargs
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
loggable.__doc__ = '''Equivalent to :func:`loggable_as(None) <loggable_as>`.'''
