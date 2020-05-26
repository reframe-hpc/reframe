# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import builtins
import glob as pyglob
import itertools
import re
import sys

import reframe.utility as util
from reframe.core.deferrable import deferrable, _DeferredExpression
from reframe.core.exceptions import SanityError


def _format(s, *args, **kwargs):
    '''Safely format string ``s``.

    Returns ``s.format(*args, **kwargs)`` if no exception is thrown, otherwise
    ``s``.
    '''
    try:
        return s.format(*args, **kwargs)
    except (IndexError, KeyError):
        return s


# Create an alias decorator
sanity_function = deferrable


# Deferrable versions of selected builtins

@deferrable
def abs(x):
    '''Replacement for the built-in :func:`abs() <python:abs>` function.'''
    return builtins.abs(x)


@deferrable
def all(iterable):
    '''Replacement for the built-in :func:`all() <python:all>` function.'''
    return builtins.all(iterable)


@deferrable
def any(iterable):
    '''Replacement for the built-in :func:`any() <python:any>` function.'''
    return builtins.any(iterable)


@deferrable
def chain(*iterables):
    '''Replacement for the :func:`itertools.chain() <python:itertools.chain>`
    function.'''
    return itertools.chain(*iterables)


@deferrable
def enumerate(iterable, start=0):
    '''Replacement for the built-in
    :func:`enumerate() <python:enumerate>` function.'''
    return builtins.enumerate(iterable, start)


@deferrable
def filter(function, iterable):
    '''Replacement for the built-in
    :func:`filter() <python:filter>` function.'''
    return builtins.filter(function, iterable)


@deferrable
def getattr(obj, attr, *args):
    '''Replacement for the built-in
    :func:`getattr() <python:getattr>` function.'''
    return builtins.getattr(obj, attr, *args)


@deferrable
def hasattr(obj, name):
    '''Replacement for the built-in
    :func:`hasattr() <python:hasattr>` function.'''
    return builtins.hasattr(obj, name)


@deferrable
def len(s):
    '''Replacement for the built-in :func:`len() <python:len>` function.'''
    return builtins.len(s)


@deferrable
def map(function, *iterables):
    '''Replacement for the built-in :func:`map() <python:map>` function.'''
    return builtins.map(function, *iterables)


@deferrable
def max(*args):
    '''Replacement for the built-in :func:`max() <python:max>` function.'''
    return builtins.max(*args)


@deferrable
def min(*args):
    '''Replacement for the built-in :func:`min() <python:min>` function.'''
    return builtins.min(*args)


@deferrable
def print(*objects, sep=' ', end='\n', file=None, flush=False):
    '''Replacement for the built-in :func:`print() <python:print>` function.

    The only difference is that this function returns the ``objects``, so that
    you can use it transparently inside a complex sanity expression. For
    example, you could write the following to print the matches returned from
    the :func:`extractall()` function:

    .. code:: python

        self.sanity_patterns = sn.assert_eq(
            sn.count(sn.print(sn.extract_all(...))), 10
        )

    If ``file`` is None, :func:`print` will print its arguments to the
    standard output. Unlike the builtin :func:`print() <python:print>`
    function, we don't bind the ``file`` argument to :attr:`sys.stdout` by
    default. This would capture :attr:`sys.stdout` at the time this function
    is defined and would prevent it from seeing changes to :attr:`sys.stdout`,
    such as redirects, in the future.
    '''

    if file is None:
        file = sys.stdout

    builtins.print(*objects, sep=sep, end=end, file=file, flush=flush)
    return objects


@deferrable
def reversed(seq):
    '''Replacement for the built-in
    :func:`reversed() <python:reversed>` function.'''
    return builtins.reversed(seq)


@deferrable
def round(number, *args):
    '''Replacement for the built-in
    :func:`round() <python:round>` function.'''
    return builtins.round(number, *args)


@deferrable
def setattr(obj, name, value):
    '''Replacement for the built-in
    :func:`setattr() <python:setattr>` function.'''
    builtins.setattr(obj, name, value)


@deferrable
def sorted(iterable, *args):
    '''Replacement for the built-in
    :func:`sorted() <python:sorted>` function.'''
    return builtins.sorted(iterable, *args)


@deferrable
def sum(iterable, *args):
    '''Replacement for the built-in :func:`sum() <python:sum>` function.'''
    return builtins.sum(iterable, *args)


@deferrable
def zip(*iterables):
    '''Replacement for the built-in :func:`zip() <python:zip>` function.'''
    return builtins.zip(*iterables)


# Alternatives for non-overridable operators

@deferrable
def and_(a, b):
    '''Deferrable version of the :keyword:`and` operator.

    :returns: ``a and b``.'''
    return builtins.all([a, b])


@deferrable
def or_(a, b):
    '''Deferrable version of the :keyword:`or` operator.

    :returns: ``a or b``.'''
    return builtins.any([a, b])


@deferrable
def not_(a):
    '''Deferrable version of the :keyword:`not` operator.

    :returns: ``not a``.'''
    return not a


@deferrable
def contains(seq, key):
    '''Deferrable version of the :keyword:`in` operator.

    :returns: ``key in seq``.'''
    return key in seq


# Deferrable assert functions

@deferrable
def assert_true(x, msg=None):
    '''Assert that ``x`` is evaluated to ``True``.

    :returns: ``True`` on success.
    :raises reframe.core.exceptions.SanityError: if assertion fails.
    '''
    if builtins.bool(x) is not True:
        error_msg = msg or '{0} is not True'
        raise SanityError(_format(error_msg, x))

    return True


@deferrable
def assert_false(x, msg=None):
    '''Assert that ``x`` is evaluated to ``False``.

    :returns: ``True`` on success.
    :raises reframe.core.exceptions.SanityError: if assertion fails.
    '''
    if builtins.bool(x) is not False:
        error_msg = msg or '{0} is not False'
        raise SanityError(_format(error_msg, x))

    return True


@deferrable
def assert_eq(a, b, msg=None):
    '''Assert that ``a == b``.

    :returns: ``True`` on success.
    :raises reframe.core.exceptions.SanityError: if assertion fails.
    '''
    if a != b:
        error_msg = msg or '{0} != {1}'
        raise SanityError(_format(error_msg, a, b))

    return True


@deferrable
def assert_ne(a, b, msg=None):
    '''Assert that ``a != b``.

    :returns: ``True`` on success.
    :raises reframe.core.exceptions.SanityError: if assertion fails.
    '''
    if a == b:
        error_msg = msg or '{0} == {1}'
        raise SanityError(_format(error_msg, a, b))

    return True


@deferrable
def assert_in(item, container, msg=None):
    '''Assert that ``item`` is in ``container``.

    :returns: ``True`` on success.
    :raises reframe.core.exceptions.SanityError: if assertion fails.
    '''
    if item not in container:
        error_msg = msg or '{0} is not in {1}'
        raise SanityError(_format(error_msg, item, container))

    return True


@deferrable
def assert_not_in(item, container, msg=None):
    '''Assert that ``item`` is not in ``container``.

    :returns: ``True`` on success.
    :raises reframe.core.exceptions.SanityError: if assertion fails.
    '''
    if item in container:
        error_msg = msg or '{0} is in {1}'
        raise SanityError(_format(error_msg, item, container))

    return True


@deferrable
def assert_gt(a, b, msg=None):
    '''Assert that ``a > b``.

    :returns: ``True`` on success.
    :raises reframe.core.exceptions.SanityError: if assertion fails.
    '''
    if a <= b:
        error_msg = msg or '{0} <= {1}'
        raise SanityError(_format(error_msg, a, b))

    return True


@deferrable
def assert_ge(a, b, msg=None):
    '''Assert that ``a >= b``.

    :returns: ``True`` on success.
    :raises reframe.core.exceptions.SanityError: if assertion fails.
    '''
    if a < b:
        error_msg = msg or '{0} < {1}'
        raise SanityError(_format(error_msg, a, b))

    return True


@deferrable
def assert_lt(a, b, msg=None):
    '''Assert that ``a < b``.

    :returns: ``True`` on success.
    :raises reframe.core.exceptions.SanityError: if assertion fails.
    '''
    if a >= b:
        error_msg = msg or '{0} >= {1}'
        raise SanityError(_format(error_msg, a, b))

    return True


@deferrable
def assert_le(a, b, msg=None):
    '''Assert that ``a <= b``.

    :returns: ``True`` on success.
    :raises reframe.core.exceptions.SanityError: if assertion fails.
    '''
    if a > b:
        error_msg = msg or '{0} > {1}'
        raise SanityError(_format(error_msg, a, b))

    return True


@deferrable
def assert_found(patt, filename, msg=None, encoding='utf-8'):
    '''Assert that regex pattern ``patt`` is found in the file ``filename``.

    :arg patt: The regex pattern to search.
        Any standard Python `regular expression
        <https://docs.python.org/3/library/re.html#regular-expression-syntax>`_
        is accepted.
    :arg filename: The name of the file to examine.
        Any :class:`OSError` raised while processing the file will be
        propagated as a :class:`reframe.core.exceptions.SanityError`.
    :arg encoding: The name of the encoding used to decode the file.
    :returns: ``True`` on success.
    :raises reframe.core.exceptions.SanityError: if assertion fails.
    '''
    num_matches = count(finditer(patt, filename, encoding))
    try:
        evaluate(assert_true(num_matches))
    except SanityError:
        error_msg = msg or "pattern `{0}' not found in `{1}'"
        raise SanityError(_format(error_msg, patt, filename))
    else:
        return True


@deferrable
def assert_not_found(patt, filename, msg=None, encoding='utf-8'):
    '''Assert that regex pattern ``patt`` is not found in the file
    ``filename``.

    This is the inverse of :func:`assert_found()`.

    :returns: ``True`` on success.
    :raises reframe.core.exceptions.SanityError: if assertion fails.
    '''
    try:
        evaluate(assert_found(patt, filename, msg, encoding))
    except SanityError:
        return True
    else:
        error_msg = msg or "pattern `{0}' found in `{1}'"
        raise SanityError(_format(error_msg, patt, filename))


@deferrable
def assert_bounded(val, lower=None, upper=None, msg=None):
    '''Assert that ``lower <= val <= upper``.

    :arg val: The value to check.
    :arg lower: The lower bound. If ``None``, it defaults to ``-inf``.
    :arg upper: The upper bound. If ``None``, it defaults to ``inf``.
    :returns: ``True`` on success.
    :raises reframe.core.exceptions.SanityError: if assertion fails.
    '''
    if lower is None:
        lower = builtins.float('-inf')

    if upper is None:
        upper = builtins.float('inf')

    if val >= lower and val <= upper:
        return True

    error_msg = msg or 'value {0} not within bounds {1}..{2}'
    raise SanityError(_format(error_msg, val, lower, upper))


@deferrable
def assert_reference(val, ref, lower_thres=None, upper_thres=None, msg=None):
    '''Assert that value ``val`` respects the reference value ``ref``.

    :arg val: The value to check.
    :arg ref: The reference value.
    :arg lower_thres: The lower threshold value expressed as a negative decimal
        fraction of the reference value.  Must be in [-1, 0] for ref >= 0.0 and
        in [-inf, 0] for ref < 0.0.
        If ``None``, no lower thresholds is applied.
    :arg upper_thres: The upper threshold value expressed as a decimal fraction
        of the reference value. Must be in [0, inf] for ref >= 0.0 and
        in [0, 1] for ref < 0.0.
        If ``None``, no upper thresholds is applied.
    :returns: ``True`` on success.
    :raises reframe.core.exceptions.SanityError: if assertion fails or if the
        lower and upper thresholds do not have appropriate values.
    '''
    if lower_thres is not None:
        lower_thres_limit = -1 if ref >= 0 else None
        try:
            evaluate(assert_bounded(lower_thres, lower_thres_limit, 0))
        except SanityError:
            raise SanityError('invalid low threshold value: %s' %
                              lower_thres) from None

    if upper_thres is not None:
        upper_thres_limit = None if ref >= 0 else 1
        try:
            evaluate(assert_bounded(upper_thres, 0, upper_thres_limit))
        except SanityError:
            raise SanityError('invalid high threshold value: %s' %
                              upper_thres) from None

    def calc_bound(thres):
        if thres is None:
            return None

        # Inverse threshold if ref < 0
        if ref < 0:
            thres = -thres

        return ref*(1 + thres)

    lower = calc_bound(lower_thres) or float('-inf')
    upper = calc_bound(upper_thres) or float('inf')
    try:
        evaluate(assert_bounded(val, lower, upper))
    except SanityError:
        error_msg = msg or '{0} is beyond reference value {1} (l={2}, u={3})'
        raise SanityError(_format(error_msg, val, ref, lower, upper)) from None
    else:
        return True


# Pattern matching functions

@deferrable
def finditer(patt, filename, encoding='utf-8'):
    '''Get an iterator over the matches of the regex ``patt`` in ``filename``.

    This function is equivalent to :func:`findall()` except that it returns
    a generator object instead of a list, which you can use to iterate over
    the raw matches.
    '''
    try:
        with open(filename, 'rt', encoding=encoding) as fp:
            yield from re.finditer(patt, fp.read(), re.MULTILINE)
    except OSError as e:
        # Re-raise it as sanity error
        raise SanityError('%s: %s' % (filename, e.strerror))


@deferrable
def findall(patt, filename, encoding='utf-8'):
    '''Get all matches of regex ``patt`` in ``filename``.

    :arg patt: The regex pattern to search.
        Any standard Python `regular expression
        <https://docs.python.org/3/library/re.html#regular-expression-syntax>`_
        is accepted.
    :arg filename: The name of the file to examine.
    :arg encoding: The name of the encoding used to decode the file.
    :returns: A list of raw `regex match objects
        <https://docs.python.org/3/library/re.html#match-objects>`_.
    :raises reframe.core.exceptions.SanityError: In case an :class:`OSError` is
        raised while processing ``filename``.
    '''
    return list(evaluate(x) for x in finditer(patt, filename, encoding))


@deferrable
def extractiter(patt, filename, tag=0, conv=None, encoding='utf-8'):
    '''Get an iterator over the values extracted from the capturing group
    ``tag`` of a matching regex ``patt`` in the file ``filename``.

    This function is equivalent to :func:`extractall` except that it returns
    a generator object, instead of a list, which you can use to iterate over
    the extracted values.
    '''
    for m in finditer(patt, filename, encoding):
        try:
            val = m.group(tag)
        except (IndexError, KeyError):
            raise SanityError(
                "no such group in pattern `%s': %s" % (patt, tag))

        try:
            yield conv(val) if callable(conv) else val
        except ValueError:
            fn_name = '<unknown>'
            try:
                # Assume conv is standard function
                fn_name = conv.__name__
            except AttributeError:
                try:
                    # Assume conv is callable object
                    fn_name = conv.__class__.__name__
                except AttributeError:
                    pass

            raise SanityError("could not convert value `%s' using `%s()'" %
                              (val, fn_name))


@deferrable
def extractall(patt, filename, tag=0, conv=None, encoding='utf-8'):
    '''Extract all values from the capturing group ``tag`` of a matching regex
    ``patt`` in the file ``filename``.

    :arg patt: The regex pattern to search.
        Any standard Python `regular expression
        <https://docs.python.org/3/library/re.html#regular-expression-syntax>`_
        is accepted.
    :arg filename: The name of the file to examine.
    :arg encoding: The name of the encoding used to decode the file.
    :arg tag: The regex capturing group to be extracted.
        Group ``0`` refers always to the whole match.
        Since the file is processed line by line, this means that group ``0``
        returns the whole line that was matched.
    :arg conv: A callable that takes a single argument and returns a new value.
        If provided, it will be used to convert the extracted values before
        returning them.
    :returns: A list of the extracted values from the matched regex.
    :raises reframe.core.exceptions.SanityError: In case of errors.
    '''
    return list(evaluate(x)
                for x in extractiter(patt, filename, tag, conv, encoding))


@deferrable
def extractsingle(patt, filename, tag=0, conv=None, item=0, encoding='utf-8'):
    '''Extract a single value from the capturing group ``tag`` of a matching
    regex ``patt`` in the file ``filename``.

    This function is equivalent to ``extractall(patt, filename, tag,
    conv)[item]``, except that it raises a ``SanityError`` if ``item`` is out
    of bounds.

    :arg patt: as in :func:`extractall`.
    :arg filename: as in :func:`extractall`.
    :arg encoding: as in :func:`extractall`.
    :arg tag: as in :func:`extractall`.
    :arg conv: as in :func:`extractall`.
    :arg item: the specific element to extract.
    :returns: The extracted value.
    :raises reframe.core.exceptions.SanityError: In case of errors.

    '''
    try:
        # Explicitly evaluate the expression here, so as to force any exception
        # to be thrown in this context and not during the evaluation of an
        # expression containing this one.
        return evaluate(extractall(patt, filename, tag, conv, encoding)[item])
    except IndexError:
        raise SanityError(
            "not enough matches of pattern `%s' in file `%s' "
            "so as to extract item `%s'" % (patt, filename, item)
        )


# Numeric functions

@deferrable
def avg(iterable):
    '''Return the average of all the elements of ``iterable``.'''

    # We walk over the iterable manually in case this is a generator
    total = 0
    num_vals = None
    for num_vals, val in builtins.enumerate(iterable, start=1):
        total += val

    if num_vals is None:
        raise SanityError('attempt to get average on an empty container')

    return total / num_vals


# Other utility functions

@deferrable
def allx(iterable):
    '''Same as the built-in :func:`all() <python:all>` function, except that it
    returns :class:`False` if ``iterable`` is empty.

    .. versionadded:: 2.13
    '''
    return util.allx(iterable)


@deferrable
def defer(x):
    '''Defer the evaluation of variable ``x``.

    .. versionadded:: 2.21
    '''
    return x


def evaluate(expr):
    '''Evaluate a deferred expression.

    If ``expr`` is not a deferred expression, it will be returned as is.

    .. versionadded:: 2.21
    '''
    if isinstance(expr, _DeferredExpression):
        return expr.evaluate()
    else:
        return expr


@deferrable
def getitem(container, item):
    '''Get ``item`` from ``container``.

    ``container`` may refer to any container that can be indexed.

    :raises reframe.core.exceptions.SanityError: In case ``item`` cannot be
        retrieved from ``container``.
    '''
    try:
        return container[item]
    except KeyError:
        raise SanityError('key not found: %s' % item)
    except IndexError:
        raise SanityError('index out of bounds: %s' % item)


@deferrable
def count(iterable):
    '''Return the element count of ``iterable``.

    This is similar to the built-in :func:`len() <python:len>`, except that it
    can also handle any argument that supports iteration, including
    generators.
    '''
    try:
        return builtins.len(iterable)
    except TypeError:
        # Try to determine length by iterating over the iterable
        ret = 0
        for ret, _ in builtins.enumerate(iterable, start=1):
            pass

        return ret


@deferrable
def count_uniq(iterable):
    '''Return the unique element count of ``iterable``.'''
    return builtins.len(builtins.set(iterable))


@deferrable
def glob(pathname, *, recursive=False):
    '''Replacement for the :func:`glob.glob() <python:glob.glob>` function.'''
    return pyglob.glob(pathname, recursive=recursive)


@deferrable
def iglob(pathname, recursive=False):
    '''Replacement for the :func:`glob.iglob() <python:glob.iglob>`
    function.'''
    return pyglob.iglob(pathname, recursive=recursive)
