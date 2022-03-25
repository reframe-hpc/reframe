# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import builtins
import functools


def deferrable(func):
    '''Convert the decorated function to a deferred expression.

    See :ref:`deferrable-functions` for further information on deferrable
    functions.
    '''

    @functools.wraps(func)
    def _deferred(*args, **kwargs):
        return _DeferredExpression(func, *args, **kwargs)

    return _deferred


class _DeferredExpression:
    '''Represents an expression whose evaluation has been deferred.

    This class simply stores a callable and its arguments and will evaluate it
    as soon as the `evaluate()` method is called. This class implements the
    basic unary and binary operators of Python so that it makes it possible to
    defer also arbitrary expressions. Note the `not`, `and` and `or` operators
    cannot be overloaded. If you want to defer an expression containing such
    operators, you should use the provided `and_`, `or_` or `not_` deferred
    functions.

    Deferred expressions may by chained together. Chaining happens
    automatically if an argument to a function or an operand of an operator are
    deferred expressions, too. When you later evaluate the outermost
    expression, the evaluation will go down the chain of deferred expressions
    and evaluate them all.

    `_DeferredExpression` are immutable objects.
    '''

    def __init__(self, fn, *args, **kwargs):
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

        # We cache the value of the last evaluation inside a tuple.
        # We don't cache the value directly, because it can be any.
        self._cached = ()
        self._return_cached = False

    def evaluate(self, cache=False):
        # Return the cached value (if any)
        if self._return_cached and not cache:
            return self._cached[0]
        elif cache:
            self._return_cached = cache

        fn_args = []
        for arg in self._args:
            fn_args.append(
                arg.evaluate() if isinstance(arg, _DeferredExpression) else arg
            )

        fn_kwargs = {}
        for k, v in self._kwargs.items():
            fn_kwargs[k] = (
                v.evaluate() if isinstance(v, _DeferredExpression) else v
            )

        ret = self._fn(*fn_args, **fn_kwargs)

        # Evaluate the return for as long as a deferred expression returns
        # another deferred expression.
        while isinstance(ret, _DeferredExpression):
            ret = ret.evaluate()

        # Cache the results for any subsequent evaluate calls.
        self._cached = (ret,)
        return ret

    def __bool__(self):
        '''The truthy value of a deferred expression.

        This causes the immediate evaluation of the deferred expression.
        '''
        return builtins.bool(self.evaluate())

    def __str__(self):
        '''Evaluate the deferred expresion and return its string
        representation.'''
        return str(self.evaluate())

    def __iter__(self):
        '''Evaluate the deferred expression and iterate over the result.'''
        return iter(self.evaluate())

    def __rfm_json_encode__(self):
        if self._cached == ():
            return None
        else:
            return self._cached[0]

    # Overload Python operators to be able to defer any expression
    #
    # NOTE: In the following we are not using `self` for denoting the first
    # argument. These operators are just there to capture and defer the
    # corresponding expression. When we evaluate this deferred expression, they
    # will be called with the real arguments of the originally captured
    # expression. That's why, in order to avoid confusion, we do not use `self`
    # as a formal argument. For example:
    #
    # D  = make_deferrable(1)
    # D' = D == 2   --> this calls _DeferredExpression.__eq__(D, 2)
    # evaluate(D')  --> this eventually calls _DeferredExpression.__eq__(1, 2)

    @deferrable
    def __eq__(a, b):
        return a == b

    @deferrable
    def __ne__(a, b):
        return a != b

    @deferrable
    def __lt__(a, b):
        return a < b

    @deferrable
    def __le__(a, b):
        return a <= b

    @deferrable
    def __gt__(a, b):
        return a > b

    @deferrable
    def __ge__(a, b):
        return a >= b

    @deferrable
    def __getitem__(seq, key):
        return seq[key]

    @deferrable
    def __contains__(seq, key):
        '''This method triggers the evaluation of the resulting expression.

        If you want a really deferred check, you should use
        `reframe.utility.sanity.contains()`.  This happens because Python
        always converts the result of `__contains__()` to a boolean value by
        calling `bool()`, which in our case it triggers the evaluation of the
        expression.
        '''
        return key in seq

    @deferrable
    def __add__(a, b):
        return a + b

    @deferrable
    def __sub__(a, b):
        return a - b

    @deferrable
    def __mul__(a, b):
        return a * b

    @deferrable
    def __matmul__(a, b):
        return a @ b

    @deferrable
    def __truediv__(a, b):
        return a / b

    @deferrable
    def __floordiv__(a, b):
        return a // b

    @deferrable
    def __mod__(a, b):
        return a % b

    def __divmod__(self, other):
        '''This is not deferrable.

        Instead it returns a tuple of deferrables that compute the floordiv and
        the mod.
        '''
        return (self.__floordiv__(other), self.__mod__(other))

    @deferrable
    def __pow__(a, b):
        return a**b

    @deferrable
    def __lshift__(a, b):
        return a << b

    @deferrable
    def __rshift__(a, b):
        return a >> b

    @deferrable
    def __and__(a, b):
        return a & b

    @deferrable
    def __xor__(a, b):
        return a ^ b

    @deferrable
    def __or__(a, b):
        return a | b

    # Reflected operators
    @deferrable
    def __radd__(a, b):
        return b + a

    @deferrable
    def __rsub__(a, b):
        return b - a

    @deferrable
    def __rmul__(a, b):
        return b * a

    @deferrable
    def __rmatmul__(a, b):
        return b @ a

    @deferrable
    def __rtruediv__(a, b):
        return b / a

    @deferrable
    def __rfloordiv__(a, b):
        return b // a

    @deferrable
    def __rmod__(a, b):
        return b % a

    def __rdivmod__(self, other):
        '''This is not deferrable.

        Instead it returns a tuple of deferrables that compute the rfloordiv
        and the rmod.
        '''
        return (self.__rfloordiv__(other), self.__rmod__(other))

    @deferrable
    def __rpow__(a, b):
        return b**a

    @deferrable
    def __rlshift__(a, b):
        return b << a

    @deferrable
    def __rrshift__(a, b):
        return b >> a

    @deferrable
    def __rand__(a, b):
        return a & b

    @deferrable
    def __rxor__(a, b):
        return b ^ a

    @deferrable
    def __ror__(a, b):
        return b | a

    # Augmented operators
    #
    # NOTE: These are usually part of mutable objects, however
    # _DeferredExpression remains immutable, since it eventually delegates
    # their evaluation to the objects it wraps
    @deferrable
    def __iadd__(a, b):
        a += b
        return a

    @deferrable
    def __isub__(a, b):
        a -= b
        return a

    @deferrable
    def __imul__(a, b):
        a *= b
        return a

    @deferrable
    def __imatmul__(a, b):
        a @= b
        return a

    @deferrable
    def __itruediv__(a, b):
        a /= b
        return a

    @deferrable
    def __ifloordiv__(a, b):
        a //= b
        return a

    @deferrable
    def __imod__(a, b):
        a %= b
        return a

    @deferrable
    def __ipow__(a, b):
        a **= b
        return a

    @deferrable
    def __ilshift__(a, b):
        a <<= b
        return a

    @deferrable
    def __irshift__(a, b):
        a >>= b
        return a

    @deferrable
    def __iand__(a, b):
        a &= b
        return a

    @deferrable
    def __ixor__(a, b):
        a ^= b
        return a

    @deferrable
    def __ior__(a, b):
        a |= b
        return a

    # Unary operators

    @deferrable
    def __neg__(a):
        return -a

    @deferrable
    def __pos__(a):
        return +a

    @deferrable
    def __abs__(a):
        return abs(a)

    @deferrable
    def __invert__(a):
        return ~a


class _DeferredPerformanceExpression(_DeferredExpression):
    '''Represents a performance function whose evaluation has been deferred.

    It extends the :class:`_DeferredExpression` class by adding the ``unit``
    attribute. This attribute represents the unit of the performance
    metric to be extracted by the performance function.
    '''

    def __init__(self, fn, unit, *args, **kwargs):
        super().__init__(fn, *args, **kwargs)

        if not isinstance(unit, str):
            raise TypeError(
                'performance units must be a string'
            )

        self._unit = unit

    @classmethod
    def construct_from_deferred_expr(cls, expr, unit):
        if not isinstance(expr, _DeferredExpression):
            raise TypeError("'expr' argument is not an instance of the "
                            "_DeferredExpression class")

        return cls(expr._fn, unit, *(expr._args), **(expr._kwargs))

    @property
    def unit(self):
        return self._unit
