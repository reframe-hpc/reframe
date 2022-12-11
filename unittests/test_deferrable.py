# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import reframe.utility.sanity as sn


def test_defer():
    from reframe.core.deferrable import _DeferredExpression

    a = sn.defer(3)
    assert isinstance(a, _DeferredExpression)


def test_deferrable_perf():
    from reframe.core.deferrable import _DeferredPerformanceExpression as dpe

    a = sn.defer(3)
    b = dpe.construct_from_deferred_expr(a, 'some_unit')
    assert b.unit == 'some_unit'

    # Test wrong unit type
    with pytest.raises(TypeError):
        dpe(lambda x: x, 3)

    # Test not from deferred expr
    with pytest.raises(TypeError):
        dpe.construct_from_deferred_expr(lambda x: x, 'some_unit')


def test_evaluate():
    a = sn.defer(3)
    assert 3 == a.evaluate()
    assert 3 == sn.evaluate(a)
    assert 3 == sn.evaluate(3)


def test_recursive_evaluate():
    @sn.deferrable
    def c():
        @sn.deferrable
        def b():
            @sn.deferrable
            def a():
                return sn.defer(3)

            return a()
        return b()

    assert 3 == c().evaluate()


def test_evaluate_cached():
    # A dummy mutable
    my_list = [1]

    @sn.deferrable
    def my_expr():
        return my_list[0]

    expr = my_expr()
    assert expr.evaluate() == 1
    my_list = [2]
    assert expr.evaluate(cache=True) == 2
    my_list = [3]
    assert expr.evaluate() == 2

    # Test that using cache=True updates the previously cached result
    assert expr.evaluate(cache=True) == 3
    my_list = [4]
    assert expr.evaluate() == 3


def test_implicit_eval():
    # Call to bool() on a deferred expression triggers its immediate
    # evaluation.
    a = sn.defer(3)
    assert 3 == a


def test_str():
    assert '[1, 2]' == str(sn.defer([1, 2]))


def test_iter():
    l = [1, 2, 3]
    dl = sn.defer(l)
    l.append(4)
    for i, e in enumerate(dl, start=1):
        assert i == e


@sn.deferrable
def _add(a, b):
    return a + b


def test_kwargs_passing():
    expr = _add(a=4, b=2) / 3
    assert 2 == expr


@pytest.fixture
def value_wrapper():
    class _Value:
        def __init__(self):
            self._value = 0

        @property
        @sn.deferrable
        def value(self):
            return self._value

    return _Value()


def test_eq(value_wrapper):
    expr = value_wrapper.value == 2
    value_wrapper._value = 2
    assert expr


def test_ne(value_wrapper):
    expr = value_wrapper.value != 0
    value_wrapper._value = 2
    assert expr


def test_lt(value_wrapper):
    expr = value_wrapper.value < 0
    value_wrapper._value = -1
    assert expr


def test_le(value_wrapper):
    expr = value_wrapper.value <= 0
    assert expr

    value_wrapper._value = -1
    assert expr


def test_gt(value_wrapper):
    expr = value_wrapper.value > 1
    value_wrapper._value = 2
    assert expr


def test_ge(value_wrapper):
    expr = value_wrapper.value >= 1
    value_wrapper._value = 1
    assert expr

    value_wrapper._value = 2
    assert expr


def test_getitem_list():
    l = [1, 2]
    expr = sn.defer(l)[1] == 3
    l[1] = 3
    assert expr


def test_contains_list():
    l = [1, 2]
    assert 2 in sn.defer(l)


def test_contains_set():
    s = {1, 2}
    assert 2 in sn.defer(s)


def test_contains_dict():
    d = {1: 'a', 2: 'b'}
    assert 2 in sn.defer(d)


class V:
    '''A simple mutable wrapper of an integer value.

    This class is used as a testbed for checking the behaviour of applying
    augmented operators on a deferred expression.
    '''

    def __init__(self, value):
        self._value = value

    def __iadd__(self, other):
        self._value += other._value
        return self

    def __isub__(self, other):
        self._value -= other._value
        return self

    def __imul__(self, other):
        self._value *= other._value
        return self

    def __itruediv__(self, other):
        self._value /= other._value
        return self

    def __ifloordiv__(self, other):
        self._value //= other._value
        return self

    def __imod__(self, other):
        self._value %= other._value
        return self

    def __ipow__(self, other):
        self._value **= other._value
        return self

    def __ilshift__(self, other):
        self._value <<= other._value
        return self

    def __irshift__(self, other):
        self._value >>= other._value
        return self

    def __iand__(self, other):
        self._value &= other._value
        return self

    def __ixor__(self, other):
        self._value ^= other._value
        return self

    def __ior__(self, other):
        self._value |= other._value
        return self

    def __repr__(self):
        return repr(self._value)


def test_add():
    a = sn.defer(1)
    assert 4 == a + 3
    assert 4 == 3 + a


def test_sub():
    a = sn.defer(1)
    assert -2 == a - 3
    assert 2 == 3 - a


def test_mul():
    a = sn.defer(1)
    assert 3 == a * 3
    assert 3 == 3 * a


def test_truediv():
    a = sn.defer(3)
    assert 1.5 == a / 2
    assert 2 / 3 == 2 / a


def test_floordiv():
    a = sn.defer(3)
    assert 1 == a // 2
    assert 0 == 2 // a


def test_mod():
    a = sn.defer(3)
    assert 1 == a % 2
    assert 2 == 2 % a


def test_divmod():
    a = sn.defer(3)
    q, r = divmod(a, 2)
    assert 1 == q
    assert 1 == r

    # Test rdivmod here
    q, r = divmod(2, a)
    assert 0 == q
    assert 2 == r


def test_pow():
    a = sn.defer(3)
    assert 9 == a**2
    assert 8 == 2**a


def test_lshift():
    a = sn.defer(1)
    assert 4 == a << 2
    assert 2 << 1 == 2 << a


def test_rshift():
    a = sn.defer(8)
    assert 1 == a >> 3
    assert 3 >> 8 == 3 >> a


def test_and():
    a = sn.defer(7)
    assert 2 == a & 2
    assert 2 == 2 & a


def test_xor():
    a = sn.defer(7)
    assert 0 == a ^ 7
    assert 0 == 7 ^ a


def test_or():
    a = sn.defer(2)
    assert 7 == a | 5
    assert 7 == 5 | a


def test_expr_chaining():
    a = sn.defer(2)
    assert 64 == a**((a + 1) * a)


def test_neg():
    a = sn.defer(3)
    assert -3 == -a


def test_pos():
    a = sn.defer(3)
    assert +3 == +a


def test_abs():
    a = sn.defer(-3)
    assert 3 == abs(a)


def test_invert():
    a = sn.defer(3)
    assert ~3 == ~a


def test_iadd():
    v = V(1)
    dv = sn.defer(v)
    dv += V(3)
    sn.evaluate(dv)
    assert 4 == v._value


def test_isub():
    v = V(1)
    dv = sn.defer(v)
    dv -= V(3)
    sn.evaluate(dv)
    assert -2 == v._value


def test_imul():
    v = V(1)
    dv = sn.defer(v)
    dv *= V(3)
    sn.evaluate(dv)
    assert 3 == v._value


def test_itruediv():
    v = V(3)
    dv = sn.defer(v)
    dv /= V(2)
    sn.evaluate(dv)
    assert 1.5 == v._value


def test_ifloordiv():
    v = V(3)
    dv = sn.defer(v)
    dv //= V(2)
    sn.evaluate(dv)
    assert 1 == v._value


def test_imod():
    v = V(3)
    dv = sn.defer(v)
    dv %= V(2)
    sn.evaluate(dv)
    assert 1 == v._value


def test_ipow():
    v = V(3)
    dv = sn.defer(v)
    dv **= V(2)
    sn.evaluate(dv)
    assert 9 == v._value


def test_ilshift():
    v = V(1)
    dv = sn.defer(v)
    dv <<= V(2)
    sn.evaluate(dv)
    assert 4 == v._value


def test_irshift():
    v = V(8)
    dv = sn.defer(v)
    dv >>= V(3)
    sn.evaluate(dv)
    assert 1 == v._value


def test_iand():
    v = V(7)
    dv = sn.defer(v)
    dv &= V(2)
    sn.evaluate(dv)
    assert 2 == v._value


def test_ixor():
    v = V(7)
    dv = sn.defer(v)
    dv ^= V(7)
    sn.evaluate(dv)
    assert 0 == v._value


def test_ior():
    v = V(2)
    dv = sn.defer(v)
    dv |= V(5)
    sn.evaluate(dv)
    assert 7 == v._value
