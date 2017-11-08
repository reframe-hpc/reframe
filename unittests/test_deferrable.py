import unittest

from reframe.core.deferrable import *


class TestDeferredUtilities(unittest.TestCase):
    def test_make_deferrable(self):
        from reframe.core.deferrable import _DeferredExpression

        a = make_deferrable(3)
        self.assertIsInstance(a, _DeferredExpression)

    def test_evaluate(self):
        a = make_deferrable(3)
        self.assertEqual(3, a.evaluate())
        self.assertEqual(3, evaluate(a))
        self.assertEqual(3, evaluate(3))

    def test_implicit_eval(self):
        # Call to bool() on a deferred expression triggers its immediate
        # evaluation.
        a = make_deferrable(3)
        self.assertEqual(3, a)

    def test_str(self):
        self.assertEqual('[1, 2]', str(make_deferrable([1, 2])))

    def test_iter(self):
        l = [1, 2, 3]
        dl = make_deferrable(l)
        l.append(4)
        for i, e in enumerate(dl, start=1):
            self.assertEqual(i, e)


class TestKeywordArgs(unittest.TestCase):
    @deferrable
    def add(self, a, b):
        return a + b

    def test_kwargs_passing(self):
        expr = self.add(a=4, b=2) / 3
        self.assertEqual(2, expr)


class TestDeferredRichComparison(unittest.TestCase):
    def setUp(self):
        self._value = 0

    @property
    @deferrable
    def value(self):
        return self._value

    def test_eq(self):
        expr = self.value == 2
        self._value = 2
        self.assertTrue(expr)

    def test_ne(self):
        expr = self.value != 0
        self._value = 2
        self.assertTrue(expr)

    def test_lt(self):
        expr = self.value < 0
        self._value = -1
        self.assertTrue(expr)

    def test_le(self):
        expr = self.value <= 0
        self.assertTrue(expr)

        self._value = -1
        self.assertTrue(expr)

    def test_gt(self):
        expr = self.value > 1
        self._value = 2
        self.assertTrue(expr)

    def test_ge(self):
        expr = self.value >= 1
        self._value = 1
        self.assertTrue(expr)

        self._value = 2
        self.assertTrue(expr)


class TestDeferredContainerOps(unittest.TestCase):
    def test_list_getitem(self):
        l = [1, 2]
        expr = make_deferrable(l)[1] == 3
        l[1] = 3
        self.assertTrue(expr)

    def test_list_contains(self):
        l = [1, 2]
        self.assertIn(2, make_deferrable(l))

    def test_set_contains(self):
        s = {1, 2}
        self.assertIn(2, make_deferrable(s))

    def test_dict_contains(self):
        d = {1: 'a', 2: 'b'}
        self.assertIn(2, make_deferrable(d))


class V:
    """A simple mutable wrapper of an integer value.

    This class is used as a testbed for checking the behaviour of applying
    augmented operators on a deferred expression.
    """

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


class TestDeferredNumericOps(unittest.TestCase):
    def test_add(self):
        a = make_deferrable(1)
        self.assertEqual(4, a + 3)
        self.assertEqual(4, 3 + a)

    def test_sub(self):
        a = make_deferrable(1)
        self.assertEqual(-2, a - 3)
        self.assertEqual(2, 3 - a)

    def test_mul(self):
        a = make_deferrable(1)
        self.assertEqual(3, a * 3)
        self.assertEqual(3, 3 * a)

    def test_truediv(self):
        a = make_deferrable(3)
        self.assertEqual(1.5, a / 2)
        self.assertEqual(2 / 3, 2 / a)

    def test_floordiv(self):
        a = make_deferrable(3)
        self.assertEqual(1, a // 2)
        self.assertEqual(0, 2 // a)

    def test_mod(self):
        a = make_deferrable(3)
        self.assertEqual(1, a % 2)
        self.assertEqual(2, 2 % a)

    def test_divmod(self):
        a = make_deferrable(3)
        q, r = divmod(a, 2)
        self.assertEqual(1, q)
        self.assertEqual(1, r)

        # Test rdivmod here
        q, r = divmod(2, a)
        self.assertEqual(0, q)
        self.assertEqual(2, r)

    def test_pow(self):
        a = make_deferrable(3)
        self.assertEqual(9, a**2)
        self.assertEqual(8, 2**a)

    def test_lshift(self):
        a = make_deferrable(1)
        self.assertEqual(4, a << 2)
        self.assertEqual(2 << 1, 2 << a)

    def test_rshift(self):
        a = make_deferrable(8)
        self.assertEqual(1, a >> 3)
        self.assertEqual(3 >> 8, 3 >> a)

    def test_and(self):
        a = make_deferrable(7)
        self.assertEqual(2, a & 2)
        self.assertEqual(2, 2 & a)

    def test_xor(self):
        a = make_deferrable(7)
        self.assertEqual(0, a ^ 7)
        self.assertEqual(0, 7 ^ a)

    def test_or(self):
        a = make_deferrable(2)
        self.assertEqual(7, a | 5)
        self.assertEqual(7, 5 | a)

    def test_expr_chaining(self):
        a = make_deferrable(2)
        self.assertEqual(64, a**((a + 1) * a))

    def test_neg(self):
        a = make_deferrable(3)
        self.assertEqual(-3, -a)

    def test_pos(self):
        a = make_deferrable(3)
        self.assertEqual(+3, +a)

    def test_abs(self):
        a = make_deferrable(-3)
        self.assertEqual(3, abs(a))

    def test_invert(self):
        a = make_deferrable(3)
        self.assertEqual(~3, ~a)

    def test_iadd(self):
        v = V(1)
        dv = make_deferrable(v)
        dv += V(3)
        evaluate(dv)
        self.assertEqual(4, v._value)

    def test_isub(self):
        v = V(1)
        dv = make_deferrable(v)
        dv -= V(3)
        evaluate(dv)
        self.assertEqual(-2, v._value)

    def test_imul(self):
        v = V(1)
        dv = make_deferrable(v)
        dv *= V(3)
        evaluate(dv)
        self.assertEqual(3, v._value)

    def test_itruediv(self):
        v = V(3)
        dv = make_deferrable(v)
        dv /= V(2)
        evaluate(dv)
        self.assertEqual(1.5, v._value)

    def test_ifloordiv(self):
        v = V(3)
        dv = make_deferrable(v)
        dv //= V(2)
        evaluate(dv)
        self.assertEqual(1, v._value)

    def test_imod(self):
        v = V(3)
        dv = make_deferrable(v)
        dv %= V(2)
        evaluate(dv)
        self.assertEqual(1, v._value)

    def test_ipow(self):
        v = V(3)
        dv = make_deferrable(v)
        dv **= V(2)
        evaluate(dv)
        self.assertEqual(9, v._value)

    def test_ilshift(self):
        v = V(1)
        dv = make_deferrable(v)
        dv <<= V(2)
        evaluate(dv)
        self.assertEqual(4, v._value)

    def test_irshift(self):
        v = V(8)
        dv = make_deferrable(v)
        dv >>= V(3)
        evaluate(dv)
        self.assertEqual(1, v._value)

    def test_iand(self):
        v = V(7)
        dv = make_deferrable(v)
        dv &= V(2)
        evaluate(dv)
        self.assertEqual(2, v._value)

    def test_ixor(self):
        v = V(7)
        dv = make_deferrable(v)
        dv ^= V(7)
        evaluate(dv)
        self.assertEqual(0, v._value)

    def test_ior(self):
        v = V(2)
        dv = make_deferrable(v)
        dv |= V(5)
        evaluate(dv)
        self.assertEqual(7, v._value)
