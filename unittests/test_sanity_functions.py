import builtins
import itertools
import os
import unittest

from tempfile import NamedTemporaryFile
from reframe.core.deferrable import deferrable, evaluate, make_deferrable
from reframe.core.environments import save_environment
from reframe.utility.sanity import *
from unittests.fixtures import TEST_RESOURCES


class TestDeferredBuiltins(unittest.TestCase):
    def setUp(self):
        self._a = 0
        self._b = 1

    # Property retrieval must also be deferred, if we want lazy evaluation of
    # their values. The following demonstrates two different ways to achieve
    # this. Notice that the `getattr` is actually a call to
    # `reframe.utility.sanity.getattr`, since we have imported everything.

    @property
    @sanity_function
    def a(self):
        return self._a

    @property
    def b(self):
        return getattr(self, '_b')

    def test_abs(self):
        self.assertEqual(1.0, abs(1.0))
        self.assertEqual(0.0, abs(0.0))
        self.assertEqual(1.0, abs(-1.0))
        self.assertEqual(2.0, abs(make_deferrable(-2.0)))

    def test_and(self):
        expr = and_(self.a, self.b)
        self._a = 1
        self._b = 1

        self.assertTrue(expr)
        self.assertFalse(not_(expr))

    def test_or(self):
        expr = or_(self.a, self.b)
        self._a = 0
        self._b = 0
        self.assertFalse(expr)
        self.assertTrue(not_(expr))

    def test_all(self):
        l = [1, 1, 0]
        expr = all(l)
        l[2] = 1
        self.assertTrue(expr)

    def test_any(self):
        l = [0, 0, 1]
        expr = any(l)
        l[2] = 0
        self.assertFalse(expr)

    def test_chain(self):
        list1 = ['A', 'B', 'C']
        list2 = ['D', 'E', 'F']
        chain1 = evaluate(chain(make_deferrable(list1), list2))
        chain2 = itertools.chain(list1, list2)
        self.assertTrue(builtins.all(
            (a == b for a, b in builtins.zip(chain1, chain2))))

    def test_enumerate(self):
        de = enumerate(make_deferrable([1, 2]), start=1)
        for i, e in de:
            self.assertEqual(i, e)

    def test_filter(self):
        df = filter(lambda x: x if x % 2 else None,
                    make_deferrable([1, 2, 3, 4, 5]))

        for i, x in enumerate(df, start=1):
            self.assertEqual(2*i - 1, x)

        # Alternative testing
        self.assertEqual([1, 3, 5], list(evaluate(df)))

    def test_hasattr(self):
        e = hasattr(self, '_c')
        self.assertFalse(e)

        self._c = 1
        self.assertTrue(e)

    def test_len(self):
        l = [1, 2]
        dl = make_deferrable(l)
        self.assertEqual(2, len(dl))

        l.append(3)
        self.assertEqual(3, len(dl))

    def test_map(self):
        l = [1, 2, 3]
        dm = map(lambda x: 2*x + 1, l)
        for i, x in enumerate(dm, start=1):
            self.assertEqual(2*i + 1, x)

        # Alternative test
        self.assertEqual([3, 5, 7], list(evaluate(dm)))

    def test_max(self):
        l = [1, 2]
        dl = make_deferrable(l)
        self.assertEqual(2, max(dl))

        l.append(3)
        self.assertEqual(3, max(dl))

    def test_min(self):
        l = [1, 2]
        dl = make_deferrable(l)
        self.assertEqual(1, min(dl))

        l.append(0)
        self.assertEqual(0, min(dl))

    def test_reversed(self):
        l = [1, 2, 3]
        dr = reversed(l)
        self.assertEqual([3, 2, 1], list(evaluate(dr)))

    def test_round(self):
        self.assertEqual(1.0, round(make_deferrable(1.4)))

    def test_setattr(self):
        dset = setattr(self, '_a', 5)
        self.assertEqual(0, self._a)
        evaluate(dset)
        self.assertEqual(5, self._a)

    def test_sorted(self):
        l = [2, 3, 1]
        ds = sorted(l)
        self.assertEqual([1, 2, 3], list(evaluate(ds)))

    def test_sum(self):
        self.assertEqual(3, sum([1, 1, 1]))
        self.assertEqual(3, sum(make_deferrable([1, 1, 1])))

    def test_zip(self):
        la = [1, 2, 3]
        lb = make_deferrable(['a', 'b', 'c'])

        la_new = []
        lb_new = []
        for a, b in zip(la, lb):
            la_new.append(a)
            lb_new.append(b)

        self.assertEqual([1, 2, 3], la_new)
        self.assertEqual(['a', 'b', 'c'], lb_new)


class TestAsserts(unittest.TestCase):
    def setUp(self):
        self.utf16_file = os.path.join(TEST_RESOURCES, 'src', 'homer.txt')

    def test_assert_true(self):
        self.assertTrue(assert_true(True))
        self.assertTrue(assert_true(1))
        self.assertTrue(assert_true([1]))
        self.assertTrue(assert_true(range(1)))
        self.assertRaisesRegex(SanityError, 'False is not True',
                               evaluate, assert_true(False))
        self.assertRaisesRegex(SanityError, '0 is not True',
                               evaluate, assert_true(0))
        self.assertRaisesRegex(SanityError, '\[\] is not True',
                               evaluate, assert_true([]))
        self.assertRaisesRegex(SanityError, 'range\(.+\) is not True',
                               evaluate, assert_true(range(0)))
        self.assertRaisesRegex(SanityError, 'not true',
                               evaluate, assert_true(0, msg='not true'))

    def test_assert_true_with_deferrables(self):
        self.assertTrue(assert_true(make_deferrable(True)))
        self.assertTrue(assert_true(make_deferrable(1)))
        self.assertTrue(assert_true(make_deferrable([1])))
        self.assertRaisesRegex(SanityError, 'False is not True',
                               evaluate, assert_true(make_deferrable(False)))
        self.assertRaisesRegex(SanityError, '0 is not True',
                               evaluate, assert_true(make_deferrable(0)))
        self.assertRaisesRegex(SanityError, '\[\] is not True',
                               evaluate, assert_true(make_deferrable([])))

    def test_assert_false(self):
        self.assertTrue(assert_false(False))
        self.assertTrue(assert_false(0))
        self.assertTrue(assert_false([]))
        self.assertTrue(assert_false(range(0)))
        self.assertRaisesRegex(SanityError, 'True is not False',
                               evaluate, assert_false(True))
        self.assertRaisesRegex(SanityError, '1 is not False',
                               evaluate, assert_false(1))
        self.assertRaisesRegex(SanityError, '\[1\] is not False',
                               evaluate, assert_false([1]))
        self.assertRaisesRegex(SanityError, 'range\(.+\) is not False',
                               evaluate, assert_false(range(1)))

    def test_assert_false_with_deferrables(self):
        self.assertTrue(assert_false(make_deferrable(False)))
        self.assertTrue(assert_false(make_deferrable(0)))
        self.assertTrue(assert_false(make_deferrable([])))
        self.assertRaisesRegex(SanityError, 'True is not False',
                               evaluate, assert_false(make_deferrable(True)))
        self.assertRaisesRegex(SanityError, '1 is not False',
                               evaluate, assert_false(make_deferrable(1)))
        self.assertRaisesRegex(SanityError, '\[1\] is not False',
                               evaluate, assert_false(make_deferrable([1])))

    def test_assert_eq(self):
        self.assertTrue(assert_eq(1, 1))
        self.assertTrue(assert_eq(1, True))
        self.assertRaisesRegex(SanityError, '1 != 2',
                               evaluate, assert_eq(1, 2))
        self.assertRaisesRegex(SanityError, '1 != False',
                               evaluate, assert_eq(1, False))
        self.assertRaisesRegex(
            SanityError, '1 is not equals to 2',
            evaluate, assert_eq(1, 2, '{0} is not equals to {1}'))

    def test_assert_eq_with_deferrables(self):
        self.assertTrue(assert_eq(1, make_deferrable(1)))
        self.assertTrue(assert_eq(make_deferrable(1), True))
        self.assertRaisesRegex(SanityError, '1 != 2',
                               evaluate, assert_eq(make_deferrable(1), 2))
        self.assertRaisesRegex(SanityError, '1 != False',
                               evaluate, assert_eq(make_deferrable(1), False))

    def test_assert_ne(self):
        self.assertTrue(assert_ne(1, 2))
        self.assertTrue(assert_ne(1, False))
        self.assertRaisesRegex(SanityError, '1 == 1',
                               evaluate, assert_ne(1, 1))
        self.assertRaisesRegex(SanityError, '1 == True',
                               evaluate, assert_ne(1, True))

    def test_assert_ne_with_deferrables(self):
        self.assertTrue(assert_ne(1, make_deferrable(2)))
        self.assertTrue(assert_ne(make_deferrable(1), False))
        self.assertRaisesRegex(SanityError, '1 == 1',
                               evaluate, assert_ne(make_deferrable(1), 1))
        self.assertRaisesRegex(SanityError, '1 == True',
                               evaluate, assert_ne(make_deferrable(1), True))

    def test_assert_gt(self):
        self.assertTrue(assert_gt(3, 1))
        self.assertRaisesRegex(SanityError, '1 <= 3',
                               evaluate, assert_gt(1, 3))

    def test_assert_gt_with_deferrables(self):
        self.assertTrue(assert_gt(3, make_deferrable(1)))
        self.assertRaisesRegex(SanityError, '1 <= 3',
                               evaluate, assert_gt(1, make_deferrable(3)))

    def test_assert_ge(self):
        self.assertTrue(assert_ge(3, 1))
        self.assertTrue(assert_ge(3, 3))
        self.assertRaisesRegex(SanityError, '1 < 3',
                               evaluate, assert_ge(1, 3))

    def test_assert_ge_with_deferrables(self):
        self.assertTrue(assert_ge(3, make_deferrable(1)))
        self.assertTrue(assert_ge(3, make_deferrable(3)))
        self.assertRaisesRegex(SanityError, '1 < 3',
                               evaluate, assert_ge(1, make_deferrable(3)))

    def test_assert_lt(self):
        self.assertTrue(assert_lt(1, 3))
        self.assertRaisesRegex(SanityError, '3 >= 1',
                               evaluate, assert_lt(3, 1))

    def test_assert_lt_with_deferrables(self):
        self.assertTrue(assert_lt(1, make_deferrable(3)))
        self.assertRaisesRegex(SanityError, '3 >= 1',
                               evaluate, assert_lt(3, make_deferrable(1)))

    def test_assert_le(self):
        self.assertTrue(assert_le(1, 1))
        self.assertTrue(assert_le(3, 3))
        self.assertRaisesRegex(SanityError, '3 > 1',
                               evaluate, assert_le(3, 1))

    def test_assert_le_with_deferrables(self):
        self.assertTrue(assert_le(1, make_deferrable(3)))
        self.assertTrue(assert_le(3, make_deferrable(3)))
        self.assertRaisesRegex(SanityError, '3 > 1',
                               evaluate, assert_le(3, make_deferrable(1)))

    def test_assert_in(self):
        self.assertTrue(assert_in(1, [1, 2, 3]))
        self.assertRaisesRegex(SanityError, '0 is not in \[1, 2, 3\]',
                               evaluate, assert_in(0, [1, 2, 3]))

    def test_assert_in_with_deferrables(self):
        self.assertTrue(assert_in(1, make_deferrable([1, 2, 3])))
        self.assertRaisesRegex(
            SanityError, '0 is not in \[1, 2, 3\]',
            evaluate, assert_in(0, make_deferrable([1, 2, 3])))

    def test_assert_not_in(self):
        self.assertTrue(assert_not_in(0, [1, 2, 3]))
        self.assertRaisesRegex(SanityError, '1 is in \[1, 2, 3\]',
                               evaluate, assert_not_in(1, [1, 2, 3]))

    def test_assert_not_in_with_deferrables(self):
        self.assertTrue(assert_not_in(0, make_deferrable([1, 2, 3])))
        self.assertRaisesRegex(
            SanityError, '1 is in \[1, 2, 3\]',
            evaluate, assert_not_in(1, make_deferrable([1, 2, 3])))

    def test_assert_bounded(self):
        self.assertTrue(assert_bounded(1, -1.5, 1.5))
        self.assertTrue(assert_bounded(1, upper=1.5))
        self.assertTrue(assert_bounded(1, lower=-1.5))
        self.assertTrue(assert_bounded(1))
        self.assertRaisesRegex(SanityError,
                               'value 1 not within bounds -0\.5\.\.0\.5',
                               evaluate, assert_bounded(1, -0.5, 0.5))
        self.assertRaisesRegex(SanityError,
                               'value 1 not within bounds -inf\.\.0\.5',
                               evaluate, assert_bounded(1, upper=0.5))
        self.assertRaisesRegex(SanityError,
                               'value 1 not within bounds 1\.5\.\.inf',
                               evaluate, assert_bounded(1, lower=1.5))
        self.assertRaisesRegex(
            SanityError, 'value 1 is out of bounds', evaluate,
            assert_bounded(1, -0.5, 0.5, 'value {0} is out of bounds'))

    def test_assert_reference(self):
        self.assertTrue(assert_reference(0.9, 1, -0.2, 0.1))
        self.assertTrue(assert_reference(0.9, 1, upper_thres=0.1))
        self.assertTrue(assert_reference(0.9, 1, lower_thres=-0.2))
        self.assertTrue(assert_reference(0.9, 1))

        # Check negatives
        self.assertTrue(assert_reference(-0.9, -1, -0.2, 0.1))
        self.assertTrue(assert_reference(-0.9, -1, -0.2))
        self.assertTrue(assert_reference(-0.9, -1, upper_thres=0.1))
        self.assertTrue(assert_reference(-0.9, -1))

        self.assertRaisesRegex(
            SanityError,
            '0\.5 is beyond reference value 1 \(l=0\.8, u=1\.1\)',
            evaluate, assert_reference(0.5, 1, -0.2, 0.1)
        )
        self.assertRaisesRegex(
            SanityError,
            '0\.5 is beyond reference value 1 \(l=0\.8, u=inf\)',
            evaluate, assert_reference(0.5, 1, -0.2)
        )
        self.assertRaisesRegex(
            SanityError,
            '1\.5 is beyond reference value 1 \(l=0\.8, u=1\.1\)',
            evaluate, assert_reference(1.5, 1, -0.2, 0.1)
        )
        self.assertRaisesRegex(
            SanityError,
            '1\.5 is beyond reference value 1 \(l=-inf, u=1\.1\)',
            evaluate, assert_reference(
                1.5, 1, lower_thres=None, upper_thres=0.1)
        )
        self.assertRaisesRegex(
            SanityError,
            '-0\.8 is beyond reference value -1 \(l=-1\.2, u=-0\.9\)',
            evaluate, assert_reference(-0.8, -1, -0.2, 0.1)
        )

        # Check invalid thresholds
        self.assertRaisesRegex(SanityError,
                               'invalid high threshold value: -0\.1',
                               evaluate, assert_reference(0.9, 1, -0.2, -0.1))
        self.assertRaisesRegex(SanityError,
                               'invalid high threshold value: 1\.1',
                               evaluate, assert_reference(0.9, 1, -0.2, 1.1))
        self.assertRaisesRegex(SanityError,
                               'invalid low threshold value: 0\.2',
                               evaluate, assert_reference(0.9, 1, 0.2, 0.1))
        self.assertRaisesRegex(SanityError,
                               'invalid low threshold value: 1\.2',
                               evaluate, assert_reference(0.9, 1, 1.2, 0.1))

    def _write_tempfile(self):
        ret = None
        with NamedTemporaryFile('wt', delete=False) as fp:
            ret = fp.name
            fp.write('Step: 1\n')
            fp.write('Step: 2\n')
            fp.write('Step: 3\n')

        return ret

    def test_assert_found(self):
        tempfile = self._write_tempfile()
        self.assertTrue(assert_found('Step: \d+', tempfile))
        self.assertTrue(assert_found('Step: \d+', make_deferrable(tempfile)))
        self.assertRaises(SanityError, evaluate,
                          assert_found('foo: \d+', tempfile))
        os.remove(tempfile)

    def test_assert_found_encoding(self):
        self.assertTrue(
            assert_found('Odyssey', self.utf16_file, encoding='utf-16')
        )

    def test_assert_not_found(self):
        tempfile = self._write_tempfile()
        self.assertTrue(assert_not_found('foo: \d+', tempfile))
        self.assertTrue(
            assert_not_found('foo: \d+', make_deferrable(tempfile))
        )
        self.assertRaises(SanityError, evaluate,
                          assert_not_found('Step: \d+', tempfile))
        os.remove(tempfile)

    def test_assert_not_found_encoding(self):
        self.assertTrue(
            assert_not_found('Iliad', self.utf16_file, encoding='utf-16')
        )


class TestUtilityFunctions(unittest.TestCase):
    def test_getitem(self):
        l = [1, 2, 3]
        d = {'a': 1, 'b': 2, 'c': 3}

        self.assertEqual(2, getitem(l, 1))
        self.assertEqual(2, getitem(d, 'b'))
        self.assertRaisesRegex(SanityError, 'index out of bounds: 10',
                               evaluate, getitem(l, 10))
        self.assertRaisesRegex(SanityError, 'key not found: k',
                               evaluate, getitem(d, 'k'))

    def test_getitem_with_deferrables(self):
        l = make_deferrable([1, 2, 3])
        d = make_deferrable({'a': 1, 'b': 2, 'c': 3})

        self.assertEqual(2, getitem(l, 1))
        self.assertEqual(2, getitem(d, 'b'))
        self.assertRaisesRegex(SanityError, 'index out of bounds: 10',
                               evaluate, getitem(l, 10))
        self.assertRaisesRegex(SanityError, 'key not found: k',
                               evaluate, getitem(d, 'k'))

    def test_count(self):
        # Use a custom generator for testing
        def myrange(n):
            for i in range(n):
                yield i

        self.assertEqual(3, count([1, 2, 3]))
        self.assertEqual(3, count((1, 2, 3)))
        self.assertEqual(3, count({1, 2, 3}))
        self.assertEqual(3, count({'a': 1, 'b': 2, 'c': 3}))
        self.assertEqual(3, count(range(3)))
        self.assertEqual(3, count(myrange(3)))

        # Test empty sequences
        self.assertEqual(0, count([]))
        self.assertEqual(0, count({}))
        self.assertEqual(0, count(set()))
        self.assertEqual(0, count(range(0)))
        self.assertEqual(0, count(myrange(0)))

    def test_glob(self):
        filepatt = os.path.join(TEST_RESOURCES, '*.py')
        self.assertTrue(glob(filepatt))
        self.assertTrue(glob(make_deferrable(filepatt)))

    def test_iglob(self):
        filepatt = os.path.join(TEST_RESOURCES, '*.py')
        self.assertTrue(count(iglob(filepatt)))
        self.assertTrue(count(iglob(make_deferrable(filepatt))))


class TestPatternMatchingFunctions(unittest.TestCase):
    def setUp(self):
        self.tempfile = None
        self.utf16_file = os.path.join(TEST_RESOURCES, 'src', 'homer.txt')
        with NamedTemporaryFile('wt', delete=False) as fp:
            self.tempfile = fp.name
            fp.write('Step: 1\n')
            fp.write('Step: 2\n')
            fp.write('Step: 3\n')

    def tearDown(self):
        os.remove(self.tempfile)

    def test_findall(self):
        res = evaluate(findall('Step: \d+', self.tempfile))
        self.assertEqual(3, builtins.len(res))

        res = evaluate(findall('Step:.*', self.tempfile))
        self.assertEqual(3, builtins.len(res))

        res = evaluate(findall('Step: [12]', self.tempfile))
        self.assertEqual(2, builtins.len(res))

        # Check the matches
        for expected, match in builtins.zip(['Step: 1', 'Step: 2'], res):
            self.assertEqual(expected, match.group(0))

        # Check groups
        res = evaluate(findall('Step: (?P<no>\d+)', self.tempfile))
        for step, match in builtins.enumerate(res, start=1):
            self.assertEqual(step, builtins.int(match.group(1)))
            self.assertEqual(step, builtins.int(match.group('no')))

    def test_findall_encoding(self):
        res = evaluate(
            findall(r'Odyssey', self.utf16_file, encoding='utf-16')
        )
        self.assertEqual(1, len(res))

    def test_findall_error(self):
        self.assertRaises(SanityError, evaluate,
                          findall('Step: \d+', 'foo.txt'))

    def test_extractall(self):
        # Check numeric groups
        res = evaluate(extractall('Step: (?P<no>\d+)', self.tempfile, 1))
        for expected, v in builtins.enumerate(res, start=1):
            self.assertEqual(str(expected), v)

        # Check named groups
        res = evaluate(extractall('Step: (?P<no>\d+)', self.tempfile, 'no'))
        for expected, v in builtins.enumerate(res, start=1):
            self.assertEqual(str(expected), v)

        # Check convert function
        res = evaluate(extractall('Step: (?P<no>\d+)',
                                  self.tempfile, 'no', builtins.int))
        for expected, v in builtins.enumerate(res, start=1):
            self.assertEqual(expected, v)

    def test_extractall_encoding(self):
        res = evaluate(
            extractall(r'Odyssey', self.utf16_file, encoding='utf-16')
        )
        self.assertEqual(1, len(res))

    def test_extractall_error(self):
        self.assertRaises(SanityError, evaluate,
                          extractall('Step: (\d+)', 'foo.txt', 1))
        self.assertRaises(
            SanityError, evaluate,
            extractall('Step: (?P<no>\d+)', self.tempfile, conv=builtins.int)
        )
        self.assertRaises(SanityError, evaluate,
                          extractall('Step: (\d+)', self.tempfile, 2))
        self.assertRaises(
            SanityError, evaluate,
            extractall('Step: (?P<no>\d+)', self.tempfile, 'foo'))

    def test_extractall_custom_conv(self):
        res = evaluate(extractall('Step: (\d+)', self.tempfile, 1,
                                  lambda x: builtins.int(x)))
        for expected, v in builtins.enumerate(res, start=1):
            self.assertEqual(expected, v)

        # Check error in custom function
        self.assertRaises(
            SanityError, evaluate,
            extractall('Step: (\d+)', self.tempfile,
                       conv=lambda x: builtins.int(x))
        )

        # Check error with a callable object
        class C:
            def __call__(self, x):
                return builtins.int(x)

        self.assertRaises(
            SanityError, evaluate,
            extractall('Step: (\d+)', self.tempfile, conv=C())
        )

    def test_extractsingle(self):
        for i in range(1, 4):
            self.assertEqual(
                i,
                extractsingle('Step: (\d+)', self.tempfile,
                              1, builtins.int, i-1)
            )

        # Test out of bounds access
        self.assertRaises(
            SanityError, evaluate,
            extractsingle('Step: (\d+)', self.tempfile, 1, builtins.int, 100)
        )

    def test_extractsingle_encoding(self):
        res = evaluate(
            extractsingle(r'Odyssey', self.utf16_file, encoding='utf-16')
        )
        self.assertNotEqual(-1, res.find('Odyssey'))

    def test_safe_format(self):
        from reframe.utility.sanity import _format

        s = 'There is {0} and {1}.'
        self.assertEqual(s, _format(s))
        self.assertEqual(s, _format(s, 'bacon'))
        self.assertEqual('There is egg and spam.', _format(s, 'egg', 'spam'))
        self.assertEqual('There is egg and bacon.',
                         _format(s, 'egg', 'bacon', 'spam'))

        s = 'no placeholders'
        self.assertEqual(s, _format(s))
        self.assertEqual(s, _format(s, 'bacon'))


class TestNumericFunctions(unittest.TestCase):
    def test_avg(self):
        res = evaluate(avg([1, 2, 3, 4]))
        self.assertEqual(2.5, res)

        # Check result when passing a generator
        res = evaluate(avg(range(1, 5)))
        self.assertEqual(2.5, res)

        # Check with single element container
        res = evaluate(avg(range(1, 2)))
        self.assertEqual(1, res)

        # Check with empty container
        self.assertRaises(SanityError, evaluate, avg([]))
