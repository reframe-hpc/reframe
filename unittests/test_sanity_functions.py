import itertools
import os
import unittest
from tempfile import NamedTemporaryFile

import reframe.utility.sanity as sn
from reframe.core.deferrable import evaluate, make_deferrable
from reframe.core.exceptions import SanityError
from unittests.fixtures import TEST_RESOURCES_CHECKS


class TestDeferredBuiltins(unittest.TestCase):
    def setUp(self):
        self._a = 0
        self._b = 1

    # Property retrieval must also be deferred, if we want lazy evaluation of
    # their values. The following demonstrates two different ways to achieve
    # this. Notice that the `getattr` is actually a call to
    # `reframe.utility.sanity.getattr`, since we have imported everything.

    @property
    @sn.sanity_function
    def a(self):
        return self._a

    @property
    def b(self):
        return sn.getattr(self, '_b')

    def test_abs(self):
        self.assertEqual(1.0, sn.abs(1.0))
        self.assertEqual(0.0, sn.abs(0.0))
        self.assertEqual(1.0, sn.abs(-1.0))
        self.assertEqual(2.0, sn.abs(make_deferrable(-2.0)))

    def test_and(self):
        expr = sn.and_(self.a, self.b)
        self._a = 1
        self._b = 1

        self.assertTrue(expr)
        self.assertFalse(sn.not_(expr))

    def test_or(self):
        expr = sn.or_(self.a, self.b)
        self._a = 0
        self._b = 0
        self.assertFalse(expr)
        self.assertTrue(sn.not_(expr))

    def test_all(self):
        l = [1, 1, 0]
        expr = sn.all(l)
        l[2] = 1
        self.assertTrue(expr)

    def test_allx(self):
        self.assertTrue(sn.allx([1, 1, 1]))
        self.assertFalse(sn.allx([1, 0]))
        self.assertFalse(sn.allx([]))
        self.assertFalse(sn.allx(i for i in range(0)))
        self.assertTrue(sn.allx(i for i in range(1, 2)))
        with self.assertRaises(TypeError):
            sn.evaluate(sn.allx(None))

    def test_any(self):
        l = [0, 0, 1]
        expr = sn.any(l)
        l[2] = 0
        self.assertFalse(expr)

    def test_enumerate(self):
        de = sn.enumerate(make_deferrable([1, 2]), start=1)
        for i, e in de:
            self.assertEqual(i, e)

    def test_filter(self):
        df = sn.filter(lambda x: x if x % 2 else None,
                       make_deferrable([1, 2, 3, 4, 5]))

        for i, x in sn.enumerate(df, start=1):
            self.assertEqual(2*i - 1, x)

        # Alternative testing
        self.assertEqual([1, 3, 5], list(evaluate(df)))

    def test_hasattr(self):
        e = sn.hasattr(self, '_c')
        self.assertFalse(e)

        self._c = 1
        self.assertTrue(e)

    def test_len(self):
        l = [1, 2]
        dl = make_deferrable(l)
        self.assertEqual(2, sn.len(dl))

        l.append(3)
        self.assertEqual(3, sn.len(dl))

    def test_map(self):
        l = [1, 2, 3]
        dm = sn.map(lambda x: 2*x + 1, l)
        for i, x in sn.enumerate(dm, start=1):
            self.assertEqual(2*i + 1, x)

        # Alternative test
        self.assertEqual([3, 5, 7], list(evaluate(dm)))

    def test_max(self):
        l = [1, 2]
        dl = make_deferrable(l)
        self.assertEqual(2, sn.max(dl))

        l.append(3)
        self.assertEqual(3, sn.max(dl))

    def test_min(self):
        l = [1, 2]
        dl = make_deferrable(l)
        self.assertEqual(1, sn.min(dl))

        l.append(0)
        self.assertEqual(0, sn.min(dl))

    def test_reversed(self):
        l = [1, 2, 3]
        dr = sn.reversed(l)
        self.assertEqual([3, 2, 1], list(evaluate(dr)))

    def test_round(self):
        self.assertEqual(1.0, sn.round(make_deferrable(1.4)))

    def test_setattr(self):
        dset = sn.setattr(self, '_a', 5)
        self.assertEqual(0, self._a)
        evaluate(dset)
        self.assertEqual(5, self._a)

    def test_sorted(self):
        l = [2, 3, 1]
        ds = sn.sorted(l)
        self.assertEqual([1, 2, 3], list(evaluate(ds)))

    def test_sum(self):
        self.assertEqual(3, sn.sum([1, 1, 1]))
        self.assertEqual(3, sn.sum(make_deferrable([1, 1, 1])))

    def test_zip(self):
        la = [1, 2, 3]
        lb = make_deferrable(['a', 'b', 'c'])

        la_new = []
        lb_new = []
        for a, b in sn.zip(la, lb):
            la_new.append(a)
            lb_new.append(b)

        self.assertEqual([1, 2, 3], la_new)
        self.assertEqual(['a', 'b', 'c'], lb_new)


class TestAsserts(unittest.TestCase):
    def setUp(self):
        self.utf16_file = os.path.join(TEST_RESOURCES_CHECKS,
                                       'src', 'homer.txt')

    def test_assert_true(self):
        self.assertTrue(sn.assert_true(True))
        self.assertTrue(sn.assert_true(1))
        self.assertTrue(sn.assert_true([1]))
        self.assertTrue(sn.assert_true(range(1)))
        self.assertRaisesRegex(SanityError, 'False is not True',
                               evaluate, sn.assert_true(False))
        self.assertRaisesRegex(SanityError, '0 is not True',
                               evaluate, sn.assert_true(0))
        self.assertRaisesRegex(SanityError, r'\[\] is not True',
                               evaluate, sn.assert_true([]))
        self.assertRaisesRegex(SanityError, r'range\(.+\) is not True',
                               evaluate, sn.assert_true(range(0)))
        self.assertRaisesRegex(SanityError, 'not true',
                               evaluate, sn.assert_true(0, msg='not true'))

    def test_assert_true_with_deferrables(self):
        self.assertTrue(sn.assert_true(make_deferrable(True)))
        self.assertTrue(sn.assert_true(make_deferrable(1)))
        self.assertTrue(sn.assert_true(make_deferrable([1])))
        self.assertRaisesRegex(SanityError, 'False is not True',
                               evaluate, sn.assert_true(make_deferrable(False)))
        self.assertRaisesRegex(SanityError, '0 is not True',
                               evaluate, sn.assert_true(make_deferrable(0)))
        self.assertRaisesRegex(SanityError, r'\[\] is not True',
                               evaluate, sn.assert_true(make_deferrable([])))

    def test_assert_false(self):
        self.assertTrue(sn.assert_false(False))
        self.assertTrue(sn.assert_false(0))
        self.assertTrue(sn.assert_false([]))
        self.assertTrue(sn.assert_false(range(0)))
        self.assertRaisesRegex(SanityError, 'True is not False',
                               evaluate, sn.assert_false(True))
        self.assertRaisesRegex(SanityError, '1 is not False',
                               evaluate, sn.assert_false(1))
        self.assertRaisesRegex(SanityError, r'\[1\] is not False',
                               evaluate, sn.assert_false([1]))
        self.assertRaisesRegex(SanityError, r'range\(.+\) is not False',
                               evaluate, sn.assert_false(range(1)))

    def test_assert_false_with_deferrables(self):
        self.assertTrue(sn.assert_false(make_deferrable(False)))
        self.assertTrue(sn.assert_false(make_deferrable(0)))
        self.assertTrue(sn.assert_false(make_deferrable([])))
        self.assertRaisesRegex(SanityError, 'True is not False',
                               evaluate, sn.assert_false(make_deferrable(True)))
        self.assertRaisesRegex(SanityError, '1 is not False',
                               evaluate, sn.assert_false(make_deferrable(1)))
        self.assertRaisesRegex(SanityError, r'\[1\] is not False',
                               evaluate, sn.assert_false(make_deferrable([1])))

    def test_assert_eq(self):
        self.assertTrue(sn.assert_eq(1, 1))
        self.assertTrue(sn.assert_eq(1, True))
        self.assertRaisesRegex(SanityError, '1 != 2',
                               evaluate, sn.assert_eq(1, 2))
        self.assertRaisesRegex(SanityError, '1 != False',
                               evaluate, sn.assert_eq(1, False))
        self.assertRaisesRegex(
            SanityError, '1 is not equals to 2',
            evaluate, sn.assert_eq(1, 2, '{0} is not equals to {1}'))

    def test_assert_eq_with_deferrables(self):
        self.assertTrue(sn.assert_eq(1, make_deferrable(1)))
        self.assertTrue(sn.assert_eq(make_deferrable(1), True))
        self.assertRaisesRegex(SanityError, '1 != 2',
                               evaluate, sn.assert_eq(make_deferrable(1), 2))
        self.assertRaisesRegex(SanityError, '1 != False',
                               evaluate, sn.assert_eq(make_deferrable(1), False))

    def test_assert_ne(self):
        self.assertTrue(sn.assert_ne(1, 2))
        self.assertTrue(sn.assert_ne(1, False))
        self.assertRaisesRegex(SanityError, '1 == 1',
                               evaluate, sn.assert_ne(1, 1))
        self.assertRaisesRegex(SanityError, '1 == True',
                               evaluate, sn.assert_ne(1, True))

    def test_assert_ne_with_deferrables(self):
        self.assertTrue(sn.assert_ne(1, make_deferrable(2)))
        self.assertTrue(sn.assert_ne(make_deferrable(1), False))
        self.assertRaisesRegex(SanityError, '1 == 1',
                               evaluate, sn.assert_ne(make_deferrable(1), 1))
        self.assertRaisesRegex(SanityError, '1 == True',
                               evaluate, sn.assert_ne(make_deferrable(1), True))

    def test_assert_gt(self):
        self.assertTrue(sn.assert_gt(3, 1))
        self.assertRaisesRegex(SanityError, '1 <= 3',
                               evaluate, sn.assert_gt(1, 3))

    def test_assert_gt_with_deferrables(self):
        self.assertTrue(sn.assert_gt(3, make_deferrable(1)))
        self.assertRaisesRegex(SanityError, '1 <= 3',
                               evaluate, sn.assert_gt(1, make_deferrable(3)))

    def test_assert_ge(self):
        self.assertTrue(sn.assert_ge(3, 1))
        self.assertTrue(sn.assert_ge(3, 3))
        self.assertRaisesRegex(SanityError, '1 < 3',
                               evaluate, sn.assert_ge(1, 3))

    def test_assert_ge_with_deferrables(self):
        self.assertTrue(sn.assert_ge(3, make_deferrable(1)))
        self.assertTrue(sn.assert_ge(3, make_deferrable(3)))
        self.assertRaisesRegex(SanityError, '1 < 3',
                               evaluate, sn.assert_ge(1, make_deferrable(3)))

    def test_assert_lt(self):
        self.assertTrue(sn.assert_lt(1, 3))
        self.assertRaisesRegex(SanityError, '3 >= 1',
                               evaluate, sn.assert_lt(3, 1))

    def test_assert_lt_with_deferrables(self):
        self.assertTrue(sn.assert_lt(1, make_deferrable(3)))
        self.assertRaisesRegex(SanityError, '3 >= 1',
                               evaluate, sn.assert_lt(3, make_deferrable(1)))

    def test_assert_le(self):
        self.assertTrue(sn.assert_le(1, 1))
        self.assertTrue(sn.assert_le(3, 3))
        self.assertRaisesRegex(SanityError, '3 > 1',
                               evaluate, sn.assert_le(3, 1))

    def test_assert_le_with_deferrables(self):
        self.assertTrue(sn.assert_le(1, make_deferrable(3)))
        self.assertTrue(sn.assert_le(3, make_deferrable(3)))
        self.assertRaisesRegex(SanityError, '3 > 1',
                               evaluate, sn.assert_le(3, make_deferrable(1)))

    def test_assert_in(self):
        self.assertTrue(sn.assert_in(1, [1, 2, 3]))
        self.assertRaisesRegex(SanityError, r'0 is not in \[1, 2, 3\]',
                               evaluate, sn.assert_in(0, [1, 2, 3]))

    def test_assert_in_with_deferrables(self):
        self.assertTrue(sn.assert_in(1, make_deferrable([1, 2, 3])))
        self.assertRaisesRegex(
            SanityError, r'0 is not in \[1, 2, 3\]',
            evaluate, sn.assert_in(0, make_deferrable([1, 2, 3])))

    def test_assert_not_in(self):
        self.assertTrue(sn.assert_not_in(0, [1, 2, 3]))
        self.assertRaisesRegex(SanityError, r'1 is in \[1, 2, 3\]',
                               evaluate, sn.assert_not_in(1, [1, 2, 3]))

    def test_assert_not_in_with_deferrables(self):
        self.assertTrue(sn.assert_not_in(0, make_deferrable([1, 2, 3])))
        self.assertRaisesRegex(
            SanityError, r'1 is in \[1, 2, 3\]',
            evaluate, sn.assert_not_in(1, make_deferrable([1, 2, 3])))

    def test_assert_bounded(self):
        self.assertTrue(sn.assert_bounded(1, -1.5, 1.5))
        self.assertTrue(sn.assert_bounded(1, upper=1.5))
        self.assertTrue(sn.assert_bounded(1, lower=-1.5))
        self.assertTrue(sn.assert_bounded(1))
        self.assertRaisesRegex(SanityError,
                               r'value 1 not within bounds -0\.5\.\.0\.5',
                               evaluate, sn.assert_bounded(1, -0.5, 0.5))
        self.assertRaisesRegex(SanityError,
                               r'value 1 not within bounds -inf\.\.0\.5',
                               evaluate, sn.assert_bounded(1, upper=0.5))
        self.assertRaisesRegex(SanityError,
                               r'value 1 not within bounds 1\.5\.\.inf',
                               evaluate, sn.assert_bounded(1, lower=1.5))
        self.assertRaisesRegex(
            SanityError, 'value 1 is out of bounds', evaluate,
            sn.assert_bounded(1, -0.5, 0.5, 'value {0} is out of bounds'))

    def test_assert_reference(self):
        self.assertTrue(sn.assert_reference(0.9, 1, -0.2, 0.1))
        self.assertTrue(sn.assert_reference(0.9, 1, upper_thres=0.1))
        self.assertTrue(sn.assert_reference(0.9, 1, lower_thres=-0.2))
        self.assertTrue(sn.assert_reference(0.9, 1))

        # Check negatives
        self.assertTrue(sn.assert_reference(-0.9, -1, -0.2, 0.1))
        self.assertTrue(sn.assert_reference(-0.9, -1, -0.2))
        self.assertTrue(sn.assert_reference(-0.9, -1, upper_thres=0.1))
        self.assertTrue(sn.assert_reference(-0.9, -1))

        # Check upper threshold values greater than 1
        self.assertTrue(sn.assert_reference(20.0, 10.0, None, 3.0))
        self.assertTrue(sn.assert_reference(-50.0, -20.0, -2.0, 0.5))

        self.assertRaisesRegex(
            SanityError,
            r'0\.5 is beyond reference value 1 \(l=0\.8, u=1\.1\)',
            evaluate, sn.assert_reference(0.5, 1, -0.2, 0.1)
        )
        self.assertRaisesRegex(
            SanityError,
            r'0\.5 is beyond reference value 1 \(l=0\.8, u=inf\)',
            evaluate, sn.assert_reference(0.5, 1, -0.2)
        )
        self.assertRaisesRegex(
            SanityError,
            r'1\.5 is beyond reference value 1 \(l=0\.8, u=1\.1\)',
            evaluate, sn.assert_reference(1.5, 1, -0.2, 0.1)
        )
        self.assertRaisesRegex(
            SanityError,
            r'1\.5 is beyond reference value 1 \(l=-inf, u=1\.1\)',
            evaluate, sn.assert_reference(
                1.5, 1, lower_thres=None, upper_thres=0.1)
        )
        self.assertRaisesRegex(
            SanityError,
            r'-0\.8 is beyond reference value -1 \(l=-1\.2, u=-0\.9\)',
            evaluate, sn.assert_reference(-0.8, -1, -0.2, 0.1)
        )

        # Check invalid thresholds
        self.assertRaisesRegex(SanityError,
                               r'invalid high threshold value: -0\.1',
                               evaluate, sn.assert_reference(0.9, 1, -0.2, -0.1))
        self.assertRaisesRegex(SanityError,
                               r'invalid low threshold value: 0\.2',
                               evaluate, sn.assert_reference(0.9, 1, 0.2, 0.1))
        self.assertRaisesRegex(SanityError,
                               r'invalid low threshold value: 1\.2',
                               evaluate, sn.assert_reference(0.9, 1, 1.2, 0.1))

        # check invalid thresholds greater than 1
        self.assertRaisesRegex(SanityError,
                               r'invalid low threshold value: -2\.0',
                               evaluate, sn.assert_reference(0.9, 1, -2.0, 0.1))
        self.assertRaisesRegex(SanityError,
                               r'invalid high threshold value: 1\.5',
                               evaluate, sn.assert_reference(-1.5, -1, -0.5, 1.5))

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
        self.assertTrue(sn.assert_found(r'Step: \d+', tempfile))
        self.assertTrue(sn.assert_found(
            r'Step: \d+', make_deferrable(tempfile)))
        self.assertRaises(SanityError, evaluate,
                          sn.assert_found(r'foo: \d+', tempfile))
        os.remove(tempfile)

    def test_assert_found_encoding(self):
        self.assertTrue(
            sn.assert_found('Odyssey', self.utf16_file, encoding='utf-16')
        )

    def test_assert_not_found(self):
        tempfile = self._write_tempfile()
        self.assertTrue(sn.assert_not_found(r'foo: \d+', tempfile))
        self.assertTrue(
            sn.assert_not_found(r'foo: \d+', make_deferrable(tempfile))
        )
        self.assertRaises(SanityError, evaluate,
                          sn.assert_not_found(r'Step: \d+', tempfile))
        os.remove(tempfile)

    def test_assert_not_found_encoding(self):
        self.assertTrue(
            sn.assert_not_found(r'Iliad', self.utf16_file, encoding='utf-16')
        )


class TestUtilityFunctions(unittest.TestCase):
    def test_getitem(self):
        l = [1, 2, 3]
        d = {'a': 1, 'b': 2, 'c': 3}

        self.assertEqual(2, sn.getitem(l, 1))
        self.assertEqual(2, sn.getitem(d, 'b'))
        self.assertRaisesRegex(SanityError, 'index out of bounds: 10',
                               evaluate, sn.getitem(l, 10))
        self.assertRaisesRegex(SanityError, 'key not found: k',
                               evaluate, sn.getitem(d, 'k'))

    def test_getitem_with_deferrables(self):
        l = make_deferrable([1, 2, 3])
        d = make_deferrable({'a': 1, 'b': 2, 'c': 3})

        self.assertEqual(2, sn.getitem(l, 1))
        self.assertEqual(2, sn.getitem(d, 'b'))
        self.assertRaisesRegex(SanityError, 'index out of bounds: 10',
                               evaluate, sn.getitem(l, 10))
        self.assertRaisesRegex(SanityError, 'key not found: k',
                               evaluate, sn.getitem(d, 'k'))

    def test_count(self):
        # Use a custom generator for testing
        def myrange(n):
            for i in range(n):
                yield i

        self.assertEqual(3, sn.count([1, 2, 3]))
        self.assertEqual(3, sn.count((1, 2, 3)))
        self.assertEqual(3, sn.count({1, 2, 3}))
        self.assertEqual(3, sn.count({'a': 1, 'b': 2, 'c': 3}))
        self.assertEqual(3, sn.count(range(3)))
        self.assertEqual(3, sn.count(myrange(3)))

        # Test empty sequences
        self.assertEqual(0, sn.count([]))
        self.assertEqual(0, sn.count({}))
        self.assertEqual(0, sn.count(set()))
        self.assertEqual(0, sn.count(range(0)))
        self.assertEqual(0, sn.count(myrange(0)))

    def test_count_uniq(self):
        # Use a custom generator for testing
        def my_mod_range(n, mod=2):
            for i in range(n):
                yield i % mod

        self.assertEqual(4, sn.count_uniq([1, 2, 3, 4, 4, 3, 2, 1]))
        self.assertEqual(1, sn.count_uniq((1, 1, 1)))
        self.assertEqual(3, sn.count_uniq({1, 2, 3, 2, 3}))
        self.assertEqual(3, sn.count_uniq({'a': 1, 'b': 2, 'c': 3}))
        self.assertEqual(2, sn.count_uniq(my_mod_range(10)))
        self.assertEqual(3, sn.count_uniq(my_mod_range(10, 3)))

        # Test empty sequences
        self.assertEqual(0, sn.count_uniq([]))
        self.assertEqual(0, sn.count_uniq({}))
        self.assertEqual(0, sn.count_uniq(set()))
        self.assertEqual(0, sn.count_uniq(my_mod_range(0)))
        self.assertEqual(0, sn.count_uniq(range(0)))

        # Test deferred expressions
        d = [1, 2, 2, 1]
        self.assertEqual(2, sn.count_uniq(make_deferrable(d)))

    def test_glob(self):
        filepatt = os.path.join(TEST_RESOURCES_CHECKS, '*.py')
        self.assertTrue(sn.glob(filepatt))
        self.assertTrue(sn.glob(make_deferrable(filepatt)))

    def test_iglob(self):
        filepatt = os.path.join(TEST_RESOURCES_CHECKS, '*.py')
        self.assertTrue(sn.count(sn.iglob(filepatt)))
        self.assertTrue(sn.count(sn.iglob(make_deferrable(filepatt))))

    def test_chain(self):
        list1 = ['A', 'B', 'C']
        list2 = ['D', 'E', 'F']
        chain1 = evaluate(sn.chain(make_deferrable(list1), list2))
        chain2 = itertools.chain(list1, list2)
        self.assertTrue(all((a == b for a, b in zip(chain1, chain2))))


class TestPatternMatchingFunctions(unittest.TestCase):
    def setUp(self):
        self.tempfile = None
        self.utf16_file = os.path.join(TEST_RESOURCES_CHECKS,
                                       'src', 'homer.txt')
        with NamedTemporaryFile('wt', delete=False) as fp:
            self.tempfile = fp.name
            fp.write('Step: 1\n')
            fp.write('Step: 2\n')
            fp.write('Step: 3\n')

    def tearDown(self):
        os.remove(self.tempfile)

    def test_findall(self):
        res = evaluate(sn.findall(r'Step: \d+', self.tempfile))
        self.assertEqual(3, len(res))

        res = evaluate(sn.findall('Step:.*', self.tempfile))
        self.assertEqual(3, len(res))

        res = evaluate(sn.findall('Step: [12]', self.tempfile))
        self.assertEqual(2, len(res))

        # Check the matches
        for expected, match in zip(['Step: 1', 'Step: 2'], res):
            self.assertEqual(expected, match.group(0))

        # Check groups
        res = evaluate(sn.findall(r'Step: (?P<no>\d+)', self.tempfile))
        for step, match in enumerate(res, start=1):
            self.assertEqual(step, int(match.group(1)))
            self.assertEqual(step, int(match.group('no')))

    def test_findall_encoding(self):
        res = evaluate(
            sn.findall('Odyssey', self.utf16_file, encoding='utf-16')
        )
        self.assertEqual(1, len(res))

    def test_findall_error(self):
        self.assertRaises(SanityError, evaluate,
                          sn.findall(r'Step: \d+', 'foo.txt'))

    def test_extractall(self):
        # Check numeric groups
        res = evaluate(sn.extractall(r'Step: (?P<no>\d+)', self.tempfile, 1))
        for expected, v in enumerate(res, start=1):
            self.assertEqual(str(expected), v)

        # Check named groups
        res = evaluate(sn.extractall(r'Step: (?P<no>\d+)', self.tempfile, 'no'))
        for expected, v in enumerate(res, start=1):
            self.assertEqual(str(expected), v)

        # Check convert function
        res = evaluate(sn.extractall(r'Step: (?P<no>\d+)',
                                     self.tempfile, 'no', int))
        for expected, v in enumerate(res, start=1):
            self.assertEqual(expected, v)

    def test_extractall_encoding(self):
        res = evaluate(
            sn.extractall('Odyssey', self.utf16_file, encoding='utf-16')
        )
        self.assertEqual(1, len(res))

    def test_extractall_error(self):
        self.assertRaises(SanityError, evaluate,
                          sn.extractall(r'Step: (\d+)', 'foo.txt', 1))
        self.assertRaises(
            SanityError, evaluate,
            sn.extractall(r'Step: (?P<no>\d+)',
                          self.tempfile, conv=int)
        )
        self.assertRaises(SanityError, evaluate,
                          sn.extractall(r'Step: (\d+)', self.tempfile, 2))
        self.assertRaises(
            SanityError, evaluate,
            sn.extractall(r'Step: (?P<no>\d+)', self.tempfile, 'foo'))

    def test_extractall_custom_conv(self):
        res = evaluate(sn.extractall(r'Step: (\d+)', self.tempfile, 1,
                                     lambda x: int(x)))
        for expected, v in enumerate(res, start=1):
            self.assertEqual(expected, v)

        # Check error in custom function
        self.assertRaises(
            SanityError, evaluate,
            sn.extractall(r'Step: (\d+)', self.tempfile,
                          conv=lambda x: int(x))
        )

        # Check error with a callable object
        class C:
            def __call__(self, x):
                return int(x)

        self.assertRaises(
            SanityError, evaluate,
            sn.extractall(r'Step: (\d+)', self.tempfile, conv=C())
        )

    def test_extractsingle(self):
        for i in range(1, 4):
            self.assertEqual(
                i,
                sn.extractsingle(r'Step: (\d+)', self.tempfile, 1, int, i-1)
            )

        # Test out of bounds access
        self.assertRaises(
            SanityError, evaluate,
            sn.extractsingle(r'Step: (\d+)', self.tempfile, 1, int, 100)
        )

    def test_extractsingle_encoding(self):
        res = evaluate(
            sn.extractsingle(r'Odyssey', self.utf16_file, encoding='utf-16')
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
        res = evaluate(sn.avg([1, 2, 3, 4]))
        self.assertEqual(2.5, res)

        # Check result when passing a generator
        res = evaluate(sn.avg(range(1, 5)))
        self.assertEqual(2.5, res)

        # Check with single element container
        res = evaluate(sn.avg(range(1, 2)))
        self.assertEqual(1, res)

        # Check with empty container
        self.assertRaises(SanityError, evaluate, sn.avg([]))
