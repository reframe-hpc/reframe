# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import io
import itertools
import os
import pytest
import sys


import reframe.utility.sanity as sn
from reframe.core.exceptions import SanityError
from unittests.fixtures import TEST_RESOURCES_CHECKS


@pytest.fixture
def fixt():
    class _Fixture:
        def __init__(self):
            self._a = 0
            self._b = 1

        # Property retrieval must also be deferred, if we want lazy evaluation
        # of their values. The following demonstrates two different ways to
        # achieve this. Notice that the `getattr` is actually a call to
        # `reframe.utility.sanity.getattr`, since we have imported everything.

        @property
        @sn.sanity_function
        def a(self):
            return self._a

        @property
        def b(self):
            return sn.getattr(self, '_b')

    return _Fixture()


def test_abs():
    assert 1.0 == sn.abs(1.0)
    assert 0.0 == sn.abs(0.0)
    assert 1.0 == sn.abs(-1.0)
    assert 2.0 == sn.abs(sn.defer(-2.0))


def test_and(fixt):
    expr = sn.and_(fixt.a, fixt.b)
    fixt._a = 1
    fixt._b = 1

    assert expr
    assert not sn.not_(expr)


def test_or(fixt):
    expr = sn.or_(fixt.a, fixt.b)
    fixt._a = 0
    fixt._b = 0
    assert not expr
    assert sn.not_(expr)


def test_all():
    l = [1, 1, 0]
    expr = sn.all(l)
    l[2] = 1
    assert expr


def test_allx():
    assert sn.allx([1, 1, 1])
    assert not sn.allx([1, 0])
    assert not sn.allx([])
    assert not sn.allx(i for i in range(0))
    assert sn.allx(i for i in range(1, 2))
    with pytest.raises(TypeError):
        sn.evaluate(sn.allx(None))


def test_any():
    l = [0, 0, 1]
    expr = sn.any(l)
    l[2] = 0
    assert not expr


def test_enumerate():
    de = sn.enumerate(sn.defer([1, 2]), start=1)
    for i, e in de:
        assert i == e


def test_filter():
    df = sn.filter(lambda x: x if x % 2 else None,
                   sn.defer([1, 2, 3, 4, 5]))

    for i, x in sn.enumerate(df, start=1):
        assert 2*i - 1 == x

    # Alternative testing
    assert [1, 3, 5] == list(sn.evaluate(df))


def test_hasattr(fixt):
    e = sn.hasattr(fixt, '_c')
    assert not e

    fixt._c = 1
    assert e


def test_len():
    l = [1, 2]
    dl = sn.defer(l)
    assert 2 == sn.len(dl)

    l.append(3)
    assert 3 == sn.len(dl)


def test_map():
    l = [1, 2, 3]
    dm = sn.map(lambda x: 2*x + 1, l)
    for i, x in sn.enumerate(dm, start=1):
        assert 2*i + 1 == x

    # Alternative test
    assert [3, 5, 7] == list(sn.evaluate(dm))


def test_max():
    l = [1, 2]
    dl = sn.defer(l)
    assert 2 == sn.max(dl)

    l.append(3)
    assert 3 == sn.max(dl)


def test_min():
    l = [1, 2]
    dl = sn.defer(l)
    assert 1 == sn.min(dl)

    l.append(0)
    assert 0 == sn.min(dl)


def test_print_stdout():
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        x, y = sn.evaluate(sn.print(1, sn.defer(2)))

    assert stdout.getvalue() == '1 2\n'
    assert x == 1
    assert y == 2


def test_print_stderr():
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        x, y = sn.evaluate(sn.print(1, sn.defer(2), file=sys.stderr))

    assert stderr.getvalue() == '1 2\n'
    assert x == 1
    assert y == 2


def test_print_separator():
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        x, y = sn.evaluate(sn.print(1, sn.defer(2), sep='|'))

    assert stdout.getvalue() == '1|2\n'
    assert x == 1
    assert y == 2


def test_print_end():
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        x, y = sn.evaluate(sn.print(1, sn.defer(2), end=''))

    assert stdout.getvalue() == '1 2'
    assert x == 1
    assert y == 2


def test_reversed():
    l = [1, 2, 3]
    dr = sn.reversed(l)
    assert [3, 2, 1] == list(sn.evaluate(dr))


def test_round():
    assert 1.0 == sn.round(sn.defer(1.4))


def test_setattr(fixt):
    dset = sn.setattr(fixt, '_a', 5)
    assert 0 == fixt._a
    sn.evaluate(dset)
    assert 5 == fixt._a


def test_sorted():
    l = [2, 3, 1]
    ds = sn.sorted(l)
    assert [1, 2, 3] == list(sn.evaluate(ds))


def test_sum():
    assert 3 == sn.sum([1, 1, 1])
    assert 3 == sn.sum(sn.defer([1, 1, 1]))


def test_zip():
    la = [1, 2, 3]
    lb = sn.defer(['a', 'b', 'c'])

    la_new = []
    lb_new = []
    for a, b in sn.zip(la, lb):
        la_new.append(a)
        lb_new.append(b)

    assert [1, 2, 3] == la_new
    assert ['a', 'b', 'c'] == lb_new


@pytest.fixture
def utf16_file():
    return os.path.join(TEST_RESOURCES_CHECKS, 'src', 'homer.txt')


def test_assert_true():
    assert sn.assert_true(True)
    assert sn.assert_true(1)
    assert sn.assert_true([1])
    assert sn.assert_true(range(1))
    with pytest.raises(SanityError, match='False is not True'):
        sn.evaluate(sn.assert_true(False))

    with pytest.raises(SanityError, match='0 is not True'):
        sn.evaluate(sn.assert_true(0))

    with pytest.raises(SanityError, match=r'\[\] is not True'):
        sn.evaluate(sn.assert_true([]))

    with pytest.raises(SanityError, match=r'range\(.+\) is not True'):
        sn.evaluate(sn.assert_true(range(0)))

    with pytest.raises(SanityError, match='not true'):
        sn.evaluate(sn.assert_true(0, msg='not true'))

def test_assert_true_with_deferrables():
    assert sn.assert_true(sn.defer(True))
    assert sn.assert_true(sn.defer(1))
    assert sn.assert_true(sn.defer([1]))
    with pytest.raises(SanityError, match='False is not True'):
        sn.evaluate(sn.assert_true(sn.defer(False)))

    with pytest.raises(SanityError, match='0 is not True'):
        sn.evaluate(sn.assert_true(sn.defer(0)))

    with pytest.raises(SanityError, match=r'\[\] is not True'):
        sn.evaluate(sn.assert_true(sn.defer([])))

def test_assert_false():
    assert sn.assert_false(False)
    assert sn.assert_false(0)
    assert sn.assert_false([])
    assert sn.assert_false(range(0))
    with pytest.raises(SanityError, match='True is not False'):
        sn.evaluate(sn.assert_false(True))

    with pytest.raises(SanityError, match='1 is not False'):
        sn.evaluate(sn.assert_false(1))

    with pytest.raises(SanityError, match=r'\[1\] is not False'):
        sn.evaluate(sn.assert_false([1]))

    with pytest.raises(SanityError, match=r'range\(.+\) is not False'):
        sn.evaluate(sn.assert_false(range(1)))


def test_assert_false_with_deferrables():
    assert sn.assert_false(sn.defer(False))
    assert sn.assert_false(sn.defer(0))
    assert sn.assert_false(sn.defer([]))
    with pytest.raises(SanityError, match='True is not False'):
        sn.evaluate(sn.assert_false(sn.defer(True)))

    with pytest.raises(SanityError, match='1 is not False'):
        sn.evaluate(sn.assert_false(sn.defer(1)))

    with pytest.raises(SanityError, match=r'\[1\] is not False'):
        sn.evaluate(sn.assert_false(sn.defer([1])))


def test_assert_eq():
    assert sn.assert_eq(1, 1)
    assert sn.assert_eq(1, True)
    with pytest.raises(SanityError, match='1 != 2'):
        sn.evaluate(sn.assert_eq(1, 2))

    with pytest.raises(SanityError, match='1 != False'):
        sn.evaluate(sn.assert_eq(1, False))

    with pytest.raises(SanityError, match='1 is not equals to 2'):
        sn.evaluate(sn.assert_eq(1, 2, '{0} is not equals to {1}'))


def test_assert_eq_with_deferrables():
    assert sn.assert_eq(1, sn.defer(1))
    assert sn.assert_eq(sn.defer(1), True)
    with pytest.raises(SanityError, match='1 != 2'):
        sn.evaluate(sn.assert_eq(sn.defer(1), 2))

    with pytest.raises(SanityError, match='1 != False'):
        sn.evaluate(sn.assert_eq(sn.defer(1), False))


def test_assert_ne():
    assert sn.assert_ne(1, 2)
    assert sn.assert_ne(1, False)
    with pytest.raises(SanityError, match='1 == 1'):
        sn.evaluate(sn.assert_ne(1, 1))

    with pytest.raises(SanityError, match='1 == True'):
        sn.evaluate(sn.assert_ne(1, True))


def test_assert_ne_with_deferrables():
    assert sn.assert_ne(1, sn.defer(2))
    assert sn.assert_ne(sn.defer(1), False)
    with pytest.raises(SanityError, match='1 == 1'):
        sn.evaluate(sn.assert_ne(sn.defer(1), 1))

    with pytest.raises(SanityError, match='1 == True'):
        sn.evaluate(sn.assert_ne(sn.defer(1), True))


def test_assert_gt():
    assert sn.assert_gt(3, 1)
    with pytest.raises(SanityError, match='1 <= 3'):
        sn.evaluate(sn.assert_gt(1, 3))


def test_assert_gt_with_deferrables():
    assert sn.assert_gt(3, sn.defer(1))
    with pytest.raises(SanityError, match='1 <= 3'):
        sn.evaluate(sn.assert_gt(1, sn.defer(3)))


def test_assert_ge():
    assert sn.assert_ge(3, 1)
    assert sn.assert_ge(3, 3)
    with pytest.raises(SanityError, match='1 < 3'):
        sn.evaluate(sn.assert_ge(1, 3))


def test_assert_ge_with_deferrables():
    assert sn.assert_ge(3, sn.defer(1))
    assert sn.assert_ge(3, sn.defer(3))
    with pytest.raises(SanityError, match='1 < 3'):
        sn.evaluate(sn.assert_ge(1, sn.defer(3)))


def test_assert_lt():
    assert sn.assert_lt(1, 3)
    with pytest.raises(SanityError, match='3 >= 1'):
        sn.evaluate(sn.assert_lt(3, 1))


def test_assert_lt_with_deferrables():
    assert sn.assert_lt(1, sn.defer(3))
    with pytest.raises(SanityError, match='3 >= 1'):
        sn.evaluate(sn.assert_lt(3, sn.defer(1)))


def test_assert_le():
    assert sn.assert_le(1, 1)
    assert sn.assert_le(3, 3)
    with pytest.raises(SanityError, match='3 > 1'):
        sn.evaluate(sn.assert_le(3, 1))


def test_assert_le_with_deferrables():
    assert sn.assert_le(1, sn.defer(3))
    assert sn.assert_le(3, sn.defer(3))
    with pytest.raises(SanityError, match='3 > 1'):
        sn.evaluate(sn.assert_le(3, sn.defer(1)))


def test_assert_in():
    assert sn.assert_in(1, [1, 2, 3])
    with pytest.raises(SanityError, match=r'0 is not in \[1, 2, 3\]'):
        sn.evaluate(sn.assert_in(0, [1, 2, 3]))


def test_assert_in_with_deferrables():
    assert sn.assert_in(1, sn.defer([1, 2, 3]))
    with pytest.raises(SanityError, match=r'0 is not in \[1, 2, 3\]'):
        sn.evaluate(sn.assert_in(0, sn.defer([1, 2, 3])))


def test_assert_not_in():
    assert sn.assert_not_in(0, [1, 2, 3])
    with pytest.raises(SanityError, match=r'1 is in \[1, 2, 3\]'):
        sn.evaluate(sn.assert_not_in(1, [1, 2, 3]))


def test_assert_not_in_with_deferrables():
    assert sn.assert_not_in(0, sn.defer([1, 2, 3]))
    with pytest.raises(SanityError, match=r'1 is in \[1, 2, 3\]'):
        sn.evaluate(sn.assert_not_in(1, sn.defer([1, 2, 3])))


def test_assert_bounded():
    assert sn.assert_bounded(1, -1.5, 1.5)
    assert sn.assert_bounded(1, upper=1.5)
    assert sn.assert_bounded(1, lower=-1.5)
    assert sn.assert_bounded(1)
    with pytest.raises(SanityError,
                       match=r'value 1 not within bounds -0\.5\.\.0\.5'):
        sn.evaluate(sn.assert_bounded(1, -0.5, 0.5))

    with pytest.raises(SanityError,
                       match=r'value 1 not within bounds -inf\.\.0\.5'):
        sn.evaluate(sn.assert_bounded(1, upper=0.5))

    with pytest.raises(SanityError,
                       match=r'value 1 not within bounds 1\.5\.\.inf'):
        sn.evaluate(sn.assert_bounded(1, lower=1.5))

    with pytest.raises(SanityError, match='value 1 is out of bounds'):
        sn.evaluate(sn.assert_bounded(
            1, -0.5, 0.5, 'value {0} is out of bounds'))


def test_assert_reference():
    assert sn.assert_reference(0.9, 1, -0.2, 0.1)
    assert sn.assert_reference(0.9, 1, upper_thres=0.1)
    assert sn.assert_reference(0.9, 1, lower_thres=-0.2)
    assert sn.assert_reference(0.9, 1)

    # Check negatives
    assert sn.assert_reference(-0.9, -1, -0.2, 0.1)
    assert sn.assert_reference(-0.9, -1, -0.2)
    assert sn.assert_reference(-0.9, -1, upper_thres=0.1)
    assert sn.assert_reference(-0.9, -1)

    # Check upper threshold values greater than 1
    assert sn.assert_reference(20.0, 10.0, None, 3.0)
    assert sn.assert_reference(-50.0, -20.0, -2.0, 0.5)
    with pytest.raises(SanityError, match=r'0\.5 is beyond reference value 1 '
                                          r'\(l=0\.8, u=1\.1\)'):
        sn.evaluate(sn.assert_reference(0.5, 1, -0.2, 0.1))

    with pytest.raises(SanityError, match=r'0\.5 is beyond reference value 1 '
                                          r'\(l=0\.8, u=inf\)'):
        sn.evaluate(sn.assert_reference(0.5, 1, -0.2))

    with pytest.raises(SanityError, match=r'1\.5 is beyond reference value 1 '
                                          r'\(l=0\.8, u=1\.1\)'):
        sn.evaluate(sn.assert_reference(1.5, 1, -0.2, 0.1))

    with pytest.raises(SanityError, match=r'1\.5 is beyond reference value 1 '
                                          r'\(l=-inf, u=1\.1\)'):
        sn.evaluate(sn.assert_reference(1.5, 1, lower_thres=None,
                                        upper_thres=0.1))

    with pytest.raises(SanityError,
                       match=r'-0\.8 is beyond reference value -1 '
                             r'\(l=-1\.2, u=-0\.9\)'):
        sn.evaluate(sn.assert_reference(-0.8, -1, -0.2, 0.1))

    # Check invalid thresholds
    with pytest.raises(SanityError,
                       match=r'invalid high threshold value: -0\.1'):
        sn.evaluate(sn.assert_reference(0.9, 1, -0.2, -0.1))

    with pytest.raises(SanityError,
                       match=r'invalid low threshold value: 0\.2'):
        sn.evaluate(sn.assert_reference(0.9, 1, 0.2, 0.1))

    with pytest.raises(SanityError,
                       match=r'invalid low threshold value: 1\.2'):
        sn.evaluate(sn.assert_reference(0.9, 1, 1.2, 0.1))

    # check invalid thresholds greater than 1
    with pytest.raises(SanityError,
                       match=r'invalid low threshold value: -2\.0'):
        sn.evaluate(sn.assert_reference(0.9, 1, -2.0, 0.1))

    with pytest.raises(SanityError,
                       match=r'invalid high threshold value: 1\.5'):
        sn.evaluate(sn.assert_reference(-1.5, -1, -0.5, 1.5))


@pytest.fixture
def tempfile(tmp_path):
     tmp_file = tmp_path / 'tempfile'
     with open(tmp_file, 'w') as fp:
         fp.write('Step: 1\n')
         fp.write('Step: 2\n')
         fp.write('Step: 3\n')

     return str(tmp_file)


def test_assert_found(tempfile):
    assert sn.assert_found(r'Step: \d+', tempfile)
    assert sn.assert_found(r'Step: \d+', sn.defer(tempfile))
    with pytest.raises(SanityError):
        sn.evaluate(sn.assert_found(r'foo: \d+', tempfile))


def test_assert_found_encoding(utf16_file):
    assert sn.assert_found('Odyssey', utf16_file, encoding='utf-16')


def test_assert_not_found(tempfile):
    assert sn.assert_not_found(r'foo: \d+', tempfile)
    assert sn.assert_not_found(r'foo: \d+', sn.defer(tempfile))
    with pytest.raises(SanityError):
        sn.evaluate(sn.assert_not_found(r'Step: \d+', tempfile))


def test_assert_not_found_encoding(utf16_file):
    assert sn.assert_not_found(r'Iliad', utf16_file, encoding='utf-16')


def test_getitem():
    l = [1, 2, 3]
    d = {'a': 1, 'b': 2, 'c': 3}

    assert 2 == sn.getitem(l, 1)
    assert 2 == sn.getitem(d, 'b')
    with pytest.raises(SanityError, match='index out of bounds: 10'):
        sn.evaluate(sn.getitem(l, 10))

    with pytest.raises(SanityError, match='key not found: k'):
        sn.evaluate(sn.getitem(d, 'k'))


def test_getitem_with_deferrables():
    l = sn.defer([1, 2, 3])
    d = sn.defer({'a': 1, 'b': 2, 'c': 3})

    assert 2 == sn.getitem(l, 1)
    assert 2 == sn.getitem(d, 'b')
    with pytest.raises(SanityError, match='index out of bounds: 10'):
        sn.evaluate(sn.getitem(l, 10))

    with pytest.raises(SanityError, match='key not found: k'):
        sn.evaluate(sn.getitem(d, 'k'))


def test_count():
    # Use a custom generator for testing
    def myrange(n):
        for i in range(n):
            yield i

    assert 3 == sn.count([1, 2, 3])
    assert 3 == sn.count((1, 2, 3))
    assert 3 == sn.count({1, 2, 3})
    assert 3 == sn.count({'a': 1, 'b': 2, 'c': 3})
    assert 3 == sn.count(range(3))
    assert 3 == sn.count(myrange(3))

    # Test empty sequences
    assert 0 == sn.count([])
    assert 0 == sn.count({})
    assert 0 == sn.count(set())
    assert 0 == sn.count(range(0))
    assert 0 == sn.count(myrange(0))


def test_count_uniq():
    # Use a custom generator for testing
    def my_mod_range(n, mod=2):
        for i in range(n):
            yield i % mod

    assert 4 == sn.count_uniq([1, 2, 3, 4, 4, 3, 2, 1])
    assert 1 == sn.count_uniq((1, 1, 1))
    assert 3 == sn.count_uniq({1, 2, 3, 2, 3})
    assert 3 == sn.count_uniq({'a': 1, 'b': 2, 'c': 3})
    assert 2 == sn.count_uniq(my_mod_range(10))
    assert 3 == sn.count_uniq(my_mod_range(10, 3))

    # Test empty sequences
    assert 0 == sn.count_uniq([])
    assert 0 == sn.count_uniq({})
    assert 0 == sn.count_uniq(set())
    assert 0 == sn.count_uniq(my_mod_range(0))
    assert 0 == sn.count_uniq(range(0))

    # Test deferred expressions
    d = [1, 2, 2, 1]
    assert 2 == sn.count_uniq(sn.defer(d))


def test_glob():
    filepatt = os.path.join(TEST_RESOURCES_CHECKS, '*.py')
    assert sn.glob(filepatt)
    assert sn.glob(sn.defer(filepatt))


def test_iglob():
    filepatt = os.path.join(TEST_RESOURCES_CHECKS, '*.py')
    assert sn.count(sn.iglob(filepatt))
    assert sn.count(sn.iglob(sn.defer(filepatt)))


def test_chain():
    list1 = ['A', 'B', 'C']
    list2 = ['D', 'E', 'F']
    chain1 = sn.evaluate(sn.chain(sn.defer(list1), list2))
    chain2 = itertools.chain(list1, list2)
    assert all((a == b for a, b in zip(chain1, chain2)))





def test_findall(tempfile):
    res = sn.evaluate(sn.findall(r'Step: \d+', tempfile))
    assert 3 == len(res)

    res = sn.evaluate(sn.findall('Step:.*', tempfile))
    assert 3 == len(res)

    res = sn.evaluate(sn.findall('Step: [12]', tempfile))
    assert 2 == len(res)

    # Check the matches
    for expected, match in zip(['Step: 1', 'Step: 2'], res):
        assert expected == match.group(0)

    # Check groups
    res = sn.evaluate(sn.findall(r'Step: (?P<no>\d+)', tempfile))
    for step, match in enumerate(res, start=1):
        assert step == int(match.group(1))
        assert step == int(match.group('no'))


def test_findall_encoding(utf16_file):
    res = sn.evaluate(
        sn.findall('Odyssey', utf16_file, encoding='utf-16')
    )
    assert 1 == len(res)


def test_findall_error():
    with pytest.raises(SanityError):
        sn.evaluate(sn.findall(r'Step: \d+', 'foo.txt'))


def test_extractall(tempfile):
    # Check numeric groups
    res = sn.evaluate(sn.extractall(r'Step: (?P<no>\d+)', tempfile, 1))
    for expected, v in enumerate(res, start=1):
        assert str(expected) == v

    # Check named groups
    res = sn.evaluate(sn.extractall(r'Step: (?P<no>\d+)', tempfile, 'no'))
    for expected, v in enumerate(res, start=1):
        assert str(expected) == v

    # Check convert function
    res = sn.evaluate(sn.extractall(r'Step: (?P<no>\d+)', tempfile, 'no', int))
    for expected, v in enumerate(res, start=1):
        assert expected == v


def test_extractall_encoding(utf16_file):
    res = sn.evaluate(sn.extractall('Odyssey', utf16_file, encoding='utf-16'))
    assert 1 == len(res)


def test_extractall_error(tempfile):
    with pytest.raises(SanityError):
        sn.evaluate(sn.extractall(r'Step: (\d+)', 'foo.txt', 1))

    with pytest.raises(SanityError):
        sn.evaluate(sn.extractall(r'Step: (?P<no>\d+)', tempfile, conv=int))

    with pytest.raises(SanityError):
        sn.evaluate(sn.extractall(r'Step: (\d+)', tempfile, 2))

    with pytest.raises(SanityError):
        sn.evaluate(sn.extractall(r'Step: (?P<no>\d+)', tempfile, 'foo'))


def test_extractall_custom_conv(tempfile):
    res = sn.evaluate(sn.extractall(r'Step: (\d+)', tempfile, 1,
                                    lambda x: int(x)))
    for expected, v in enumerate(res, start=1):
        assert expected == v

    # Check error in custom function
    with pytest.raises(SanityError):
        sn.evaluate(sn.extractall(r'Step: (\d+)', tempfile,
                                  conv=lambda x: int(x)))

    # Check error with a callable object
    class C:
        def __call__(self, x):
            return int(x)

    with pytest.raises(SanityError):
        sn.evaluate(sn.extractall(r'Step: (\d+)', tempfile, conv=C()))


def test_extractsingle(tempfile):
    for i in range(1, 4):
        assert i == sn.extractsingle(r'Step: (\d+)', tempfile, 1, int, i-1)

    # Test out of bounds access
    with pytest.raises(SanityError):
        sn.evaluate(sn.extractsingle(r'Step: (\d+)', tempfile, 1, int, 100))


def test_extractsingle_encoding(utf16_file):
    res = sn.evaluate(
        sn.extractsingle(r'Odyssey', utf16_file, encoding='utf-16')
    )
    assert -1 != res.find('Odyssey')


def test_safe_format():
    from reframe.utility.sanity import _format

    s = 'There is {0} and {1}.'
    assert s == _format(s)
    assert s == _format(s, 'bacon')
    assert 'There is egg and spam.' == _format(s, 'egg', 'spam')
    assert 'There is egg and bacon.' == _format(s, 'egg', 'bacon', 'spam')

    s = 'no placeholders'
    assert s == _format(s)
    assert s == _format(s, 'bacon')


def test_avg():
    res = sn.evaluate(sn.avg([1, 2, 3, 4]))
    assert 2.5 == res

    # Check result when passing a generator
    res = sn.evaluate(sn.avg(range(1, 5)))
    assert 2.5 == res

    # Check with single element container
    res = sn.evaluate(sn.avg(range(1, 2)))
    assert 1 == res

    # Check with empty container
    with pytest.raises(SanityError):
        sn.evaluate(sn.avg([]))
