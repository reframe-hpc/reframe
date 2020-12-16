# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

from unittests.resources.checks.paramcheck import (NoParams, TwoParams,
                                                   Abstract, ExtendParams)


def test_param_space_is_empty():
    class MyTest(NoParams):
        pass

    assert MyTest._rfm_params == {}


def test_params_are_present():
    class MyTest(TwoParams):
        pass

    assert MyTest._rfm_params['P0'] == ('a',)
    assert MyTest._rfm_params['P1'] == ('b',)


def test_param_override():
    class MyTest(TwoParams):
        parameter('P1', '-')

    assert MyTest._rfm_params['P0'] == ('a',)
    assert MyTest._rfm_params['P1'] == ('-',)


def test_param_inheritance():
    class MyTest(TwoParams):
        parameter('P1', 'c', inherit_params=True)

    assert MyTest._rfm_params['P0'] == ('a',)
    assert MyTest._rfm_params['P1'] == ('b', 'c',)


def test_filter_params():
    class MyTest(ExtendParams):
        parameter('P1', inherit_params=True, filter_params=lambda x: x[2:])

    assert MyTest._rfm_params['P0'] == ('a',)
    assert MyTest._rfm_params['P1'] == ('d', 'e',)
    assert MyTest._rfm_params['P2'] == ('f', 'g',)


def test_is_abstract_test():
    class MyTest(Abstract):
        pass

    assert MyTest.is_abstract_test()


def test_param_len_is_zero():
    class MyTest(Abstract):
        pass

    assert MyTest.param_space_len() == 0;


def test_extended_param_len():
    class MyTest(ExtendParams):
        pass

    assert MyTest.param_space_len() == 8


def test_param_iterator_is_empty():
    class MyTest(Abstract):
        @classmethod
        def check_param_iterator(cls):
            try:
                tmp = next(cls._rfm_param_space_iter)
                return False

            except StopIteration:
                return True

            except:
                return False

    MyTest.prepare_param_space()
    assert MyTest.check_param_iterator()


def test_params_are_none():
    class MyTest(Abstract):
        pass

    test = MyTest()
    assert test.P0 == None
    assert test.P1 == None

    class MyTest(TwoParams):
        pass

    test = MyTest()
    assert test.P0 == None
    assert test.P1 == None


def test_param_values_are_not_set():
    class MyTest(Abstract):
        pass

    MyTest.prepare_param_space()
    test = MyTest()
    assert test.P0 == None
    assert test.P1 == None


def test_param_values_are_set():
    class MyTest(TwoParams):
        pass

    MyTest.prepare_param_space()
    test = MyTest()
    assert test.P0 == 'a'
    assert test.P1 == 'b'


def test_extended_params():
    class MyTest(ExtendParams):
        pass

    test = MyTest()
    assert hasattr(test, 'P0')
    assert hasattr(test, 'P1')
    assert hasattr(test, 'P2')


def test_extended_params_are_set():
    class MyTest(ExtendParams):
        pass

    MyTest.prepare_param_space()
    for i in range(MyTest.param_space_len()):
        test = MyTest()
        assert test.P0 != None
        assert test.P1 != None
        assert test.P2 != None

    test = MyTest()
    assert test.P0 == None
    assert test.P1 == None
    assert test.P2 == None

