# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause


import pytest
import inspect

import reframe as rfm


class NoParams(rfm.RunOnlyRegressionTest):
    pass


class TwoParams(NoParams):
    parameter('P0', 'a')
    parameter('P1', 'b')


class Abstract(TwoParams):
    parameter('P0')


class ExtendParams(TwoParams):
    parameter('P1', 'c', 'd', 'e', inherit_params=True)
    parameter('P2', 'f', 'g')


def test_param_space_is_empty():
    class MyTest(NoParams):
        pass

    assert MyTest.param_space.is_empty()


def test_params_are_present():
    class MyTest(TwoParams):
        pass

    assert MyTest.param_space['P0'] == ('a',)
    assert MyTest.param_space['P1'] == ('b',)


def test_abstract_param():
    class MyTest(Abstract):
        pass

    assert MyTest.param_space['P0'] == ()
    assert MyTest.param_space['P1'] == ('b',)


def test_param_override():
    class MyTest(TwoParams):
        parameter('P1', '-')

    assert MyTest.param_space['P0'] == ('a',)
    assert MyTest.param_space['P1'] == ('-',)


def test_param_inheritance():
    class MyTest(TwoParams):
        parameter('P1', 'c', inherit_params=True)

    assert MyTest.param_space['P0'] == ('a',)
    assert MyTest.param_space['P1'] == ('b', 'c',)


def test_filter_params():
    class MyTest(ExtendParams):
        parameter('P1', inherit_params=True, filter_params=lambda x: x[2:])

    assert MyTest.param_space['P0'] == ('a',)
    assert MyTest.param_space['P1'] == ('d', 'e',)
    assert MyTest.param_space['P2'] == ('f', 'g',)


def test_is_abstract_test():
    class MyTest(Abstract):
        pass

    assert MyTest.is_abstract()


def test_is_not_abstract_test():
    class MyTest(TwoParams):
        pass

    assert not MyTest.is_abstract()


def test_param_len_is_zero():
    class MyTest(Abstract):
        pass

    assert len(MyTest.param_space) == 0;


def test_extended_param_len():
    class MyTest(ExtendParams):
        pass

    assert len(MyTest.param_space) == 8


def test_instantiate_abstract_test():
    class MyTest(Abstract):
        pass

    test = MyTest()
    assert test.P0 is None
    assert test.P1 is None


def test_param_values_are_not_set():
    class MyTest(TwoParams):
        pass

    test = MyTest()
    assert test.P0 is None
    assert test.P1 is None


def test_consume_param_space():
    class MyTest(ExtendParams):
        def __init__(self):
            pass

    for _ in MyTest.param_space:
        test = MyTest(_rfm_use_params=True)
        assert test.P0 is not None
        assert test.P1 is not None
        assert test.P2 is not None

    test = MyTest()
    assert test.P0 is None
    assert test.P1 is None
    assert test.P2 is None

    with pytest.raises(StopIteration):
        test = MyTest(_rfm_use_params=True)


def test_register_abstract_test():
    with pytest.raises(ValueError):
        @rfm.simple_test
        class MyTest(Abstract):
            pass


def test_simple_test_decorator():
    @rfm.simple_test
    class MyTest(ExtendParams):
        pass

    mod = inspect.getmodule(MyTest)
    tests = mod._rfm_gettests()
    assert len(tests) == 8
    for test in tests:
        assert test.P0 is not None
        assert test.P1 is not None
        assert test.P2 is not None
