# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause


import pytest
import inspect


import reframe as rfm
from reframe.core.exceptions import ReframeSyntaxError


class NoParams(rfm.RunOnlyRegressionTest):
    pass


class TwoParams(NoParams):
    P0 = parameter(['a'])
    P1 = parameter(['b'])


class Abstract(TwoParams):
    '''An abstract test is a test with undefined parameters.'''
    P0 = parameter()


class ExtendParams(TwoParams):
    P1 = parameter(['c', 'd', 'e'], inherit_params=True)
    P2 = parameter(['f', 'g'])


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
        P1 = parameter(['-'])

    assert MyTest.param_space['P0'] == ('a',)
    assert MyTest.param_space['P1'] == ('-',)


def test_param_inheritance():
    class MyTest(TwoParams):
        P1 = parameter(['c'], inherit_params=True)

    assert MyTest.param_space['P0'] == ('a',)
    assert MyTest.param_space['P1'] == ('b', 'c',)


def test_filter_params():
    class MyTest(ExtendParams):
        # We make the return type of the filtering function a list to ensure
        # that other iterables different to tuples are also valid.
        P1 = parameter(inherit_params=True,
                       filter_params=lambda x: list(x[2:]))

    assert MyTest.param_space['P0'] == ('a',)
    assert MyTest.param_space['P1'] == ('d', 'e',)
    assert MyTest.param_space['P2'] == ('f', 'g',)


def test_wrong_filter():
    with pytest.raises(TypeError):
        class MyTest(ExtendParams):
            '''Filter function is not a function'''
            P1 = parameter(inherit_params=True, filter_params='not callable')

    with pytest.raises(TypeError):
        class MyTest(ExtendParams):
            '''Filter function takes more than 1 argument'''
            P1 = parameter(inherit_params=True, filter_params=lambda x, y: [])

    def bad_filter(x):
        raise RuntimeError('bad filter')

    with pytest.raises(RuntimeError):
        class Foo(ExtendParams):
            '''Filter function raises'''
            P1 = parameter(inherit_params=True, filter_params=bad_filter)

    with pytest.raises(ReframeSyntaxError):
        class Foo(ExtendParams):
            '''Wrong filter return type'''
            P1 = parameter(inherit_params=True, filter_params=lambda x: 1)


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

    assert len(MyTest.param_space) == 0


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
        pass

    for i in range(MyTest.num_variants):
        test = MyTest(variant_num=i)
        assert test.P0 is not None
        assert test.P1 is not None
        assert test.P2 is not None

    test = MyTest()
    assert test.P0 is None
    assert test.P1 is None
    assert test.P2 is None

    with pytest.raises(ValueError):
        test = MyTest(variant_num=i+1)


def test_inject_params_wrong_index():
    class MyTest(ExtendParams):
        pass

    inst = MyTest()
    with pytest.raises(RuntimeError):
        MyTest.param_space.inject(inst, params_index=MyTest.num_variants+1)


def test_simple_test_decorator():
    @rfm.simple_test
    class MyTest(ExtendParams):
        pass

    mod = inspect.getmodule(MyTest)
    tests = mod._rfm_test_registry.instantiate_all()
    assert len(tests) == 8
    for test in tests:
        assert test.P0 is not None
        assert test.P1 is not None
        assert test.P2 is not None


def test_param_space_clash():
    class Spam(rfm.RegressionMixin):
        P0 = parameter([1])

    class Ham(rfm.RegressionMixin):
        P0 = parameter([2])

    with pytest.raises(ReframeSyntaxError):
        class Eggs(Spam, Ham):
            '''Trigger error from param name clashing.'''


def test_multiple_inheritance():
    class Spam(rfm.RegressionMixin):
        P0 = parameter()

    class Ham(rfm.RegressionMixin):
        P0 = parameter([2])

    class Eggs(Spam, Ham):
        '''Multiple inheritance is allowed if only one class defines P0.'''


def test_namespace_clash():
    class Spam(rfm.RegressionTest):
        foo = variable(int)

    with pytest.raises(ReframeSyntaxError):
        class Ham(Spam):
            foo = parameter([1])


def test_double_declare():
    with pytest.raises(ReframeSyntaxError):
        class MyTest(rfm.RegressionTest):
            P0 = parameter([1, 2, 3])
            P0 = parameter()


def test_overwrite_param():
    with pytest.raises(ReframeSyntaxError):
        class MyTest(TwoParams):
            P0 = [1, 2, 3]


def test_param_deepcopy():
    '''Test that there is no cross-class pollution.

    Each instance must deal with its own copies of the parameters.
    '''
    class MyParam:
        def __init__(self, val):
            self.val = val

    class Base(rfm.RegressionTest):
        p0 = parameter([MyParam(1), MyParam(2)])

    class Foo(Base):
        def __init__(self):
            self.p0.val = -20

    class Bar(Base):
        pass

    assert Foo(variant_num=0).p0.val == -20
    assert Foo(variant_num=1).p0.val == -20
    assert Bar(variant_num=0).p0.val == 1
    assert Bar(variant_num=1).p0.val == 2


def test_param_access():
    with pytest.raises(ReframeSyntaxError):
        class Foo(rfm.RegressionTest):
            p = parameter([1, 2, 3])
            x = f'accessing {p!r} in the class body is disallowed.'


def test_param_space_read_only():
    class Foo(rfm.RegressionMixin):
        pass

    with pytest.raises(ReframeSyntaxError):
        Foo.param_space['a'] = (1, 2, 3)


def test_parameter_override():
    with pytest.raises(ReframeSyntaxError):
        # Trigger the check from MetaNamespace
        class MyTest(rfm.RegressionTest):
            p = parameter([1, 2])
            p = 0

    # Trigger the check in the extend method from ParameterSpace
    class Foo(rfm.RegressionTest):
        p = parameter([1, 2])

    with pytest.raises(ReframeSyntaxError):
        class Bar(Foo):
            p = 0


def test_override_regular_attribute():
    class Foo(rfm.RegressionTest):
        p = 4
        p = parameter([1, 2])

    assert Foo.p.values == (1, 2,)


def test_override_parameter():
    with pytest.raises(ReframeSyntaxError):
        class Foo(rfm.RegressionTest):
            p = parameter([1, 2])
            p = 1

    with pytest.raises(ReframeSyntaxError):
        class Foo(rfm.RegressionTest):
            p = parameter([1, 2])
            p += 1


def test_local_paramspace_is_empty():
    class MyTest(rfm.RegressionTest):
        p = parameter([1, 2, 3])

    assert len(MyTest._rfm_local_param_space) == 0


def test_class_attr_access():
    class MyTest(rfm.RegressionTest):
        p = parameter([1, 2, 3])

    assert MyTest.p.values == (1, 2, 3,)
    with pytest.raises(ReframeSyntaxError, match='cannot override parameter'):
        MyTest.p = (4, 5,)


def test_get_variant_nums():
    class MyTest(rfm.RegressionTest):
        p = parameter(range(10))

    with pytest.raises(NameError):
        MyTest.param_space.get_variant_nums(p0=lambda x: x == 2)

    with pytest.raises(ValueError):
        MyTest.param_space.get_variant_nums(p=lambda x, y: x == 2)
