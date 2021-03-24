# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause


import pytest
import math

import reframe as rfm
from reframe.core.fields import Field


@pytest.fixture
def NoVarsTest():
    '''Variables are injected as descriptors in the classes.

    Thus, fixtures are needed to provide a fresh class to each test.
    '''
    class NoVarsTest(rfm.RegressionTest):
        pass

    yield NoVarsTest


@pytest.fixture
def OneVarTest(NoVarsTest):
    class OneVarTest(NoVarsTest):
        foo = variable(int, value=10)

    yield OneVarTest


def test_custom_variable(OneVarTest):
    assert hasattr(OneVarTest, 'foo')
    assert not isinstance(OneVarTest.foo, Field)
    inst = OneVarTest()
    assert hasattr(OneVarTest, 'foo')
    assert isinstance(OneVarTest.foo, Field)
    assert hasattr(inst, 'foo')
    assert inst.foo == 10


def test_redeclare_builtin_var_clash(NoVarsTest):
    with pytest.raises(ValueError):
        class MyTest(NoVarsTest):
            name = variable(str)


def test_name_clash_builtin_property(NoVarsTest):
    with pytest.raises(ValueError):
        class MyTest(NoVarsTest):
            current_environ = variable(str)


def test_redeclare_var_clash(OneVarTest):
    with pytest.raises(ValueError):
        class MyTest(OneVarTest):
            foo = variable(str)


def test_inheritance_clash(NoVarsTest):
    class MyMixin(rfm.RegressionMixin):
        name = variable(str)

    with pytest.raises(ValueError):
        class MyTest(NoVarsTest, MyMixin):
            '''Trigger error from inheritance clash.'''


def test_instantiate_and_inherit(OneVarTest):
    '''Instantiation will inject the vars as class attributes.

    Ensure that inheriting from this class after the instantiation does not
    raise a namespace clash with the vars.
    '''
    inst = OneVarTest()

    class MyTest(OneVarTest):
        pass


def test_var_space_clash():
    class Spam(rfm.RegressionMixin):
        v0 = variable(int, value=1)

    class Ham(rfm.RegressionMixin):
        v0 = variable(int, value=2)

    with pytest.raises(ValueError):
        class Eggs(Spam, Ham):
            '''Trigger error from var name clashing.'''


def test_double_declare():
    with pytest.raises(ValueError):
        class MyTest(rfm.RegressionTest):
            v0 = variable(int, value=1)
            v0 = variable(float, value=0.5)


def test_double_action_on_variable():
    '''Modifying a variable in the class body is permitted.'''
    class MyTest(rfm.RegressionTest):
        v0 = variable(int, value=2)
        v0 += 2

    assert MyTest.v0 == 4


def test_set_var(OneVarTest):
    class MyTest(OneVarTest):
        foo = 4

    inst = MyTest()
    assert hasattr(OneVarTest, 'foo')
    assert not isinstance(OneVarTest.foo, Field)
    assert hasattr(MyTest, 'foo')
    assert isinstance(MyTest.foo, Field)
    assert hasattr(inst, 'foo')
    assert inst.foo == 4


def test_var_type(OneVarTest):
    class MyTest(OneVarTest):
        foo = 'bananas'

    with pytest.raises(TypeError):
        inst = MyTest()


def test_require_var(OneVarTest):
    class MyTest(OneVarTest):
        foo = required

        def __init__(self):
            print(self.foo)

    with pytest.raises(AttributeError):
        inst = MyTest()


def test_required_var_not_present(OneVarTest):
    class MyTest(OneVarTest):
        foo = required

        def __init__(self):
            pass

    mytest = MyTest()


def test_invalid_field():
    class Foo:
        '''An invalid descriptor'''

    with pytest.raises(ValueError):
        class MyTest(rfm.RegressionTest):
            a = variable(int, value=4, field=Foo)


def test_var_deepcopy():
    '''Test that there is no cross-class pollution.

    Each instance must have its own copies of each variable.
    '''
    class Base(rfm.RegressionTest):
        my_var = variable(list, value=[1, 2])

    class Foo(Base):
        def __init__(self):
            self.my_var += [3]

    class Bar(Base):
        pass

    class Baz(Base):
        my_var = []

    assert Base().my_var == [1, 2]
    assert Foo().my_var == [1, 2, 3]
    assert Bar().my_var == [1, 2]
    assert Baz().my_var == []


def test_variable_access():
    class Foo(rfm.RegressionMixin):
        my_var = variable(str, value='bananas')
        x = f'accessing {my_var!r} works because it has a default value.'

    assert 'bananas' in getattr(Foo, 'x')
    with pytest.raises(ValueError):
        class Foo(rfm.RegressionMixin):
            my_var = variable(int)
            x = f'accessing {my_var!r} fails because its value is not set.'


def test_var_space_is_read_only():
    class Foo(rfm.RegressionMixin):
        pass

    with pytest.raises(ValueError):
        Foo._rfm_var_space['v'] = 0


def test_override_regular_attribute():
    class Foo(rfm.RegressionTest):
        v = 0
        v = variable(int, value=40)

    assert Foo.v == 40


def test_var_name_is_set():
    class MyTest (rfm.RegressionTest):
        v = variable(int)

    assert MyTest.v.name == 'v'


def test_variable_with_attribute():
    class Foo:
        pass

    class MyTest(rfm.RegressionTest):
        v = variable(Foo, value=Foo())
        v.my_attr = 'Injected attribute'

    assert MyTest().v.my_attr == 'Injected attribute'


def test_local_varspace_is_empty():
    class MyTest(rfm.RegressionTest):
        v = variable(int, value=0)

    assert len(MyTest._rfm_local_var_space) == 0


def test_upstream_var_fetching():
    class Foo(rfm.RegressionTest):
        v0 = variable(int, value=10)

    class Bar(Foo):
        v1 = variable(int, value=v0*2)

    assert Bar().v1 == 20


def test_var_basic_operators():
    class A(rfm.RegressionTest):
        v = variable(int, value=2)
        assert f'{v}' == '2'
        assert str(v) == '2'
        assert format(v) == format(2)
        assert bytes(v) == bytes(2)
        assert hash(v) == hash(2)
        assert bool(v) == bool(2)


def test_var_comp():
    class A(rfm.RegressionTest):
        v = variable(int, value=2)
        assert v < 3
        assert v <= 2
        assert v == 2
        assert v != 3
        assert v > 1
        assert v >= 2


def test_var_container_operators():
    class A(rfm.RegressionTest):
        v = variable(dict, value={'a': 1, 'b': 2})
        assert len(v) == 2
        assert v['a'] == 1
        v['c'] = 3
        assert len(v) == 3
        del v['c']
        assert len(v) == 2
        with pytest.raises(KeyError):
            vv = v['c']
        assert 'a' in v

        vv = variable(list, value=['a', 'b'])
        reversed_keys = ''
        for i in reversed(vv):
            reversed_keys += i
        assert reversed_keys == 'ba'
        iter_keys = ''
        for i in iter(vv):
            iter_keys += i
        assert iter_keys == 'ab'


def test_var_add_operator():
    class A(rfm.RegressionTest):
        v = variable(int, value=2)
        assert (v + 1) == 3
        assert (1 + v) == 3
        v += 1
        assert v == 3


def test_var_add_operator():
    class A(rfm.RegressionTest):
        v = variable(int, value=3)
        assert (v - 1) == 2
        assert (1 - v) == -2
        v -= 1
        assert v == 2


def test_var_mul_operator():
    class A(rfm.RegressionTest):
        v = variable(int, value=2)
        assert (v * 2) == 4
        assert (2 * v) == 4
        v *= 2
        assert v == 4


def test_var_div_operator():
    class A(rfm.RegressionTest):
        v = variable(int, value=4)
        assert (v / 3) == 4/3
        assert (3 / v) == 3/4
        v /= 2
        assert v == 2


def test_var_floordiv_operator():
    class A(rfm.RegressionTest):
        v = variable(int, value=7)
        assert (v // 2) == 3
        assert (15 // v) == 2
        v //= 2
        assert v == 3


def test_var_mod_operator():
    class A(rfm.RegressionTest):
        v = variable(int, value=4)
        assert (v % 2) == 0
        assert (3 % v) == 3
        v %= 2
        assert v == 0


def test_var_divmod_operator():
    class A(rfm.RegressionTest):
        v = variable(int, value=4)
        assert divmod(v, 2) == (2, 0)
        assert divmod(5, v) == (1, 1)


def test_var_pow_operator():
    class A(rfm.RegressionTest):
        v = variable(int, value=2)
        assert (v ** 2) == 4
        assert (3 ** v) == 9
        v **= 2
        assert v == 4


def test_var_lshift_operator():
    class A(rfm.RegressionTest):
        v = variable(int, value=2)
        assert (v << 1) == 4
        assert (1 << v) == 4
        v <<= 1
        assert v == 4


def test_var_rshift_operator():
    class A(rfm.RegressionTest):
        v = variable(int, value=8)
        assert (v >> 1) == 4
        assert (1024 >> v) == 4
        v >>= 1
        assert v == 4


def test_var_and_operator():
    class A(rfm.RegressionTest):
        v = variable(int, value=3)
        assert (v & 2) == 2
        assert (2 & v) == 2
        v &= 2
        assert v == 2


def test_var_or_operator():
    class A(rfm.RegressionTest):
        v = variable(int, value=1)
        assert (v | 2) == 3
        assert (2 | v) == 3
        v |= 2
        assert v == 3


def test_var_xor_operator():
    class A(rfm.RegressionTest):
        v = variable(int, value=4)
        assert (v ^ 1) == 5
        assert (2 ^ v) == 6
        v ^= 1
        assert v == 5


def test_other_numerical_operators():
    class A(rfm.RegressionTest):
        npi = variable(float, value=-3.141592)
        v = variable(int, value=2)
        assert -npi == 3.141592
        assert +npi == -3.141592
        assert abs(npi) == 3.141592
        assert ~v == -3
        assert int(npi) == -3
        assert float(v) == float(2)
        assert complex(v) == complex(2)
        assert round(npi, 4) == -3.1416
        assert math.trunc(npi) == -3
        assert math.floor(npi) == -4
        assert math.ceil(npi) == -3
