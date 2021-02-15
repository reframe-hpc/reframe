# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause


import pytest


import reframe as rfm
from reframe.core.fields import Field


@pytest.fixture
def NoVarsTest():
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
    with pytest.raises(ValueError):
        class MyTest(rfm.RegressionTest):
            v0 = variable(int, value=2)
            v0 = 2


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


def test_require_undeclared_variable(NoVarsTest):
    with pytest.raises(ValueError):
        class MyTest(NoVarsTest):
            foo = required


def test_invalid_field():
    class Foo:
        '''An invalid descriptor'''

    with pytest.raises(ValueError):
        class MyTest(rfm.RegressionTest):
            a = variable(int, value=4, field=Foo)
