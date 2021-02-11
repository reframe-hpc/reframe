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
        var('foo', int, value=10)

    yield OneVarTest


def test_custom_var(OneVarTest):
    assert not hasattr(OneVarTest, 'foo')
    inst = OneVarTest()
    assert hasattr(OneVarTest, 'foo')
    assert isinstance(OneVarTest.foo, Field)
    assert hasattr(inst, 'foo')
    assert inst.foo == 10


def test_instantiate_and_inherit(NoVarsTest):
    inst = NoVarsTest()
    with pytest.raises(ValueError):
        class MyTest(NoVarsTest):
            '''Error from name clashing'''


def test_redeclare_builtin_var_clash(NoVarsTest):
    with pytest.raises(ValueError):
        class MyTest(NoVarsTest):
            var('name', str)


def test_redeclare_var_clash(OneVarTest):
    with pytest.raises(ValueError):
        class MyTest(OneVarTest):
            var('foo', str)


def test_inheritance_clash(NoVarsTest):
    class MyMixin(rfm.RegressionMixin):
        var('name', str)

    with pytest.raises(ValueError):
        class MyTest(NoVarsTest, MyMixin):
            '''Trigger error from inheritance clash.'''


def test_var_space_clash():
    class Spam(rfm.RegressionMixin):
        var('v0', int, value=1)

    class Ham(rfm.RegressionMixin):
        var('v0', int, value=2)

    with pytest.raises(ValueError):
        class Eggs(Spam, Ham):
            '''Trigger error from var name clashing.'''


def test_double_declare():
    with pytest.raises(ValueError):
        class MyTest(rfm.RegressionTest):
            var('v0', int, value=1)
            var('v0', float, value=0.5)


def test_double_action_on_var():
    with pytest.raises(ValueError):
        class MyTest(rfm.RegressionTest):
            set_var('v0', 2)
            var('v0', int, value=2)


def test_namespace_clash(NoVarsTest):
    with pytest.raises(ValueError):
        class MyTest(NoVarsTest):
            var('current_environ', str)


def test_set_var(OneVarTest):
    class MyTest(OneVarTest):
        set_var('foo', 4)

    inst = MyTest()
    assert not hasattr(OneVarTest, 'foo')
    assert hasattr(MyTest, 'foo')
    assert hasattr(inst, 'foo')
    assert inst.foo == 4


def test_var_type(OneVarTest):
    class MyTest(OneVarTest):
        set_var('foo', 'bananas')

    with pytest.raises(TypeError):
        inst = MyTest()


def test_set_undef(NoVarsTest):
    with pytest.raises(ValueError):
        class MyTest(NoVarsTest):
            set_var('foo', 4)


def test_require_var(OneVarTest):
    class MyTest(OneVarTest):
        require_var('foo')

        def __init__(self):
            print(self.foo)

    with pytest.raises(AttributeError):
        inst = MyTest()


def test_required_var_not_present(OneVarTest):
    class MyTest(OneVarTest):
        require_var('foo')

        def __init__(self):
            pass

    mytest = MyTest()


def test_require_undeclared_var(NoVarsTest):
    with pytest.raises(ValueError):
        class MyTest(NoVarsTest):
            require_var('foo')


def test_invalid_field():
    class Foo:
        '''An invalid descriptor'''

    with pytest.raises(ValueError):
        class MyTest(rfm.RegressionTest):
            var('a', int, value=4, field=Foo)
