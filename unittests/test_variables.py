# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause


import pytest


import reframe as rfm
from reframe.core.fields import Field


@pytest.fixture
def novars():
    class NoVars(rfm.RegressionTest):
        pass
    yield NoVars


@pytest.fixture
def onevar(novars):
    class OneVar(novars):
        var('foo', int, value=10)
    yield OneVar


def test_custom_var(onevar):
    assert not hasattr(onevar, 'foo')
    inst = onevar()
    assert hasattr(onevar, 'foo')
    assert isinstance(onevar.foo, Field)
    assert hasattr(inst, 'foo')
    assert inst.foo == 10


def test_instantiate_and_inherit(novars):
    inst = novars()
    with pytest.raises(NameError):
        class MyTest(novars):
            pass


def test_redeclare_var_clash(novars):
    with pytest.raises(ValueError):
        class MyTest(novars):
            var('name', str)


def test_inheritance_clash(novars):
    class MyMixin(rfm.RegressionMixin):
        var('name', str)

    with pytest.raises(ValueError):
        class MyTest(novars, MyMixin):
            pass


def test_namespace_clash(novars):
    with pytest.raises(NameError):
        class MyTest(novars):
            var('current_environ', str)


def test_set_var(onevar):
    class MyTest(onevar):
        set_var('foo', 4)

    inst = MyTest()
    assert not hasattr(onevar, 'foo')
    assert hasattr(MyTest, 'foo')
    assert hasattr(inst, 'foo')
    assert inst.foo == 4


def test_var_type(onevar):
    class MyTest(onevar):
        set_var('foo', 'bananas')

    with pytest.raises(TypeError):
        inst = MyTest()


def test_set_undef(novars):
    with pytest.raises(ValueError):
        class MyTest(novars):
            set_var('foo', 4)


def test_require_var(onevar):
    class MyTest(onevar):
        require_var('foo')

        def __init__(self):
            print(self.foo)

    with pytest.raises(AttributeError):
        inst = MyTest()


def test_required_var_not_present(onevar):
    class MyTest(onevar):
        require_var('foo')

        def __init__(self):
            pass

    try:
        inst = MyTest()
    except Exception:
        pytest.fail('class instantiation failed')


def test_require_undef(novars):
    with pytest.raises(ValueError):
        class MyTest(novars):
            require_var('foo')
