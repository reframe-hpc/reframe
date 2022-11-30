# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import reframe.utility.typecheck as types


def assert_type_hierarchy(builtin_type, ctype):
    assert isinstance(ctype, type)
    assert issubclass(builtin_type, ctype)
    assert issubclass(ctype[int], ctype)
    assert issubclass(ctype[ctype[int]], ctype)
    assert not issubclass(ctype[int], ctype[float])
    assert not issubclass(ctype[ctype[int]], ctype[int])
    assert not issubclass(ctype[int], ctype[ctype[int]])


def test_bool_type():
    assert isinstance(True, types.Bool)
    assert isinstance(False, types.Bool)
    assert not isinstance('foo', types.Bool)

    # Test invalid arguments
    with pytest.raises(TypeError):
        types.Bool('foo')

    with pytest.raises(TypeError):
        types.Bool('True')

    with pytest.raises(TypeError):
        types.Bool('False')

    # Test for boolean conversion
    assert types.Bool('true') is True
    assert types.Bool('yes') is True
    assert types.Bool('false') is False
    assert types.Bool('no') is False


def test_list_type():
    l = [1, 2]
    ll = [[1, 2], [3, 4]]
    assert isinstance(l, types.List)
    assert isinstance(l, types.List[int])
    assert not isinstance(l, types.List[float])

    assert isinstance(ll, types.List)
    assert isinstance(ll, types.List[types.List[int]])
    assert_type_hierarchy(list, types.List)

    # Test invalid arguments
    with pytest.raises(TypeError):
        types.List[3]

    with pytest.raises(TypeError):
        types.List[int, float]

    # Test type conversions
    assert types.List[int]('1,2') == [1, 2]
    assert types.List[int]('1') == [1]

    with pytest.raises(ValueError):
        types.List[int]('foo')

    with pytest.raises(TypeError):
        types.List[int](1)


def test_set_type():
    s = {1, 2}
    ls = [{1, 2}, {3, 4}]
    assert isinstance(s, types.Set)
    assert isinstance(s, types.Set[int])
    assert not isinstance(s, types.Set[float])

    assert isinstance(ls, types.List)
    assert isinstance(ls, types.List[types.Set[int]])
    assert_type_hierarchy(set, types.Set)

    # Test invalid arguments
    with pytest.raises(TypeError):
        types.Set[3]

    with pytest.raises(TypeError):
        types.Set[int, float]

    assert types.Set[int]('1,2') == {1, 2}
    assert types.Set[int]('1') == {1}

    with pytest.raises(ValueError):
        types.Set[int]('foo')

    with pytest.raises(TypeError):
        types.Set[int](1)


def test_uniform_tuple_type():
    t = (1, 2)
    tl = ([1, 2], [3, 4])
    lt = [(1, 2), (3, 4)]
    assert isinstance(t, types.Tuple)
    assert isinstance(t, types.Tuple[int])
    assert not isinstance(t, types.Tuple[float])

    assert isinstance(tl, types.Tuple)
    assert isinstance(tl, types.Tuple[types.List[int]])

    assert isinstance(lt, types.List)
    assert isinstance(lt, types.List[types.Tuple[int]])
    assert_type_hierarchy(tuple, types.Tuple)

    # Test invalid arguments
    with pytest.raises(TypeError):
        types.Set[3]

    assert types.Tuple[int]('1,2') == (1, 2)
    assert types.Tuple[int]('1') == (1,)

    with pytest.raises(ValueError):
        types.Tuple[int]('foo')

    with pytest.raises(TypeError):
        types.Tuple[int](1)


def test_non_uniform_tuple_type():
    t = (1, 2.3, '4', ['a', 'b'])
    assert isinstance(t, types.Tuple)
    assert isinstance(t, types.Tuple[int, float, str, types.List[str]])
    assert not isinstance(t, types.Tuple[float, int, str])
    assert not isinstance(t, types.Tuple[float, int, str, int])

    # Test invalid arguments
    with pytest.raises(TypeError):
        types.Set[int, 3]

    assert types.Tuple[int, str]('1,2') == (1, '2')

    with pytest.raises(TypeError):
        types.Tuple[int, str]('1')

    with pytest.raises(TypeError):
        types.Tuple[int, str](1)


def test_mapping_type():
    d = {'one': 1, 'two': 2}
    dl = {'one': [1, 2], 'two': [3, 4]}
    ld = [{'one': 1, 'two': 2}, {'three': 3}]
    assert isinstance(d, types.Dict)
    assert isinstance(d, types.Dict[str, int])
    assert not isinstance(d, types.Dict[int, int])

    assert isinstance(dl, types.Dict)
    assert isinstance(dl, types.Dict[str, types.List[int]])
    assert isinstance(ld, types.List[types.Dict[str, int]])

    # Test invalid arguments
    with pytest.raises(TypeError):
        types.Dict[int]

    with pytest.raises(TypeError):
        types.Dict[int, 3]

    # Test conversions
    assert types.Dict[str, int]('a:1,b:2') == {'a': 1, 'b': 2}

    with pytest.raises(TypeError):
        types.Dict[str, int]('a:1,b')


def test_str_type():
    s = '123'
    ls = ['1', '23', '456']
    assert isinstance(s, types.Str)
    assert isinstance(s, types.Str[r'\d+'])
    assert isinstance(ls, types.List[types.Str[r'\d+']])
    assert not isinstance(s, types.Str[r'a.*'])
    assert not isinstance(ls, types.List[types.Str[r'a.*']])
    assert not isinstance('hello, world', types.Str[r'\S+'])

    # Test invalid arguments
    with pytest.raises(TypeError):
        types.Str[int]

    # Test conversion
    typ = types.Str[r'\d+']
    assert typ('10') == '10'

    with pytest.raises(TypeError):
        types.Str[r'\d+'](1)


def test_type_names():
    assert 'List' == types.List.__name__
    assert 'List[int]' == types.List[int].__name__
    assert ('Dict[str,List[int]]' ==
            types.Dict[str, types.List[int]].__name__)
    assert ('Tuple[int,Set[float],str]' ==
            types.Tuple[int, types.Set[float], str].__name__)
    assert r"List[Str[r'\d+']]" == types.List[types.Str[r'\d+']].__name__


def test_custom_types():
    class C:
        def __init__(self, v):
            self.__v = v

        def __hash__(self):
            return hash(self.__v)

    l = [C(0), C(1), C(2)]
    d = {0: C(0), 1: C(1)}
    cd = {C(0): 0, C(1): 1}
    t = (0, C(0), '1')
    assert isinstance(l, types.List[C])
    assert isinstance(d, types.Dict[int, C])
    assert isinstance(cd, types.Dict[C, int])
    assert isinstance(t, types.Tuple[int, C, str])


def test_custom_types_conversion():
    class X(metaclass=types.ConvertibleType):
        def __init__(self, x):
            self.x = x

        @classmethod
        def __rfm_cast_str__(cls, s):
            return X(int(s))

    class Y:
        def __init__(self, s):
            self.y = int(s)

    class Z:
        def __init__(self, x, y):
            self.z = x + y

    assert X('3').x == 3
    assert X(3).x   == 3
    assert X(x='foo').x == 'foo'

    with pytest.raises(TypeError):
        X(3, 4)

    with pytest.raises(TypeError):
        X(s=3)

    assert Y('1').y == 1
    assert Z(5, 3).z  == 8


def test_composition_of_types():
    t = types.Integer | types.Float
    assert isinstance(1, t)
    assert isinstance(1.2, t)
    assert not isinstance([1], t)

    t = ~types.Float
    assert isinstance(1, t)
    assert not isinstance(1.2, t)
    assert isinstance([1], t)

    t = types.Integer | types.List[types.Integer | types.Float]
    assert isinstance(1, t)
    assert not isinstance(1.2, t)
    assert isinstance([1], t)
    assert isinstance([1.2], t)
    assert not isinstance(None, t)

    # Check the other direction
    t = types.List[types.Integer | types.Float] | types.Integer
    assert isinstance(1, t)
    assert not isinstance(1.2, t)
    assert isinstance([1], t)
    assert isinstance([1.2], t)
    assert not isinstance(None, t)

    # Test custom types
    class T:
        pass

    MetaT = types.make_meta_type('MetaT', T)

    t = T()
    assert isinstance(t, T)
    assert isinstance(t, MetaT)
    assert isinstance(1, MetaT | types.Integer)
    assert isinstance(1, ~MetaT)
