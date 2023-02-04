# Copyright 2016-2023 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import reframe.utility.typecheck as typ


def assert_type_hierarchy(builtin_type, ctype):
    assert isinstance(ctype, type)
    assert issubclass(builtin_type, ctype)
    assert issubclass(ctype[int], ctype)
    assert issubclass(ctype[ctype[int]], ctype)
    assert not issubclass(ctype[int], ctype[float])
    assert not issubclass(ctype[ctype[int]], ctype[int])
    assert not issubclass(ctype[int], ctype[ctype[int]])


def test_bool_type():
    assert isinstance(True, typ.Bool)
    assert isinstance(False, typ.Bool)
    assert not isinstance('foo', typ.Bool)

    # Test invalid arguments
    with pytest.raises(TypeError):
        typ.Bool('foo')

    with pytest.raises(TypeError):
        typ.Bool('True')

    with pytest.raises(TypeError):
        typ.Bool('False')

    # Test for boolean conversion
    assert typ.Bool('true') is True
    assert typ.Bool('yes') is True
    assert typ.Bool('false') is False
    assert typ.Bool('no') is False


def test_list_type():
    l = [1, 2]
    ll = [[1, 2], [3, 4]]
    assert isinstance(l, typ.List)
    assert isinstance(l, typ.List[int])
    assert not isinstance(l, typ.List[float])

    assert isinstance(ll, typ.List)
    assert isinstance(ll, typ.List[typ.List[int]])
    assert_type_hierarchy(list, typ.List)

    # Test invalid arguments
    with pytest.raises(TypeError):
        typ.List[3]

    with pytest.raises(TypeError):
        typ.List[int, float]

    # Test type conversions
    assert typ.List[int]('1,2') == [1, 2]
    assert typ.List[int]('1') == [1]

    with pytest.raises(ValueError):
        typ.List[int]('foo')

    with pytest.raises(TypeError):
        typ.List[int](1)


def test_set_type():
    s = {1, 2}
    ls = [{1, 2}, {3, 4}]
    assert isinstance(s, typ.Set)
    assert isinstance(s, typ.Set[int])
    assert not isinstance(s, typ.Set[float])

    assert isinstance(ls, typ.List)
    assert isinstance(ls, typ.List[typ.Set[int]])
    assert_type_hierarchy(set, typ.Set)

    # Test invalid arguments
    with pytest.raises(TypeError):
        typ.Set[3]

    with pytest.raises(TypeError):
        typ.Set[int, float]

    assert typ.Set[int]('1,2') == {1, 2}
    assert typ.Set[int]('1') == {1}

    with pytest.raises(ValueError):
        typ.Set[int]('foo')

    with pytest.raises(TypeError):
        typ.Set[int](1)


def test_uniform_tuple_type():
    t = (1, 2)
    tl = ([1, 2], [3, 4])
    lt = [(1, 2), (3, 4)]
    assert isinstance(t, typ.Tuple)
    assert isinstance(t, typ.Tuple[int])
    assert not isinstance(t, typ.Tuple[float])

    assert isinstance(tl, typ.Tuple)
    assert isinstance(tl, typ.Tuple[typ.List[int]])

    assert isinstance(lt, typ.List)
    assert isinstance(lt, typ.List[typ.Tuple[int]])
    assert_type_hierarchy(tuple, typ.Tuple)

    # Test invalid arguments
    with pytest.raises(TypeError):
        typ.Set[3]

    assert typ.Tuple[int]('1,2') == (1, 2)
    assert typ.Tuple[int]('1') == (1,)

    with pytest.raises(ValueError):
        typ.Tuple[int]('foo')

    with pytest.raises(TypeError):
        typ.Tuple[int](1)


def test_non_uniform_tuple_type():
    t = (1, 2.3, '4', ['a', 'b'])
    assert isinstance(t, typ.Tuple)
    assert isinstance(t, typ.Tuple[int, float, str, typ.List[str]])
    assert not isinstance(t, typ.Tuple[float, int, str])
    assert not isinstance(t, typ.Tuple[float, int, str, int])

    # Test invalid arguments
    with pytest.raises(TypeError):
        typ.Set[int, 3]

    assert typ.Tuple[int, str]('1,2') == (1, '2')

    with pytest.raises(TypeError):
        typ.Tuple[int, str]('1')

    with pytest.raises(TypeError):
        typ.Tuple[int, str](1)


def test_mapping_type():
    d = {'one': 1, 'two': 2}
    dl = {'one': [1, 2], 'two': [3, 4]}
    ld = [{'one': 1, 'two': 2}, {'three': 3}]
    assert isinstance(d, typ.Dict)
    assert isinstance(d, typ.Dict[str, int])
    assert not isinstance(d, typ.Dict[int, int])

    assert isinstance(dl, typ.Dict)
    assert isinstance(dl, typ.Dict[str, typ.List[int]])
    assert isinstance(ld, typ.List[typ.Dict[str, int]])

    # Test invalid arguments
    with pytest.raises(TypeError):
        typ.Dict[int]

    with pytest.raises(TypeError):
        typ.Dict[int, 3]

    # Test conversions
    assert typ.Dict[str, int]('a:1,b:2') == {'a': 1, 'b': 2}

    with pytest.raises(TypeError):
        typ.Dict[str, int]('a:1,b')


def test_str_type():
    s = '123'
    ls = ['1', '23', '456']
    assert isinstance(s, typ.Str)
    assert isinstance(s, typ.Str[r'\d+'])
    assert isinstance(ls, typ.List[typ.Str[r'\d+']])
    assert not isinstance(s, typ.Str[r'a.*'])
    assert not isinstance(ls, typ.List[typ.Str[r'a.*']])
    assert not isinstance('hello, world', typ.Str[r'\S+'])

    # Test invalid arguments
    with pytest.raises(TypeError):
        typ.Str[int]

    # Test conversion
    str_type = typ.Str[r'\d+']
    assert str_type('10') == '10'

    with pytest.raises(TypeError):
        typ.Str[r'\d+'](1)


def test_type_names():
    assert 'List' == typ.List.__name__
    assert 'List[int]' == typ.List[int].__name__
    assert ('Dict[str,List[int]]' ==
            typ.Dict[str, typ.List[int]].__name__)
    assert ('Tuple[int,Set[float],str]' ==
            typ.Tuple[int, typ.Set[float], str].__name__)
    assert r"List[Str[r'\d+']]" == typ.List[typ.Str[r'\d+']].__name__


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
    assert isinstance(l, typ.List[C])
    assert isinstance(d, typ.Dict[int, C])
    assert isinstance(cd, typ.Dict[C, int])
    assert isinstance(t, typ.Tuple[int, C, str])


def test_custom_types_conversion():
    class X(metaclass=typ.ConvertibleType):
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
    t = typ.Integer | typ.Float
    assert isinstance(1, t)
    assert isinstance(1.2, t)
    assert not isinstance([1], t)

    t = ~typ.Float
    assert isinstance(1, t)
    assert not isinstance(1.2, t)
    assert isinstance([1], t)

    # Test the sequence types
    type_pairs = [
        (typ.List, list),
        (typ.Set, set),
        (typ.Tuple, tuple)
    ]
    for meta_seq_type, seq_type in type_pairs:
        composite_types = [
            typ.Integer | meta_seq_type[typ.Integer | typ.Float | typ.Bool],
            typ.Integer | meta_seq_type[typ.Float | typ.Integer | typ.Bool],
            typ.Integer | meta_seq_type[typ.Bool | typ.Integer | typ.Float],
            meta_seq_type[typ.Integer | typ.Float | typ.Bool] | typ.Integer,
            meta_seq_type[typ.Float | typ.Integer | typ.Bool] | typ.Integer,
            meta_seq_type[typ.Bool | typ.Integer | typ.Float] | typ.Integer
        ]
        for t in composite_types:
            assert isinstance(1, t)
            assert not isinstance(1.2, t)
            assert isinstance(True, t)
            assert not isinstance(None, t)
            assert isinstance(seq_type([1]), t)
            assert isinstance(seq_type([1.2]), t)
            assert isinstance(seq_type([False]), t)

    # Test the Str type
    composite_types = [typ.Str[r'\d+'] | typ.Integer,
                       typ.Integer | typ.Str[r'\d+']]
    for t in composite_types:
        assert isinstance(1, t)
        assert isinstance('1', t)
        assert not isinstance([1], t)
        assert not isinstance(['1'], t)

    # Test the Dict type
    composite_types = [
        typ.Dict[typ.Str | typ.Integer, typ.Integer] | typ.Integer,
        typ.Integer | typ.Dict[typ.Str | typ.Integer, typ.Integer],
    ]
    for t in composite_types:
        assert isinstance({1: 1}, t)
        assert isinstance({'1': 1}, t)
        assert isinstance({1: [1]}, t)
        assert isinstance({'1': 1.2}, t)

    # Test custom types

    class T:
        pass

    MetaT = typ.make_meta_type('MetaT', T)

    t = T()
    assert isinstance(t, T)
    assert isinstance(t, MetaT)
    assert isinstance(1, MetaT | typ.Integer)
    assert isinstance(1, ~MetaT)
