# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import unittest

import reframe.utility.typecheck as types


class TestTypes(unittest.TestCase):
    def _test_type_hierarchy(self, builtin_type, ctype):
        assert issubclass(builtin_type, ctype)
        assert issubclass(ctype[int], ctype)
        assert issubclass(ctype[ctype[int]], ctype)
        assert not issubclass(ctype[int], ctype[float])
        assert not issubclass(ctype[ctype[int]], ctype[int])
        assert not issubclass(ctype[int], ctype[ctype[int]])

    def test_list_type(self):
        l = [1, 2]
        ll = [[1, 2], [3, 4]]
        assert isinstance(l, types.List)
        assert isinstance(l, types.List[int])
        assert not isinstance(l, types.List[float])

        assert isinstance(ll, types.List)
        assert isinstance(ll, types.List[types.List[int]])
        self._test_type_hierarchy(list, types.List)

        # Test invalid arguments
        with pytest.raises(TypeError):
            types.List[3]

        with pytest.raises(TypeError):
            types.List[int, float]

    def test_set_type(self):
        s = {1, 2}
        ls = [{1, 2}, {3, 4}]
        assert isinstance(s, types.Set)
        assert isinstance(s, types.Set[int])
        assert not isinstance(s, types.Set[float])

        assert isinstance(ls, types.List)
        assert isinstance(ls, types.List[types.Set[int]])
        self._test_type_hierarchy(set, types.Set)

        # Test invalid arguments
        with pytest.raises(TypeError):
            types.Set[3]

        with pytest.raises(TypeError):
            types.Set[int, float]

    def test_uniform_tuple_type(self):
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
        self._test_type_hierarchy(tuple, types.Tuple)

        # Test invalid arguments
        with pytest.raises(TypeError):
            types.Set[3]

    def test_non_uniform_tuple_type(self):
        t = (1, 2.3, '4', ['a', 'b'])
        assert isinstance(t, types.Tuple)
        assert isinstance(t, types.Tuple[int, float, str, types.List[str]])
        assert not isinstance(t, types.Tuple[float, int, str])
        assert not isinstance(t, types.Tuple[float, int, str, int])

        # Test invalid arguments
        with pytest.raises(TypeError):
            types.Set[int, 3]

    def test_mapping_type(self):
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

    def test_str_type(self):
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

    def test_type_names(self):
        assert 'List' == types.List.__name__
        assert 'List[int]' == types.List[int].__name__
        assert ('Dict[str,List[int]]' ==
                types.Dict[str, types.List[int]].__name__)
        assert ('Tuple[int,Set[float],str]' ==
                types.Tuple[int, types.Set[float], str].__name__)
        assert r"List[Str[r'\d+']]" == types.List[types.Str[r'\d+']].__name__

    def test_custom_types(self):
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
