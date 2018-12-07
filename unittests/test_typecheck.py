import unittest
import reframe.utility.typecheck as types


class TestTypes(unittest.TestCase):
    def _test_type_hierarchy(self, builtin_type, ctype):
        self.assertTrue(issubclass(builtin_type, ctype))
        self.assertTrue(issubclass(ctype[int], ctype))
        self.assertTrue(issubclass(ctype[ctype[int]], ctype))
        self.assertFalse(issubclass(ctype[int], ctype[float]))
        self.assertFalse(issubclass(ctype[ctype[int]], ctype[int]))
        self.assertFalse(issubclass(ctype[int], ctype[ctype[int]]))

    def test_list_type(self):
        l = [1, 2]
        ll = [[1, 2], [3, 4]]
        self.assertIsInstance(l, types.List)
        self.assertIsInstance(l, types.List[int])
        self.assertFalse(isinstance(l, types.List[float]))

        self.assertIsInstance(ll, types.List)
        self.assertIsInstance(ll, types.List[types.List[int]])
        self._test_type_hierarchy(list, types.List)

        # Test invalid arguments
        with self.assertRaises(TypeError):
            types.List[3]

        with self.assertRaises(TypeError):
            types.List[int, float]

    def test_set_type(self):
        s = {1, 2}
        ls = [{1, 2}, {3, 4}]
        self.assertIsInstance(s, types.Set)
        self.assertIsInstance(s, types.Set[int])
        self.assertFalse(isinstance(s, types.Set[float]))

        self.assertIsInstance(ls, types.List)
        self.assertIsInstance(ls, types.List[types.Set[int]])
        self._test_type_hierarchy(set, types.Set)

        # Test invalid arguments
        with self.assertRaises(TypeError):
            types.Set[3]

        with self.assertRaises(TypeError):
            types.Set[int, float]

    def test_uniform_tuple_type(self):
        t = (1, 2)
        tl = ([1, 2], [3, 4])
        lt = [(1, 2), (3, 4)]
        self.assertIsInstance(t, types.Tuple)
        self.assertIsInstance(t, types.Tuple[int])
        self.assertFalse(isinstance(t, types.Tuple[float]))

        self.assertIsInstance(tl, types.Tuple)
        self.assertIsInstance(tl, types.Tuple[types.List[int]])

        self.assertIsInstance(lt, types.List)
        self.assertIsInstance(lt, types.List[types.Tuple[int]])
        self._test_type_hierarchy(tuple, types.Tuple)

        # Test invalid arguments
        with self.assertRaises(TypeError):
            types.Set[3]

    def test_non_uniform_tuple_type(self):
        t = (1, 2.3, '4', ['a', 'b'])
        self.assertIsInstance(t, types.Tuple)
        self.assertIsInstance(t, types.Tuple[int, float, str, types.List[str]])
        self.assertFalse(isinstance(t, types.Tuple[float, int, str]))
        self.assertFalse(isinstance(t, types.Tuple[float, int, str, int]))

        # Test invalid arguments
        with self.assertRaises(TypeError):
            types.Set[int, 3]

    def test_mapping_type(self):
        d = {'one': 1, 'two': 2}
        dl = {'one': [1, 2], 'two': [3, 4]}
        ld = [{'one': 1, 'two': 2}, {'three': 3}]
        self.assertIsInstance(d, types.Dict)
        self.assertIsInstance(d, types.Dict[str, int])
        self.assertFalse(isinstance(d, types.Dict[int, int]))

        self.assertIsInstance(dl, types.Dict)
        self.assertIsInstance(dl, types.Dict[str, types.List[int]])
        self.assertIsInstance(ld, types.List[types.Dict[str, int]])

        # Test invalid arguments
        with self.assertRaises(TypeError):
            types.Dict[int]

        with self.assertRaises(TypeError):
            types.Dict[int, 3]

    def test_str_type(self):
        s = '123'
        ls = ['1', '23', '456']
        self.assertIsInstance(s, types.Str)
        self.assertIsInstance(s, types.Str[r'\d+'])
        self.assertIsInstance(ls, types.List[types.Str[r'\d+']])
        self.assertFalse(isinstance(s, types.Str[r'a.*']))
        self.assertFalse(isinstance(ls, types.List[types.Str[r'a.*']]))
        self.assertFalse(isinstance('hello, world', types.Str[r'\S+']))

        # Test invalid arguments
        with self.assertRaises(TypeError):
            types.Str[int]

    def test_type_names(self):
        self.assertEqual('List', types.List.__name__)
        self.assertEqual('List[int]', types.List[int].__name__)
        self.assertEqual('Dict[str,List[int]]',
                         types.Dict[str, types.List[int]].__name__)
        self.assertEqual('Tuple[int,Set[float],str]',
                         types.Tuple[int, types.Set[float], str].__name__)
        self.assertEqual("List[Str[r'\d+']]",
                         types.List[types.Str[r'\d+']].__name__)

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
        self.assertIsInstance(l, types.List[C])
        self.assertIsInstance(d, types.Dict[int, C])
        self.assertIsInstance(cd, types.Dict[C, int])
        self.assertIsInstance(t, types.Tuple[int, C, str])
