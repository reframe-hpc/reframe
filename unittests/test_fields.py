import unittest
import warnings

from reframe.core.exceptions import FieldError, ReframeDeprecationWarning
from reframe.core.fields import *


class TestFields(unittest.TestCase):
    def test_not_set_attribute(self):
        class FieldTester:
            var = Field('var')

        c = FieldTester()
        self.assertRaises(AttributeError, exec, "a = c.var",
                          globals(), locals())
        self.assertRaises(AttributeError, getattr, c, 'var')

    def test_copy_on_write_field(self):
        class FieldTester:
            cow = CopyOnWriteField('cow')

        tester = FieldTester()
        var = [1, [2, 4], 3]

        # Set copy-on-write field
        tester.cow = var

        # Verify that the lists are different
        self.assertIsNot(var, tester.cow)

        # Make sure we have a deep copy
        var[1].append(5)
        self.assertEqual(tester.cow, [1, [2, 4], 3])

    def test_constant_field(self):
        class FieldTester:
            ro = ConstantField('foo')

        tester = FieldTester()
        self.assertEqual(tester.ro, 'foo')
        self.assertEqual(FieldTester.ro, 'foo')
        self.assertRaises(FieldError, exec, "tester.ro = 'bar'",
                          globals(), locals())

    def test_alphanumeric_field(self):
        class FieldTester:
            field1 = AlphanumericField('field1', allow_none=True)
            field2 = AlphanumericField('field2')

            def __init__(self, value):
                self.field1 = value

        tester1 = FieldTester('foo')
        tester2 = FieldTester('bar')
        self.assertEqual('foo', tester1.field1)
        self.assertEqual('bar', tester2.field1)
        self.assertRaises(FieldError, FieldTester, 12)
        self.assertRaises(FieldError, FieldTester, 'foo bar')

        # Setting field2 must not affect field
        tester1.field2 = 'foobar'
        self.assertEqual('foo', tester1.field1)
        self.assertEqual('foobar', tester1.field2)

        # Setting field1 to None must be fine
        tester1.field1 = None

    def test_typed_field(self):
        class ClassA:
            def __init__(self, val):
                self.value = val

        class ClassB(ClassA):
            def __init__(self):
                super().__init__(10)

        class FieldTester:
            field = TypedField('field', ClassA)
            field_maybe_none = TypedField('field_maybe_none', ClassA,
                                          allow_none=True)

            def __init__(self, value):
                self.field = value

        tester = FieldTester(ClassA(3))
        self.assertEqual(3, tester.field.value)
        self.assertRaises(FieldError, FieldTester, 3)

        tester.field = ClassB()
        self.assertEqual(10, tester.field.value)
        self.assertRaises(FieldError, exec, 'tester.field = None',
                          globals(), locals())
        tester.field_maybe_none = None

    def test_aggregate_typed_field(self):
        class FieldTester:
            simple_int  = AggregateTypeField('simple_int', int)
            int_list    = AggregateTypeField('int_list', (list, int))
            tuple_list  = AggregateTypeField('tuple_list',
                                             (list, (tuple, int)))
            mixed_tuple = AggregateTypeField('mixed_tuple',
                                             (tuple, ((int, float, int),)))
            float_tuple = AggregateTypeField('float_tuple', (tuple, float))
            dict_list   = AggregateTypeField('dict_list',
                                             (list, (dict, (str, int))))
            multilevel_dict = AggregateTypeField(
                'multilevel_dict', (dict, (str, (dict, (str, int))))
            )
            complex_dict = AggregateTypeField(
                'complex_dict',
                (dict, (str, (dict, (str, (list, (
                    tuple, ((str, (callable, None), (callable, None)),))
                )))))
            )

            # Fields allowing None's
            int_list_none = AggregateTypeField('int_list_none',
                                               (list, (int, None)))

            dict_list_none = AggregateTypeField(
                'dict_list_none',
                (list, ((dict, (str, int)), None))
            )
            multilevel_dict_none = AggregateTypeField(
                'multilevel_dict_none',
                (dict, (str, ((dict, (str, (int, None))), None)))
            )
            mixed_tuple_none = AggregateTypeField(
                'mixed_tuple',
                (tuple, ((int, (float, None), (int, None)),))
            )

        int_list = [1, 2, 3]
        int_list_none = [1, None, 3]
        tuple_list = [(1, 2, 3), (4, 5, 6)]
        dict_list = [
            {'a': 1, 'b': 2},
            {'a': 3, 'b': 4}
        ]
        typed_tuple = (1, 2.2, 'foo')
        float_tuple = (2.3, 1.2, 5.6, 9.8)
        mixed_tuple = (1, 2.3, 3)
        multilevel_dict = {
            'foo': {
                'a': 1,
                'b': 2,
            },
            'bar': {
                'c': 3,
                'd': 4,
            }
        }
        complex_dict = {
            '-': {
                'pattern': [
                    ('foo', int, int),
                    ('bar', None, float),
                ],
                'patt': [
                    ('foobar', int, None),
                ]
            }
        }
        dict_list_none = [
            {'a': 1, 'b': 2},
            None
        ]

        # Test valid assignments
        tester = FieldTester()
        tester.simple_int = 1
        tester.int_list = int_list
        tester.int_list_none = int_list_none
        tester.tuple_list = tuple_list
        tester.dict_list = dict_list
        tester.multilevel_dict = multilevel_dict
        tester.float_tuple = float_tuple
        tester.complex_dict = complex_dict
        tester.mixed_tuple = mixed_tuple
        tester.mixed_tuple_none = (1, None, 3)
        tester.mixed_tuple_none = (1, 2.3, None)
        tester.dict_list_none = dict_list_none

        self.assertEqual(1, tester.simple_int)
        self.assertEqual(int_list, tester.int_list)
        self.assertEqual(tuple_list, tester.tuple_list)
        self.assertEqual(dict_list, tester.dict_list)
        self.assertEqual(multilevel_dict, tester.multilevel_dict)
        self.assertEqual(float_tuple, tester.float_tuple)
        self.assertEqual(complex_dict, tester.complex_dict)
        self.assertEqual(dict_list_none, tester.dict_list_none)
        self.assertEqual(int_list_none, tester.int_list_none)

        # Test empty containers
        tester.int_list = []
        tester.tuple_list = []
        tester.dict_list = [{'a': 1, 'b': 2}, {}]
        tester.multilevel_dict = {
            'foo': {},
            'bar': {
                'c': 3,
                'd': 4,
            }
        }

        # Test invalid assignments
        self.assertRaises(FieldError, exec,
                          "tester.int_list = ['a', 'b']",
                          globals(), locals())
        self.assertRaises(FieldError, exec,
                          "tester.int_list = int_list_none",
                          globals(), locals())
        self.assertRaises(FieldError, exec,
                          "tester.int_list = tuple_list",
                          globals(), locals())
        self.assertRaises(FieldError, exec,
                          "tester.dict_list = multilevel_dict",
                          globals(), locals())
        self.assertRaises(FieldError, exec,
                          "tester.dict_list = dict_list_none",
                          globals(), locals())
        self.assertRaises(FieldError, exec,
                          "tester.dict_list = 4",
                          globals(), locals())
        self.assertRaises(FieldError, exec,
                          "tester.float_tuple = mixed_tuple",
                          globals(), locals())
        self.assertRaises(FieldError, exec,
                          "tester.mixed_tuple = float_tuple",
                          globals(), locals())
        self.assertRaises(FieldError, exec,
                          "tester.complex_dict = multilevel_dict",
                          globals(), locals())

    def test_string_field(self):
        class FieldTester:
            field = StringField('field')

            def __init__(self, value):
                self.field = value

        tester = FieldTester('foo')
        self.assertEqual('foo', tester.field)
        self.assertRaises(FieldError, exec, 'tester.field = 13',
                          globals(), locals())

    def test_non_whitespace_field(self):
        class FieldTester:
            field = NonWhitespaceField('field')

        tester = FieldTester()
        tester.field = 'foobar'
        self.assertEqual('foobar', tester.field)
        self.assertRaises(FieldError, exec, 'tester.field = "foo bar"',
                          globals(), locals())

    def test_integer_field(self):
        class FieldTester:
            field  = IntegerField('field')

            def __init__(self, value):
                self.field = value

        tester = FieldTester(5)
        self.assertEqual(5, tester.field)
        self.assertRaises(FieldError, FieldTester, 'foo')
        self.assertRaises(FieldError, exec, "tester.field = 'foo'",
                          globals(), locals())

    def test_boolean_field(self):
        class FieldTester:
            field  = BooleanField('field')

            def __init__(self, value):
                self.field = value

        tester = FieldTester(True)
        self.assertEqual(True, tester.field)
        self.assertRaises(FieldError, FieldTester, 'foo')
        self.assertRaises(FieldError, exec, 'tester.field = 3',
                          globals(), locals())

    def test_typed_list_field(self):
        class FieldTester:
            field  = TypedListField('field', int)

            def __init__(self, value):
                self.field = value

        tester = FieldTester([1, 2, 3])
        self.assertEqual([1, 2, 3], tester.field)
        self.assertRaises(FieldError, FieldTester, [1, 'foo'])
        self.assertRaises(FieldError, exec, 'tester.field = 3',
                          globals(), locals())

    def test_typed_set_field(self):
        class FieldTester:
            field  = TypedSetField('field', int)

            def __init__(self, value):
                self.field = value

        tester = FieldTester({1, 2, 3})
        self.assertEqual({1, 2, 3}, tester.field)
        self.assertRaises(FieldError, FieldTester, {1, 'foo'})
        self.assertRaises(FieldError, exec, 'tester.field = [1, 2]',
                          globals(), locals())

    def test_typed_dict_field(self):
        class FieldTester:
            field  = TypedDictField('field', str, int)

            def __init__(self, value):
                self.field = value

        user_dict = {
            'foo': 1,
            'bar': 2,
            'foobar': 3
        }

        tester = FieldTester(user_dict)
        self.assertEqual(user_dict, tester.field)
        self.assertRaises(FieldError, FieldTester, {1: 'foo'})
        self.assertRaises(FieldError, FieldTester, {'foo': 1.3})
        self.assertRaises(FieldError, exec, 'tester.field = [1, 2]',
                          globals(), locals())

    def test_sanity_field(self):
        warnings.simplefilter('ignore', ReframeDeprecationWarning)

        class FieldTester:
            field = SanityPatternField('field')
            field_maybe_none = SanityPatternField('field_maybe_none',
                                                  allow_none=True)

        sanity = {
            'out': {
                '(?<foo>\S+) (?<bar>\S+)': [
                    ('foo', float, int),
                    ('bar', float, int)
                ],
                '\e': int
            },
            'bar': {
                'foobar': []
            },
        }

        tester = FieldTester()
        tester.field = sanity
        tester.field_maybe_none = None
        self.assertEqual(sanity, tester.field)
        self.assertIn('\e', sanity['out'])
        self.assertRaises(FieldError, exec, "tester.field = 34",
                          globals(), locals())
        self.assertRaises(FieldError, exec,
                          """tester.field = {
                                'out': {
                                        '(?<foo>\S+) (?<bar>\S+)': [
                                                ('foo', float, int),
                                                ('bar', float, int),
                                        ],
                                        '\e': 34
                                },
                          }""", globals(), locals())
        warnings.simplefilter('default', ReframeDeprecationWarning)

    def test_timer_field(self):
        class FieldTester:
            field = TimerField('field')
            field_maybe_none = TimerField('field_maybe_none', allow_none=True)

        tester = FieldTester()
        tester.field = (65, 22, 47)
        tester.field_maybe_none = None

        self.assertEqual((65, 22, 47), tester.field)
        self.assertRaises(FieldError, exec, 'tester.field = (2,)',
                          globals(), locals())
        self.assertRaises(FieldError, exec, 'tester.field = (2, 2)',
                          globals(), locals())
        self.assertRaises(FieldError, exec, 'tester.field = (2, 2, 3.4)',
                          globals(), locals())
        self.assertRaises(FieldError, exec, "tester.field = ('foo', 2, 3)",
                          globals(), locals())
        self.assertRaises(FieldError, exec, 'tester.field = 3',
                          globals(), locals())
        self.assertRaises(FieldError, exec, 'tester.field = (-2, 3, 5)',
                          globals(), locals())
        self.assertRaises(FieldError, exec, 'tester.field = (100, -3, 4)',
                          globals(), locals())
        self.assertRaises(FieldError, exec, 'tester.field = (100, 3, -4)',
                          globals(), locals())
        self.assertRaises(FieldError, exec, 'tester.field = (100, 65, 4)',
                          globals(), locals())
        self.assertRaises(FieldError, exec, 'tester.field = (100, 3, 65)',
                          globals(), locals())

    def test_sandbox(self):
        from reframe.core.environments import Environment
        from reframe.core.systems import System
        from reframe.utility.sandbox import Sandbox

        environ = Environment('myenv')
        system  = System('mysystem')

        sandbox = Sandbox()
        sandbox.environ = environ
        sandbox.system  = system

        self.assertIsNot(system, sandbox.system)
        self.assertIsNot(environ, sandbox.environ)

    def test_proxy_field(self):
        class Target:
            def __init__(self):
                self.a = 1
                self.b = 2

        t = Target()

        class Proxy:
            a = ForwardField(t, 'a')
            b = ForwardField(t, 'b')

        proxy = Proxy()
        self.assertEqual(1, proxy.a)
        self.assertEqual(2, proxy.b)

        proxy.a = 3
        proxy.b = 4
        self.assertEqual(3, t.a)
        self.assertEqual(4, t.b)

    def test_any_field(self):
        class FieldTester:
            value = AnyField('field', [(IntegerField,), (TypedListField, int)],
                             allow_none=True)

            def __init__(self, value):
                self.value = value

        tester = FieldTester(1)
        tester.value = 2
        tester.value = [1, 2]
        tester.value = None
        self.assertRaises(FieldError, exec, 'tester.value = 1.2',
                          globals(), locals())
        self.assertRaises(FieldError, exec, 'tester.value = {1, 2}',
                          globals(), locals())

    def test_settings(self):
        from reframe.settings import settings

        self.assertRaises(AttributeError, exec, "settings.version = '3.0'",
                          globals(), locals())


class TestScopedDict(unittest.TestCase):
    def test_construction(self):
        d = {
            'a': {'k1': 3, 'k2': 4},
            'b': {'k3': 5}
        }
        namespace_dict = ScopedDict()
        namespace_dict = ScopedDict(d)

        # Change local dict and verify that the stored values are not affected
        d['a']['k1'] = 10
        d['b']['k3'] = 10
        self.assertEqual(3, namespace_dict['a:k1'])
        self.assertEqual(5, namespace_dict['b:k3'])
        del d['b']
        self.assertIn('b:k3', namespace_dict)

        self.assertRaises(TypeError, ScopedDict, 1)
        self.assertRaises(TypeError, ScopedDict, {'a': 1, 'b': 2})
        self.assertRaises(TypeError, ScopedDict, [('a', 1), ('b', 2)])
        self.assertRaises(TypeError, ScopedDict, {'a': {1: 'k1'},
                                                  'b': {2: 'k2'}})

    def test_contains(self):
        scoped_dict = ScopedDict({
            'a': {'k1': 1, 'k2': 2},
            'a:b': {'k1': 3, 'k3': 4},
            'a:b:c': {'k2': 5, 'k3': 6},
            '*': {'k1': 7, 'k3': 9, 'k4': 10}
        })

        # Test simple lookup
        self.assertIn('a:k1', scoped_dict)
        self.assertIn('a:k2', scoped_dict)
        self.assertIn('a:k3', scoped_dict)
        self.assertIn('a:k4', scoped_dict)

        self.assertIn('a:b:k1', scoped_dict)
        self.assertIn('a:b:k2', scoped_dict)
        self.assertIn('a:b:k3', scoped_dict)
        self.assertIn('a:b:k4', scoped_dict)

        self.assertIn('a:b:c:k1', scoped_dict)
        self.assertIn('a:b:c:k1', scoped_dict)
        self.assertIn('a:b:c:k1', scoped_dict)
        self.assertIn('a:b:c:k1', scoped_dict)

        # Test global scope
        self.assertIn('k1', scoped_dict)
        self.assertNotIn('k2', scoped_dict)
        self.assertIn('k3', scoped_dict)
        self.assertIn('k4', scoped_dict)

        self.assertIn(':k1', scoped_dict)
        self.assertNotIn(':k2', scoped_dict)
        self.assertIn(':k3', scoped_dict)
        self.assertIn(':k4', scoped_dict)

        self.assertIn('*:k1', scoped_dict)
        self.assertNotIn('*:k2', scoped_dict)
        self.assertIn('*:k3', scoped_dict)
        self.assertIn('*:k4', scoped_dict)

        # Try to get full scopes as keys
        self.assertNotIn('a', scoped_dict)
        self.assertNotIn('a:b', scoped_dict)
        self.assertNotIn('a:b:c', scoped_dict)
        self.assertNotIn('a:b:c:d', scoped_dict)
        self.assertNotIn('*', scoped_dict)
        self.assertNotIn('', scoped_dict)

    def test_iter_keys(self):
        scoped_dict = ScopedDict({
            'a': {'k1': 1, 'k2': 2},
            'a:b': {'k1': 3, 'k3': 4},
            'a:b:c': {'k2': 5, 'k3': 6},
            '*': {'k1': 7, 'k3': 9, 'k4': 10}
        })

        expected_keys = [
            'a:k1', 'a:k2',
            'a:b:k1', 'a:b:k3',
            'a:b:c:k2', 'a:b:c:k3',
            '*:k1', '*:k3', '*:k4'
        ]
        self.assertEqual(sorted(expected_keys),
                         sorted(k for k in scoped_dict.keys()))

    def test_iter_items(self):
        scoped_dict = ScopedDict({
            'a': {'k1': 1, 'k2': 2},
            'a:b': {'k1': 3, 'k3': 4},
            'a:b:c': {'k2': 5, 'k3': 6},
            '*': {'k1': 7, 'k3': 9, 'k4': 10}
        })

        expected_items = [
            ('a:k1', 1), ('a:k2', 2),
            ('a:b:k1', 3), ('a:b:k3', 4),
            ('a:b:c:k2', 5), ('a:b:c:k3', 6),
            ('*:k1', 7), ('*:k3', 9), ('*:k4', 10)
        ]
        self.assertEqual(sorted(expected_items),
                         sorted(item for item in scoped_dict.items()))

    def test_iter_values(self):
        scoped_dict = ScopedDict({
            'a': {'k1': 1, 'k2': 2},
            'a:b': {'k1': 3, 'k3': 4},
            'a:b:c': {'k2': 5, 'k3': 6},
            '*': {'k1': 7, 'k3': 9, 'k4': 10}
        })

        expected_values = [1, 2, 3, 4, 5, 6, 7, 9, 10]
        self.assertEqual(expected_values,
                         sorted(v for v in scoped_dict.values()))

    def test_key_resolution(self):
        scoped_dict = ScopedDict({
            'a': {'k1': 1, 'k2': 2},
            'a:b': {'k1': 3, 'k3': 4},
            'a:b:c': {'k2': 5, 'k3': 6},
            '*': {'k1': 7, 'k3': 9, 'k4': 10}
        })

        self.assertEqual(1, scoped_dict['a:k1'])
        self.assertEqual(2, scoped_dict['a:k2'])
        self.assertEqual(9, scoped_dict['a:k3'])
        self.assertEqual(10, scoped_dict['a:k4'])

        self.assertEqual(3, scoped_dict['a:b:k1'])
        self.assertEqual(2, scoped_dict['a:b:k2'])
        self.assertEqual(4, scoped_dict['a:b:k3'])
        self.assertEqual(10, scoped_dict['a:b:k4'])

        self.assertEqual(3, scoped_dict['a:b:c:k1'])
        self.assertEqual(5, scoped_dict['a:b:c:k2'])
        self.assertEqual(6, scoped_dict['a:b:c:k3'])
        self.assertEqual(10, scoped_dict['a:b:c:k4'])

        # Test global scope
        self.assertEqual(7, scoped_dict['k1'])
        self.assertRaises(
            KeyError, exec, "scoped_dict['k2']", globals(), locals()
        )
        self.assertEqual(9, scoped_dict['k3'])
        self.assertEqual(10, scoped_dict['k4'])

        self.assertEqual(7, scoped_dict[':k1'])
        self.assertRaises(
            KeyError, exec, "scoped_dict[':k2']", globals(), locals()
        )
        self.assertEqual(9, scoped_dict[':k3'])
        self.assertEqual(10, scoped_dict[':k4'])

        self.assertEqual(7, scoped_dict['*:k1'])
        self.assertRaises(
            KeyError, exec, "scoped_dict['*:k2']", globals(), locals()
        )
        self.assertEqual(9, scoped_dict['*:k3'])
        self.assertEqual(10, scoped_dict['*:k4'])

        # Try to fool it, by requesting keys with scope names
        self.assertRaises(
            KeyError, exec, "scoped_dict['a']", globals(), locals()
        )
        self.assertRaises(
            KeyError, exec, "scoped_dict['a:b']", globals(), locals()
        )
        self.assertRaises(
            KeyError, exec, "scoped_dict['a:b:c']", globals(), locals()
        )
        self.assertRaises(
            KeyError, exec, "scoped_dict['a:b:c:d']", globals(), locals()
        )
        self.assertRaises(
            KeyError, exec, "scoped_dict['*']", globals(), locals()
        )
        self.assertRaises(
            KeyError, exec, "scoped_dict['']", globals(), locals()
        )

    def test_setitem(self):
        scoped_dict = ScopedDict({
            'a': {'k1': 1, 'k2': 2},
            'a:b': {'k1': 3, 'k3': 4},
            'a:b:c': {'k2': 5, 'k3': 6},
            '*': {'k1': 7, 'k3': 9, 'k4': 10}
        })

        scoped_dict['a:k2'] = 20
        scoped_dict['c:k2'] = 30
        scoped_dict[':k4'] = 40
        scoped_dict['*:k5'] = 50
        scoped_dict['k6'] = 60
        self.assertEqual(20, scoped_dict['a:k2'])
        self.assertEqual(30, scoped_dict['c:k2'])
        self.assertEqual(40, scoped_dict[':k4'])
        self.assertEqual(50, scoped_dict['k5'])
        self.assertEqual(60, scoped_dict['k6'])

    def test_delitem(self):
        scoped_dict = ScopedDict({
            'a': {'k1': 1, 'k2': 2},
            'a:b': {'k1': 3, 'k3': 4},
            'a:b:c': {'k2': 5, 'k3': 6},
            '*': {'k1': 7, 'k3': 9, 'k4': 10}
        })

        # delete key
        del scoped_dict['a:k1']
        self.assertEqual(7, scoped_dict['a:k1'])

        # delete a whole scope
        del scoped_dict['*']
        self.assertRaises(
            KeyError, exec, "scoped_dict[':k4']", globals(), locals()
        )
        self.assertRaises(
            KeyError, exec, "scoped_dict['a:k3']", globals(), locals()
        )

        # try to delete a non-existent key
        self.assertRaises(
            KeyError, exec, "del scoped_dict['a:k4']", globals(), locals()
        )

    def test_update(self):
        scoped_dict = ScopedDict({
            'a': {'k1': 1, 'k2': 2},
            'a:b': {'k1': 3, 'k3': 4},
            'a:b:c': {'k2': 5, 'k3': 6},
            '*': {'k1': 7, 'k3': 9, 'k4': 10}
        })

        scoped_dict_alt = ScopedDict({'a': {'k1': 3, 'k2': 5}})
        scoped_dict_alt.update({
            'a': {'k1': 1, 'k2': 2},
            'a:b': {'k1': 3, 'k3': 4},
            'a:b:c': {'k2': 5, 'k3': 6},
            '*': {'k1': 7, 'k3': 9, 'k4': 10}
        })
        self.assertEqual(scoped_dict, scoped_dict_alt)

    def test_scoped_dict_field(self):
        class FieldTester:
            field = ScopedDictField('field', int)
            field_maybe_none = ScopedDictField('field_maybe_none',
                                               int, allow_none=True)

        tester = FieldTester()

        # Test valid assignments
        tester.field = {
            'a': {'k1': 1, 'k2': 2},
            'a:b': {'k1': 3, 'k3': 4},
            'a:b:c': {'k2': 5, 'k3': 6},
            '*': {'k1': 7, 'k3': 9, 'k4': 10}
        }
        tester.field_maybe_none = None

        # Check that we have indeed a ScopedDict here
        self.assertTrue(isinstance(tester.field, ScopedDict))
        self.assertEqual(10, tester.field['a:k4'])

        # Test invalid assignments
        self.assertRaises(FieldError, exec,
                          'tester.field = {1: "a", 2: "b" }',
                          globals(), locals())
        self.assertRaises(FieldError, exec,
                          "tester.field = [('a', 1), ('b', 2)]",
                          globals(), locals())
        self.assertRaises(FieldError, exec,
                          """tester.field = {'a': {1: 'k1'},
                                             'b': {2: 'k2'}}""",
                          globals(), locals())

        # Test assigning a ScopedDict already
        tester.field = ScopedDict({})


class TestDeprecationOfFields(unittest.TestCase):
    def test_sanityfield_set_deprecation(self):
        class FieldTester:
            sanity = SanityPatternField('sanity')

        fixture = FieldTester()
        with self.assertWarns(ReframeDeprecationWarning):
            fixture.sanity = {}
