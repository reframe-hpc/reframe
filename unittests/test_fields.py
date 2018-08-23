import os
import unittest

import reframe.core.fields as fields
from reframe.utility import ScopedDict


class TestFields(unittest.TestCase):
    def test_not_set_attribute(self):
        class FieldTester:
            var = fields.Field('var')

        c = FieldTester()
        self.assertRaises(AttributeError, exec, "a = c.var",
                          globals(), locals())
        self.assertRaises(AttributeError, getattr, c, 'var')

    def test_copy_on_write_field(self):
        class FieldTester:
            cow = fields.CopyOnWriteField('cow')

        tester = FieldTester()
        var = [1, [2, 4], 3]

        # Set copy-on-write field
        tester.cow = var

        # Verify that the lists are different
        self.assertIsNot(var, tester.cow)

        # Make sure we have a deep copy
        var[1].append(5)
        self.assertEqual(tester.cow, [1, [2, 4], 3])
        self.assertIsInstance(FieldTester.cow, fields.CopyOnWriteField)

    def test_constant_field(self):
        class FieldTester:
            ro = fields.ConstantField('foo')

        tester = FieldTester()
        self.assertEqual(FieldTester.ro, 'foo')
        self.assertEqual(tester.ro, 'foo')
        self.assertRaises(ValueError, exec, "tester.ro = 'bar'",
                          globals(), locals())

    def test_typed_field(self):
        class ClassA:
            def __init__(self, val):
                self.value = val

        class ClassB(ClassA):
            def __init__(self):
                super().__init__(10)

        class FieldTester:
            field = fields.TypedField('field', ClassA)
            field_maybe_none = fields.TypedField('field_maybe_none', ClassA,
                                                 allow_none=True)

            def __init__(self, value):
                self.field = value

        tester = FieldTester(ClassA(3))
        self.assertIsInstance(FieldTester.field, fields.TypedField)
        self.assertEqual(3, tester.field.value)
        self.assertRaises(TypeError, FieldTester, 3)

        tester.field = ClassB()
        self.assertEqual(10, tester.field.value)
        self.assertRaises(TypeError, exec, 'tester.field = None',
                          globals(), locals())
        tester.field_maybe_none = None

    def test_aggregate_typed_field(self):
        class FieldTester:
            simple_int  = fields.AggregateTypeField('simple_int', int)
            int_list    = fields.AggregateTypeField('int_list', (list, int))
            tuple_list  = fields.AggregateTypeField('tuple_list',
                                                    (list, (tuple, int)))
            mixed_tuple = fields.AggregateTypeField(
                'mixed_tuple', (tuple, ((int, float, int),)))
            float_tuple = fields.AggregateTypeField('float_tuple',
                                                    (tuple, float))
            dict_list   = fields.AggregateTypeField('dict_list',
                                                    (list, (dict, (str, int))))
            multilevel_dict = fields.AggregateTypeField(
                'multilevel_dict', (dict, (str, (dict, (str, int))))
            )
            complex_dict = fields.AggregateTypeField(
                'complex_dict',
                (dict, (str, (dict, (str, (list, (
                    tuple, ((str, (callable, None), (callable, None)),))
                )))))
            )

            # Fields allowing None's
            int_list_none = fields.AggregateTypeField('int_list_none',
                                                      (list, (int, None)))

            dict_list_none = fields.AggregateTypeField(
                'dict_list_none',
                (list, ((dict, (str, int)), None))
            )
            multilevel_dict_none = fields.AggregateTypeField(
                'multilevel_dict_none',
                (dict, (str, ((dict, (str, (int, None))), None)))
            )
            mixed_tuple_none = fields.AggregateTypeField(
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

        self.assertIsInstance(FieldTester.simple_int,
                              fields.AggregateTypeField)
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
        self.assertRaises(TypeError, exec,
                          "tester.int_list = ['a', 'b']",
                          globals(), locals())
        self.assertRaises(TypeError, exec,
                          "tester.int_list = int_list_none",
                          globals(), locals())
        self.assertRaises(TypeError, exec,
                          "tester.int_list = tuple_list",
                          globals(), locals())
        self.assertRaises(TypeError, exec,
                          "tester.dict_list = multilevel_dict",
                          globals(), locals())
        self.assertRaises(TypeError, exec,
                          "tester.dict_list = dict_list_none",
                          globals(), locals())
        self.assertRaises(TypeError, exec,
                          "tester.dict_list = 4",
                          globals(), locals())
        self.assertRaises(TypeError, exec,
                          "tester.float_tuple = mixed_tuple",
                          globals(), locals())
        self.assertRaises(TypeError, exec,
                          "tester.mixed_tuple = float_tuple",
                          globals(), locals())
        self.assertRaises(TypeError, exec,
                          "tester.complex_dict = multilevel_dict",
                          globals(), locals())

    def test_string_field(self):
        class FieldTester:
            field = fields.StringField('field')

            def __init__(self, value):
                self.field = value

        tester = FieldTester('foo')
        self.assertIsInstance(FieldTester.field, fields.StringField)
        self.assertEqual('foo', tester.field)
        self.assertRaises(TypeError, exec, 'tester.field = 13',
                          globals(), locals())

    def test_string_pattern_field(self):
        class FieldTester:
            field = fields.StringPatternField('field', '\S+')

            def __init__(self, value):
                self.field = value

        tester = FieldTester('foo123')
        self.assertIsInstance(FieldTester.field, fields.StringPatternField)
        self.assertEqual('foo123', tester.field)
        self.assertRaises(TypeError, exec, 'tester.field = 13',
                          globals(), locals())
        self.assertRaises(ValueError, exec, 'tester.field = "foo 123"',
                          globals(), locals())

    def test_integer_field(self):
        class FieldTester:
            field  = fields.IntegerField('field')

            def __init__(self, value):
                self.field = value

        tester = FieldTester(5)
        self.assertIsInstance(FieldTester.field, fields.IntegerField)
        self.assertEqual(5, tester.field)
        self.assertRaises(TypeError, FieldTester, 'foo')
        self.assertRaises(TypeError, exec, "tester.field = 'foo'",
                          globals(), locals())

    def test_boolean_field(self):
        class FieldTester:
            field  = fields.BooleanField('field')

            def __init__(self, value):
                self.field = value

        tester = FieldTester(True)
        self.assertIsInstance(FieldTester.field, fields.BooleanField)
        self.assertEqual(True, tester.field)
        self.assertRaises(TypeError, FieldTester, 'foo')
        self.assertRaises(TypeError, exec, 'tester.field = 3',
                          globals(), locals())

    def test_typed_list_field(self):
        class FieldTester:
            field  = fields.TypedListField('field', str)

            def __init__(self, value):
                self.field = value

        tester = FieldTester(['a', 'b', 'c'])
        self.assertEqual(['a', 'b', 'c'], tester.field)
        self.assertRaises(TypeError, FieldTester, [1, 'foo'])
        with self.assertRaises(TypeError):
            tester.field = 'foo'

    def test_typed_sequence_field(self):
        class FieldTester:
            field  = fields.TypedSequenceField('field', str)

            def __init__(self, value):
                self.field = value

        tester = FieldTester(['a', 'b', 'c'])
        self.assertEqual(['a', 'b', 'c'], tester.field)

        # strings are valid sequences
        tester.field = 'foo'
        self.assertEqual('foo', tester.field)
        self.assertRaises(TypeError, FieldTester, [1, 'foo'])

    def test_typed_set_field(self):
        class FieldTester:
            field  = fields.TypedSetField('field', int)

            def __init__(self, value):
                self.field = value

        tester = FieldTester({1, 2, 3})
        self.assertIsInstance(FieldTester.field, fields.TypedSetField)
        self.assertEqual({1, 2, 3}, tester.field)
        self.assertRaises(TypeError, FieldTester, {1, 'foo'})
        self.assertRaises(TypeError, exec, 'tester.field = [1, 2]',
                          globals(), locals())

    def test_typed_dict_field(self):
        class FieldTester:
            field  = fields.TypedDictField('field', str, int)

            def __init__(self, value):
                self.field = value

        user_dict = {
            'foo': 1,
            'bar': 2,
            'foobar': 3
        }

        tester = FieldTester(user_dict)
        self.assertIsInstance(FieldTester.field, fields.TypedDictField)
        self.assertEqual(user_dict, tester.field)
        self.assertRaises(TypeError, FieldTester, {1: 'foo'})
        self.assertRaises(TypeError, FieldTester, {'foo': 1.3})
        self.assertRaises(TypeError, exec, 'tester.field = [1, 2]',
                          globals(), locals())

    def test_timer_field(self):
        class FieldTester:
            field = fields.TimerField('field')
            field_maybe_none = fields.TimerField('field_maybe_none',
                                                 allow_none=True)

        tester = FieldTester()
        tester.field = (65, 22, 47)
        tester.field_maybe_none = None

        self.assertIsInstance(FieldTester.field, fields.TimerField)
        self.assertEqual((65, 22, 47), tester.field)
        self.assertRaises(TypeError, exec, 'tester.field = (2,)',
                          globals(), locals())
        self.assertRaises(TypeError, exec, 'tester.field = (2, 2)',
                          globals(), locals())
        self.assertRaises(TypeError, exec, 'tester.field = (2, 2, 3.4)',
                          globals(), locals())
        self.assertRaises(TypeError, exec, "tester.field = ('foo', 2, 3)",
                          globals(), locals())
        self.assertRaises(TypeError, exec, 'tester.field = 3',
                          globals(), locals())
        self.assertRaises(ValueError, exec, 'tester.field = (-2, 3, 5)',
                          globals(), locals())
        self.assertRaises(ValueError, exec, 'tester.field = (100, -3, 4)',
                          globals(), locals())
        self.assertRaises(ValueError, exec, 'tester.field = (100, 3, -4)',
                          globals(), locals())
        self.assertRaises(ValueError, exec, 'tester.field = (100, 65, 4)',
                          globals(), locals())
        self.assertRaises(ValueError, exec, 'tester.field = (100, 3, 65)',
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
            a = fields.ForwardField(t, 'a')
            b = fields.ForwardField(t, 'b')

        proxy = Proxy()
        self.assertIsInstance(Proxy.a, fields.ForwardField)
        self.assertEqual(1, proxy.a)
        self.assertEqual(2, proxy.b)

        proxy.a = 3
        proxy.b = 4
        self.assertEqual(3, t.a)
        self.assertEqual(4, t.b)

    def test_any_field(self):
        class FieldTester:
            value = fields.AnyField('field',
                                    [(fields.IntegerField,),
                                     (fields.TypedListField, int)],
                                    allow_none=True)

            def __init__(self, value):
                self.value = value

        tester = FieldTester(1)
        tester.value = 2
        tester.value = [1, 2]
        tester.value = None
        self.assertIsInstance(FieldTester.value, fields.AnyField)
        self.assertRaises(TypeError, exec, 'tester.value = 1.2',
                          globals(), locals())
        self.assertRaises(TypeError, exec, 'tester.value = {1, 2}',
                          globals(), locals())

    def test_deprecated_field(self):
        from reframe.core.exceptions import ReframeDeprecationWarning

        class FieldTester:
            value = fields.DeprecatedField(fields.IntegerField('value'),
                                           'value field is deprecated')
            _value = fields.IntegerField('value')
            ro = fields.DeprecatedField(fields.IntegerField('ro'),
                                        'value field is deprecated',
                                        fields.DeprecatedField.OP_SET)
            _ro = fields.IntegerField('ro')
            wo = fields.DeprecatedField(fields.IntegerField('wo'),
                                        'value field is deprecated',
                                        fields.DeprecatedField.OP_GET)

            def __init__(self):
                self._value = 1
                self._ro = 2
                self.wo = 3

        tester = FieldTester()

        # Test set operation
        with self.assertWarns(ReframeDeprecationWarning):
            tester.value = 2

        with self.assertWarns(ReframeDeprecationWarning):
            tester.ro = 1

        try:
            tester.wo = 20
        except ReframeDeprecationWarning:
            self.fail('deprecation warning not expected here')

        # Test get operation
        try:
            a = tester.ro
        except ReframeDeprecationWarning:
            self.fail('deprecation warning not expected here')

        with self.assertWarns(ReframeDeprecationWarning):
            a = tester.value

        with self.assertWarns(ReframeDeprecationWarning):
            a = tester.wo

    def test_absolute_path_field(self):
        class FieldTester:
            value = fields.AbsolutePathField('value', allow_none=True)

            def __init__(self, value):
                self.value = value

        tester = FieldTester('foo')
        self.assertEquals(os.path.abspath('foo'), tester.value)

        # Test set with an absolute path already
        tester.value = os.path.abspath('foo')
        self.assertEquals(os.path.abspath('foo'), tester.value)

        # This should not raise
        tester.value = None
        self.assertRaises(TypeError, exec, 'tester.value = 1',
                          globals(), locals())

    def test_scoped_dict_field(self):
        class FieldTester:
            field = fields.ScopedDictField('field', int)
            field_maybe_none = fields.ScopedDictField('field_maybe_none',
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
        self.assertIsInstance(FieldTester.field, fields.ScopedDictField)
        self.assertIsInstance(tester.field, ScopedDict)
        self.assertEqual(10, tester.field['a:k4'])

        # Test invalid assignments
        self.assertRaises(TypeError, exec,
                          'tester.field = {1: "a", 2: "b" }',
                          globals(), locals())
        self.assertRaises(TypeError, exec,
                          "tester.field = [('a', 1), ('b', 2)]",
                          globals(), locals())
        self.assertRaises(TypeError, exec,
                          """tester.field = {'a': {1: 'k1'},
                                             'b': {2: 'k2'}}""",
                          globals(), locals())

        # Test assigning a ScopedDict already
        tester.field = ScopedDict({})
