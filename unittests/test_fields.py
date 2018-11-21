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
            field_any = fields.TypedField('field_any', ClassA, str, type(None))

            def __init__(self, value):
                self.field = value

        tester = FieldTester(ClassA(3))
        self.assertIsInstance(FieldTester.field, fields.TypedField)
        self.assertEqual(3, tester.field.value)
        self.assertRaises(TypeError, FieldTester, 3)

        tester.field = ClassB()
        self.assertEqual(10, tester.field.value)
        with self.assertRaises(TypeError):
            tester.field = None

        tester.field_any = None
        tester.field_any = 'foo'
        tester.field_any = ClassA(5)
        with self.assertRaises(TypeError):
            tester.field_any = 3

    def test_timer_field(self):
        class FieldTester:
            field = fields.TimerField('field')
            field_maybe_none = fields.TimerField(
                'field_maybe_none', type(None))

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

    def test_deprecated_field(self):
        from reframe.core.exceptions import ReframeDeprecationWarning

        class FieldTester:
            value = fields.DeprecatedField(fields.TypedField('value', int),
                                           'value field is deprecated')
            _value = fields.TypedField('value', int)
            ro = fields.DeprecatedField(fields.TypedField('ro', int),
                                        'value field is deprecated',
                                        fields.DeprecatedField.OP_SET)
            _ro = fields.TypedField('ro', int)
            wo = fields.DeprecatedField(fields.TypedField('wo', int),
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
            value = fields.AbsolutePathField('value', type(None))

            def __init__(self, value):
                self.value = value

        tester = FieldTester('foo')
        self.assertEqual(os.path.abspath('foo'), tester.value)

        # Test set with an absolute path already
        tester.value = os.path.abspath('foo')
        self.assertEqual(os.path.abspath('foo'), tester.value)

        # This should not raise
        tester.value = None
        with self.assertRaises(TypeError):
            tester.value = 1

    def test_scoped_dict_field(self):
        class FieldTester:
            field = fields.ScopedDictField('field', int)
            field_maybe_none = fields.ScopedDictField(
                'field_maybe_none', int, type(None))

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
