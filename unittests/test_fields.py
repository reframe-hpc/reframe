# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import datetime
import os
import pytest

import reframe.core.fields as fields
from reframe.core.exceptions import ReframeDeprecationWarning
from reframe.utility import ScopedDict


def test_not_set_attribute():
    class FieldTester:
        var = fields.Field('var')

    c = FieldTester()
    with pytest.raises(AttributeError):
        a = c.var

    with pytest.raises(AttributeError):
        getattr(c, 'var')


def test_copy_on_write_field():
    class FieldTester:
        cow = fields.CopyOnWriteField('cow')

    tester = FieldTester()
    var = [1, [2, 4], 3]

    # Set copy-on-write field
    tester.cow = var

    # Verify that the lists are different
    assert var is not tester.cow

    # Make sure we have a deep copy
    var[1].append(5)
    assert tester.cow == [1, [2, 4], 3]
    assert isinstance(FieldTester.cow, fields.CopyOnWriteField)


def test_constant_field():
    class FieldTester:
        ro = fields.ConstantField('foo')

    tester = FieldTester()
    assert FieldTester.ro == 'foo'
    assert tester.ro == 'foo'
    with pytest.raises(ValueError):
        tester.ro = 'bar'


def test_typed_field():
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
    assert isinstance(FieldTester.field, fields.TypedField)
    assert 3 == tester.field.value
    with pytest.raises(TypeError):
        FieldTester(3)

    tester.field = ClassB()
    assert 10 == tester.field.value
    with pytest.raises(TypeError):
        tester.field = None

    tester.field_any = None
    tester.field_any = 'foo'
    tester.field_any = ClassA(5)
    with pytest.raises(TypeError):
        tester.field_any = 3


def test_timer_field():
    class FieldTester:
        field = fields.TimerField('field')
        field_maybe_none = fields.TimerField(
            'field_maybe_none', type(None))

    tester = FieldTester()
    tester.field = '1d65h22m87s'
    tester.field_maybe_none = None
    assert isinstance(FieldTester.field, fields.TimerField)
    assert (datetime.timedelta(days=1, hours=65,
                               minutes=22, seconds=87) == tester.field)
    tester.field = datetime.timedelta(days=1, hours=65,
                                      minutes=22, seconds=87)
    assert (datetime.timedelta(days=1, hours=65,
                               minutes=22, seconds=87) == tester.field)
    tester.field = ''
    assert (datetime.timedelta(days=0, hours=0,
                               minutes=0, seconds=0) == tester.field)
    with pytest.raises(ValueError):
        tester.field = '1e'

    with pytest.raises(ValueError):
        tester.field = '-10m5s'

    with pytest.raises(ValueError):
        tester.field = '10m-5s'

    with pytest.raises(ValueError):
        tester.field = 'm10s'

    with pytest.raises(ValueError):
        tester.field = '10m10'

    with pytest.raises(ValueError):
        tester.field = '10m10m1s'

    with pytest.raises(ValueError):
        tester.field = '10m5s3m'

    with pytest.raises(ValueError):
        tester.field = '10ms'

    with pytest.raises(ValueError):
        tester.field = '10'


def test_proxy_field():
    class Target:
        def __init__(self):
            self.a = 1
            self.b = 2

    t = Target()

    class Proxy:
        a = fields.ForwardField(t, 'a')
        b = fields.ForwardField(t, 'b')

    proxy = Proxy()
    assert isinstance(Proxy.a, fields.ForwardField)
    assert 1 == proxy.a
    assert 2 == proxy.b

    proxy.a = 3
    proxy.b = 4
    assert 3 == t.a
    assert 4 == t.b


def test_deprecated_field():
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
    with pytest.warns(ReframeDeprecationWarning):
        tester.value = 2

    with pytest.warns(ReframeDeprecationWarning):
        tester.ro = 1

    try:
        tester.wo = 20
    except ReframeDeprecationWarning:
        pytest.fail('deprecation warning not expected here')

    # Test get operation
    try:
        a = tester.ro
    except ReframeDeprecationWarning:
        pytest.fail('deprecation warning not expected here')

    with pytest.warns(ReframeDeprecationWarning):
        a = tester.value

    with pytest.warns(ReframeDeprecationWarning):
        a = tester.wo


def test_absolute_path_field():
    class FieldTester:
        value = fields.AbsolutePathField('value', type(None))

        def __init__(self, value):
            self.value = value

    tester = FieldTester('foo')
    assert os.path.abspath('foo') == tester.value

    # Test set with an absolute path already
    tester.value = os.path.abspath('foo')
    assert os.path.abspath('foo') == tester.value

    # This should not raise
    tester.value = None
    with pytest.raises(TypeError):
        tester.value = 1


def test_scoped_dict_field():
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
    assert isinstance(FieldTester.field, fields.ScopedDictField)
    assert isinstance(tester.field, ScopedDict)
    assert 10 == tester.field['a:k4']

    # Test invalid assignments
    with pytest.raises(TypeError):
        tester.field = {1: "a", 2: "b"}

    with pytest.raises(TypeError):
        tester.field = [('a', 1), ('b', 2)]

    with pytest.raises(TypeError):
        tester.field = {'a': {1: 'k1'}, 'b': {2: 'k2'}}

    # Test assigning a ScopedDict already
    tester.field = ScopedDict({})
