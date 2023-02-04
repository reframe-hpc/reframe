# Copyright 2016-2023 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import datetime
import pytest
import semver
import warnings

import reframe
import reframe.core.fields as fields
from reframe.core.warnings import ReframeDeprecationWarning
from reframe.utility import ScopedDict


def test_not_set_attribute():
    class FieldTester:
        var = fields.Field()

    c = FieldTester()
    with pytest.raises(AttributeError):
        c.var

    with pytest.raises(AttributeError):
        getattr(c, 'var')


def test_alt_attr_name():
    class FieldTester:
        var = fields.Field(attr_name='foo')

    c = FieldTester()
    c.var = 5

    assert c.var == 5
    assert c.foo == 5

    c.foo = 6
    assert c.var == 6
    assert c.foo == 6

    assert 'var' not in c.__dict__
    assert 'foo' in c.__dict__


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
        field = fields.TypedField(ClassA)
        field_any = fields.TypedField(ClassA, str, type(None))

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


def test_typed_field_convertible():
    class FieldTester:
        fieldA = fields.TypedField(int, str)
        fieldB = fields.TypedField(str, int)
        fieldC = fields.TypedField(int)

    tester = FieldTester()
    tester.fieldA = fields.make_convertible('10')
    tester.fieldB = fields.make_convertible('10')
    assert tester.fieldA == 10
    assert tester.fieldB == '10'

    with pytest.raises(TypeError):
        tester.fieldC = fields.make_convertible(None)


def test_timer_field():
    class FieldTester:
        field = fields.TimerField()
        field_maybe_none = fields.TimerField(type(None))

    tester = FieldTester()
    tester.field = '1d65h22m87s'
    tester.field_maybe_none = None
    assert isinstance(FieldTester.field, fields.TimerField)
    secs = datetime.timedelta(days=1, hours=65,
                              minutes=22, seconds=87).total_seconds()
    assert tester.field == secs
    tester.field = secs
    assert tester.field == secs
    tester.field = ''
    assert tester.field == 0
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

    with pytest.raises(ValueError):
        tester.field = -10


def test_deprecated_field():
    class FieldTester:
        value = fields.DeprecatedField(fields.TypedField(int),
                                       'value field is deprecated')
        _value = fields.TypedField(int)
        ro = fields.DeprecatedField(fields.TypedField(int),
                                    'value field is deprecated',
                                    fields.DeprecatedField.OP_SET)
        _ro = fields.TypedField(int)
        wo = fields.DeprecatedField(fields.TypedField(int),
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

    with warnings.catch_warnings(record=True) as w:
        tester.wo = 20
        assert len(w) == 0

    # Test get operation
    with warnings.catch_warnings(record=True) as w:
        tester.ro
        assert len(w) == 0

    with pytest.warns(ReframeDeprecationWarning):
        tester.value

    with pytest.warns(ReframeDeprecationWarning):
        tester.wo


def test_deprecated_field_future_version():
    next_version = semver.VersionInfo.parse(reframe.VERSION).bump_minor()

    class FieldTester:
        x = fields.DeprecatedField(fields.TypedField(int), 'deprecated',
                                   from_version=str(next_version))

    # This should not raise a warning
    tester = FieldTester()
    with warnings.catch_warnings(record=True) as w:
        tester.x  = 1
        assert len(w) == 0


def test_scoped_dict_field():
    class FieldTester:
        field = fields.ScopedDictField(int)
        field_maybe_none = fields.ScopedDictField(
            int, type(None))

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
