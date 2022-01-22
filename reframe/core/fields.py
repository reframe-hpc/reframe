# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Useful descriptors for advanced operations on fields
#

import datetime
import re

import reframe.utility.typecheck as types
from reframe.core.warnings import user_deprecation_warning
from reframe.utility import ScopedDict


class _Convertible:
    '''Wrapper for values that allowed to be converted implicitly'''

    __slots__ = ('value')

    def __init__(self, value):
        self.value = value


def make_convertible(value):
    return _Convertible(value)


def remove_convertible(value):
    if isinstance(value, _Convertible):
        return value.value
    else:
        return value


class Field:
    '''Base class for attribute validators.'''

    def __init__(self, attr_name=None):
        if attr_name is not None:
            self._name = attr_name

    def __set_name__(self, owner, name):
        if not hasattr(self, '_name'):
            self._name = name

    def __get__(self, obj, objtype):
        if obj is None:
            return self

        try:
            return obj.__dict__[self._name]
        except KeyError:
            # We raise an AttributeError to emulate the standard attribute
            # access.
            raise AttributeError("%s object has no attribute '%s'" %
                                 (objtype.__name__, self._name)) from None

    def __set__(self, obj, value):
        obj.__dict__[self._name] = remove_convertible(value)


class TypedField(Field):
    '''Stores a field of predefined type'''

    def __init__(self, main_type, *other_types, attr_name=None):
        super().__init__(attr_name)
        self._types = (main_type,) + other_types
        if not all(isinstance(t, type) for t in self._types):
            raise TypeError('{0} is not a sequence of types'.
                            format(self._types))

    @property
    def valid_types(self):
        return self._types

    def _check_type(self, value):
        if not any(isinstance(value, t) for t in self._types):
            typedescr = '|'.join(t.__name__ for t in self._types)
            raise TypeError(
                "failed to set field '%s': '%s' is not of type '%s'" %
                (self._name, value, typedescr))

    def __set__(self, obj, value):
        try:
            self._check_type(value)
        except TypeError:
            raw_value = remove_convertible(value)
            if raw_value is value:
                # value was not convertible; reraise
                raise

            # Try to convert value to any of the supported types
            value = raw_value
            for t in self._types:
                try:
                    value = t(value)
                except TypeError:
                    continue
                else:
                    super().__set__(obj, value)
                    return

            # Conversion failed
            typenames = [t.__name__ for t in self._types]
            raise TypeError(
                f'failed to set field {self._name!r}: '
                f'could not convert to any of the supported types: '
                f'{typenames}'
            )
        else:
            super().__set__(obj, value)


class ConstantField(Field):
    '''Holds a constant.

    Attempt to set it will raise an exception. This field may be accessed also
    from the class and will return the same constant value.

    :arg value: the value of this field.

    '''

    def __set_name__(self, owner, name):
        pass

    def __init__(self, value):
        self._value = value

    def __get__(self, obj, objtype):
        return self._value

    def __set__(self, obj, value):
        raise ValueError('attempt to set a read-only variable')


class TimerField(TypedField):
    '''Stores a timer in the form of a :class:`datetime.timedelta` object'''

    def __init__(self, *other_types, attr_name=None):
        super().__init__(str, int, float, *other_types, attr_name=attr_name)

    def __set__(self, obj, value):
        value = remove_convertible(value)
        self._check_type(value)
        if isinstance(value, str):
            time_match = re.match(r'^((?P<days>\d+)d)?'
                                  r'((?P<hours>\d+)h)?'
                                  r'((?P<minutes>\d+)m)?'
                                  r'((?P<seconds>\d+)s)?$',
                                  value)
            if not time_match:
                raise ValueError('invalid format for timer field')

            value = datetime.timedelta(
                **{k: int(v) for k, v in time_match.groupdict().items() if v}
            ).total_seconds()
        elif isinstance(value, float) or isinstance(value, int):
            if value < 0:
                raise ValueError('timer field value cannot be negative')

        # Call Field's __set__() method, type checking is already performed
        Field.__set__(self, obj, value)


class ScopedDictField(TypedField):
    '''Stores a ScopedDict with a specific type.

    It also handles implicit conversions from ordinary dicts.'''

    def __init__(self, valuetype, *other_types, attr_name=None):
        super().__init__(types.Dict[str, types.Dict[str, valuetype]],
                         ScopedDict, *other_types, attr_name=attr_name)

    def __set__(self, obj, value):
        value = remove_convertible(value)
        self._check_type(value)
        if not isinstance(value, ScopedDict):
            value = ScopedDict(value) if value is not None else value

        Field.__set__(self, obj, value)


class DeprecatedField(Field):
    '''Field wrapper for deprecating fields.'''

    OP_GET = 1
    OP_SET = 2
    OP_ALL = OP_SET | OP_GET

    @property
    def message(self):
        return self._message

    @property
    def op(self):
        return self._op

    @property
    def from_version(self):
        return self._from_version

    def __set_name__(self, owner, name):
        self._target_field.__set_name__(owner, name)

    def __init__(self, target_field, message, op=OP_ALL, from_version='0.0.0'):
        self._target_field = target_field
        self._message = message
        self._op = op
        self._from_version = from_version

    def __set__(self, obj, value):
        if self._op & DeprecatedField.OP_SET:
            user_deprecation_warning(self._message, self._from_version)

        self._target_field.__set__(obj, value)

    def __get__(self, obj, objtype):
        if self._op & DeprecatedField.OP_GET:
            user_deprecation_warning(self._message, self._from_version)

        return self._target_field.__get__(obj, objtype)
