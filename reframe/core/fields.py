#
# Useful descriptors for advanced operations on fields
#

import copy
import os

import reframe.utility.typecheck as types
from reframe.core.exceptions import user_deprecation_warning
from reframe.utility import ScopedDict


class Field:
    """Base class for attribute validators."""

    def __init__(self, fieldname):
        self._name = fieldname

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
        obj.__dict__[self._name] = value


class ForwardField:
    """Simple field that forwards set/get to a target object."""

    def __init__(self, obj, attr):
        self._target = obj
        self._attr = attr

    def __get__(self, obj, objtype):
        if obj is None:
            return self

        return getattr(self._target, self._attr)

    def __set__(self, obj, value):
        self._target.__dict__[self._attr] = value


class TypedField(Field):
    """Stores a field of predefined type"""

    def __init__(self, fieldname, main_type, *other_types):
        super().__init__(fieldname)
        self._types = (main_type,) + other_types
        if not all(isinstance(t, type) for t in self._types):
            raise TypeError('{0} is not a sequence of types'.
                            format(self._types))

    def _check_type(self, value):
        if not any(isinstance(value, t) for t in self._types):
            typedescr = '|'.join(t.__name__ for t in self._types)
            raise TypeError(
                "failed to set field '%s': '%s' is not of type '%s'" %
                (self._name, value, typedescr))

    def __set__(self, obj, value):
        self._check_type(value)
        super().__set__(obj, value)


class CopyOnWriteField(Field):
    """Holds a copy of the variable that is assigned to it the first time"""

    def __set__(self, obj, value):
        super().__set__(obj, copy.deepcopy(value))


class ConstantField(Field):
    """Holds a constant.

    Attempt to set it will raise an exception. This field may be accessed also
    from the class and will return the same constant value.

    :arg value: the value of this field.

    """

    def __init__(self, value):
        super().__init__('__readonly')
        self._value = value

    def __get__(self, obj, objtype):
        return self._value

    def __set__(self, obj, value):
        raise ValueError('attempt to set a read-only variable')


class TimerField(TypedField):
    """Stores a timer in the form of a tuple '(hh, mm, ss)'"""

    def __init__(self, fieldname, *other_types):
        super().__init__(fieldname, types.Tuple[int, int, int], *other_types)

    def __set__(self, obj, value):
        self._check_type(value)
        if value is not None:
            # Check also the values for minutes and seconds
            h, m, s = value
            if h < 0 or m < 0 or s < 0:
                raise ValueError('timer field must have '
                                 'non-negative values')

            if m > 59 or s > 59:
                raise ValueError('minutes and seconds in a timer '
                                 'field must not exceed 59')

        # Call Field's __set__() method, type checking is already performed
        Field.__set__(self, obj, value)


class AbsolutePathField(TypedField):
    """A string field that stores an absolute path.

    Any string assigned to such a field, will be converted to an absolute path.
    """

    def __init__(self, fieldname, *other_types):
        super().__init__(fieldname, str, *other_types)

    def __set__(self, obj, value):
        self._check_type(value)
        if value is not None:
            value = os.path.abspath(value)

        # Call Field's __set__() method, type checking is already performed
        Field.__set__(self, obj, value)


class ScopedDictField(TypedField):
    """Stores a ScopedDict with a specific type.

    It also handles implicit conversions from ordinary dicts."""

    def __init__(self, fieldname, valuetype, *other_types):
        super().__init__(fieldname,
                         types.Dict[str, types.Dict[str, valuetype]],
                         ScopedDict, *other_types)

    def __set__(self, obj, value):
        self._check_type(value)
        if not isinstance(value, ScopedDict):
            value = ScopedDict(value) if value is not None else value

        Field.__set__(self, obj, value)


class DeprecatedField(Field):
    """Field wrapper for deprecating fields."""

    OP_SET = 1
    OP_GET = 2
    OP_ALL = OP_SET | OP_GET

    def __init__(self, target_field, message, op=OP_ALL):
        self._target_field = target_field
        self._message = message
        self._op = op

    def __set__(self, obj, value):
        if self._op & DeprecatedField.OP_SET:
            user_deprecation_warning(self._message)

        self._target_field.__set__(obj, value)

    def __get__(self, obj, objtype):
        if self._op & DeprecatedField.OP_GET:
            user_deprecation_warning(self._message)

        return self._target_field.__get__(obj, objtype)
