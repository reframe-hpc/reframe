#
# Useful descriptors for advanced operations on fields
#

import collections.abc
import copy
import numbers
import re

import reframe.core.debug as debug

from collections import UserDict
from reframe.core.exceptions import FieldError, user_deprecation_warning


class Field:
    """Base class for fields"""

    def __init__(self, fieldname):
        self._name = fieldname

    def __repr__(self):
        return debug.repr(self)

    def __get__(self, obj, objtype):
        return obj.__dict__[self._name]

    def __set__(self, obj, value):
        obj.__dict__[self._name] = value


class ForwardField:
    """Simple field that forwards set/get to a target object."""

    def __init__(self, obj, attr):
        self._target = obj
        self._attr = attr

    def __get__(self, obj, objtype):
        return self._target.__dict__[self._attr]

    def __set__(self, obj, value):
        self._target.__dict__[self._attr] = value


class AnyField(Field):
    """Store a value that may be validated by distinct fields."""

    def __init__(self, fieldname, fields, allow_none=False):
        super().__init__(fieldname)
        self._fields = [fieldtype(self._name, *args, allow_none=allow_none)
                        for fieldtype, *args in fields]

    def __set__(self, obj, value):
        # Try the field descriptors in turn until one validates the input
        for f in self._fields:
            try:
                f.__set__(obj, value)
                break
            except FieldError:
                pass
        else:
            # All field set attemps have failed
            # FIXME: This is not a very informative message.
            raise FieldError('attempt to set a field of different type.')


class TypedField(Field):
    """Stores a field of predefined type"""

    def __init__(self, fieldname, fieldtype, allow_none=False):
        super().__init__(fieldname)
        self._fieldtype = fieldtype
        self._allow_none = allow_none

    def __set__(self, obj, value):
        if ((value is not None or not self._allow_none) and
            not isinstance(value, self._fieldtype)):
            raise FieldError('attempt to set a field of different type. '
                             'Required: %s, got: %s' %
                             (self._fieldtype, type(value).__name__))

        super().__set__(obj, value)


class AggregateTypeField(Field):
    """Store a typed container with elements of certain type(s)

    The typespec has the following syntax:
    typespec := type |
                (typespec, None) |
                (aggr_type, typespec) |
                (seq_type, ((typespec, typespec, ...),)) |
                (map_type, (typespec, typespec))
    aggr_type := list | tuple | set | frozenset | dict
    seq_type  := list | tuple
    map_type  := dict
    type      := <any-python-type> | callable"""

    def __init__(self, fieldname, typespec, allow_none=False):
        super().__init__(fieldname)
        self._typespec = typespec
        self._allow_none = allow_none

    def __set__(self, obj, value):
        if not self._check_type_ext(value):
            raise FieldError('attempt to set an aggregate field '
                             'of different type. Required typespec: %s' %
                             str(self._typespec))

        super().__set__(obj, value)

    def _check_type_ext(self, value):
        # Checks also value against None if that's allowed
        if value is None and self._allow_none:
            return True

        return self._check_type(value, self._typespec)

    def _extract_typeinfo(self, typespec):
        """Check if a typespec is of the form (type, None)

        If yes, returns (type, True) else (typespec, False)
        """
        if not isinstance(typespec, tuple):
            return (typespec, False)

        if len(typespec) != 2:
            raise FieldError('invalid typespec: %s' % str(typespec))

        if typespec[1] is not None:
            return (typespec, False)
        else:
            return (typespec[0], True)

    def _check_type(self, value, typespec):
        # Extract the type information and check if None is allowed
        typespec, allow_none = self._extract_typeinfo(typespec)
        if value is None and allow_none:
            return True

        if not isinstance(typespec, tuple):
            # we need to make a special check if typespec == callable
            return (callable(value)
                    if typespec == callable else isinstance(value, typespec))

        if len(typespec) != 2:
            raise FieldError('invalid typespec: %s' % str(typespec))

        container_type, element_type = typespec
        if not isinstance(value, container_type):
            return False

        if issubclass(container_type, collections.abc.Sequence):
            if isinstance(element_type, tuple) and len(element_type) == 1:
                # non-uniformly typed container
                elem_types = element_type[0]
                if len(value) != len(elem_types):
                    return False

                for v, t in zip(value, elem_types):
                    if not self._check_type(v, t):
                        return False
            else:
                # uniformly typed container
                for v in value:
                    if not self._check_type(v, element_type):
                        return False

        elif issubclass(container_type, collections.abc.Set):
            for v in value:
                if not self._check_type(v, element_type):
                    return False

        elif issubclass(container_type, collections.abc.Mapping):
            if len(element_type) != 2:
                raise FieldError('invalid mapping typespec: %s' %
                                 str(element_type))

            key_type, value_type = element_type
            for k, v in value.items():
                if not self._check_type(k, key_type):
                    return False
                if not self._check_type(v, value_type):
                    return False
        else:
            return False

        return True


class AlphanumericField(TypedField):
    """Stores an alphanumeric string ([A-Za-z0-9_])"""

    def __init__(self, fieldname, allow_none=False):
        super().__init__(fieldname, str, allow_none)

    def __set__(self, obj, value):
        if value is not None:
            if not isinstance(value, str):
                raise FieldError('attempt to set an alphanumeric field '
                                 'with a non-string value')

            # Check if the string is properly formatted
            if not re.fullmatch('\w+', value, re.ASCII):
                raise FieldError('Attempt to set an alphanumeric field '
                                 'with a non-alphanumeric value')

        super().__set__(obj, value)


class NonWhitespaceField(TypedField):
    """Stores a string without any whitespace"""

    def __init__(self, fieldname, allow_none=False):
        super().__init__(fieldname, str, allow_none)

    def __set__(self, obj, value):
        if value is not None:
            if not isinstance(value, str):
                raise FieldError('Attempt to set a string field '
                                 'with a non-string value')

            if not re.fullmatch('\S+', value, re.ASCII):
                raise FieldError('Attempt to set a non-whitespace field '
                                 'with a string containing whitespace')

        super().__set__(obj, value)


class StringField(TypedField):
    """Stores a standard string object"""

    def __init__(self, fieldname, allow_none=False):
        super().__init__(fieldname, str, allow_none)


class IntegerField(TypedField):
    """Stores an integer object"""

    def __init__(self, fieldname, allow_none=False):
        super().__init__(fieldname, numbers.Integral, allow_none)


class BooleanField(TypedField):
    """Stores a boolean object"""

    def __init__(self, fieldname, allow_none=False):
        super().__init__(fieldname, bool, allow_none)


class TypedListField(AggregateTypeField):
    """Stores a list of objects of the same type"""

    def __init__(self, fieldname, elemtype, allow_none=False):
        super().__init__(
            fieldname, (collections.abc.Sequence, elemtype), allow_none)


class TypedSetField(AggregateTypeField):
    """Stores a list of objects of the same type"""

    def __init__(self, fieldname, elemtype, allow_none=False):
        super().__init__(
            fieldname, (collections.abc.Set, elemtype), allow_none)


class TypedDictField(AggregateTypeField):
    """Stores a list of objects of the same type"""

    def __init__(self, fieldname, keytype, valuetype, allow_none=False):
        super().__init__(fieldname,
                         (collections.abc.Mapping, (keytype, valuetype)),
                         allow_none)


class CopyOnWriteField(Field):
    """Holds a copy of the variable that is assigned to it the first time"""

    def __set__(self, obj, value):
        super().__set__(obj, copy.deepcopy(value))


class ReadOnlyField(Field):
    """Holds a read-only field. Attempts to set it will raise an exception"""

    def __init__(self, value):
        super().__init__('_readonly_')
        self._value = value

    def __get__(self, obj, objtype):
        return self._value

    def __set__(self, obj, value):
        raise FieldError('attempt to set a read-only variable')


class SanityPatternField(AggregateTypeField):
    """Stores a sanity or performance patterns field.

    This is a special dictionary that allows a special entry for calling a
    callback function when eof is matched
    """

    def __init__(self, fieldname, allow_none=False):
        # The type of the outer dictionary
        self._outer_typespec = (dict, (str, object))

        # The type of the inner dictionary, excluding the special '\e' entries
        self._inner_typespec = (dict, (
            str, (list, (tuple, ((str, callable, callable),))))
        )
        super().__init__(fieldname, self._inner_typespec, allow_none)

    def __set__(self, obj, value):
        if value is None and self._allow_none:
            # Call directly Field's __set__() method; no need for further type
            # checking
            Field.__set__(self, obj, value)
            return

        user_deprecation_warning(
            'This syntax for sanity and performance checking is deprecated. '
            'Please have a look in the ReFrame tutorial.'
        )

        # Check first the outer dictionary
        if not self._check_type(value, self._outer_typespec):
            raise FieldError('attempt to set a sanity pattern field '
                             'with an invalid dictionary object')

        # For the inner dictionary, we need special treatment for the '\e'
        # character
        for k, v in value.items():
            # Check the special entry '\e' first
            eof_handler = None
            if '\e' in v.keys():
                eof_handler = v['\e']
                if not callable(eof_handler):
                    raise FieldError("special key '\e' does not "
                                     "refer to a callable")

                # Remove the value temporarily
                del v['\e']

            try:
                if not self._check_type(v, self._inner_typespec):
                    raise FieldError('attempt to set a sanity pattern field'
                                     'with an invalid dictionary object')
            finally:
                # Restore '\e'
                if eof_handler is not None:
                    v['\e'] = eof_handler

        # All type checking is done; just set the value
        Field.__set__(self, obj, value)


class TimerField(AggregateTypeField):
    """Stores a timer in the form of a tuple '(hh, mm, ss)'"""

    def __init__(self, fieldname, allow_none=False):
        super().__init__(fieldname, (tuple, ((int, int, int),)), allow_none)

    def __set__(self, obj, value):
        if not self._check_type_ext(value):
            raise FieldError('attempt to set a timer field with a wrong type')

        if value is not None:
            # Check also the values for minutes and seconds
            h, m, s = value
            if h < 0 or m < 0 or s < 0:
                raise FieldError('timer field must have non-negative values')

            if m > 59 or s > 59:
                raise FieldError('minutes and seconds in a timer field '
                                 'must not exceed 59')

        # Call Field's __set__() method, type checking is already performed
        Field.__set__(self, obj, value)


# FIXME: This is not probably the best place for this to go
class ScopedDict(UserDict):
    """This is a special dict that imposes scopes on its keys.

    When a key is not found it will be searched up in the scope hierarchy."""

    def __init__(self, mapping={}, scope_sep=':', global_scope='*'):
        """Initialize a ScopedDict

        Keyword arguments:
        mapping -- A two-level mapping of the form
                   mapping = {
                        scope1: {k1: v1, k2: v2},
                        scope2: {k1: v1, k3: v3}
                   }

                   Both the scope keys and the actual dictionary keys must be
                   strings, otherwise TypeError will be raised

        scope_sep -- character that separates the scopes
        global_scope -- key to look up for the global scope"""
        super().__init__(mapping)
        self._scope_sep = scope_sep
        self._global_scope = global_scope

    @property
    def scope_separator(self):
        return self._scope_sep

    @property
    def global_scope_mark(self):
        return self._global_scope

    def update(self, other):
        if not isinstance(other, collections.abc.Mapping):
            raise TypeError('ScopedDict may only be initialized '
                            'from a mapping type')

        for scope, scope_dict in other.items():
            self._check_scope_type(scope, scope_dict)
            self.data.setdefault(scope, {})
            for k, v in scope_dict.items():
                self.data[scope][k] = v

    def __str__(self):
        # just return the internal dictionary
        return str(self.data)

    def _check_scope_type(self, key, value):
        if not isinstance(key, str):
            raise TypeError('scope keys in a ScopedDict must be strings')

        if not isinstance(value, collections.abc.Mapping):
            raise TypeError('scope namespaces must be mappings')

        for k in value.keys():
            if not isinstance(k, str):
                raise TypeError('keys must be strings')

    def _keyinfo(self, key):
        key_parts = key.rsplit(self._scope_sep, maxsplit=1)
        if len(key_parts) == 2:
            return (key_parts[0], key_parts[1])
        else:
            return (self._global_scope, key_parts[0])

    def _parent_scope(self, scope):
        scope_parts = scope.rsplit(':', maxsplit=1)[:-1]
        return scope_parts[0] if scope_parts else self._global_scope

    def _lookup(self, key):
        scope, lookup_key = self._keyinfo(key)
        while scope != self._global_scope:
            if scope in self.data and lookup_key in self.data[scope]:
                return self.data[scope][lookup_key]

            scope = self._parent_scope(scope)

        # last chance to find the key
        if scope in self.data and lookup_key in self.data[scope]:
            return self.data[scope][lookup_key]

        raise KeyError(str(key))

    def __iter__(self):
        for scope, scope_dict in self.data.items():
            for k in scope_dict.keys():
                yield self._scope_sep.join([scope, k])

    def __contains__(self, key):
        try:
            self._lookup(key)
        except KeyError:
            return False
        else:
            return True

    def __getitem__(self, key):
        try:
            return self._lookup(key)
        except KeyError:
            return self.__missing__(key)

    def __setitem__(self, key, value):
        scope, lookup_key = self._keyinfo(key)
        if scope not in self.data:
            # create the scope if does not exist
            self.data[scope] = {}

        self.data[scope][lookup_key] = value

    def __delitem__(self, key):
        """Deletes either a key or a scope if key refers to a scope.

        If key refers to scope, the whole scope entry will be deleted. If not,
        the exact key requested will be deleted. No key resolution will be
        performed."""
        if key in self.data:
            # key is a scope
            del self.data[key]
        else:
            scope, lookup_key = self._keyinfo(key)
            del self.data[scope][lookup_key]

    def __missing__(self, key):
        raise KeyError(str(key))


class ScopedDictField(AggregateTypeField):
    """Stores a ScopedDict with a specific type

    It also handles implicit conversions from ordinary dicts."""

    def __init__(self, fieldname, valuetype, allow_none=False):
        super().__init__(
            fieldname, (dict, (str, (dict, (str, valuetype)))), allow_none
        )

    def __set__(self, obj, value):
        if isinstance(value, ScopedDict):
            # value is already a ScopedDict
            Field.__set__(self, obj, value)
        else:
            # Try to convert value to a ScopedDict
            if not self._check_type_ext(value):
                raise FieldError('attempt to set a ScopedDict '
                                 'of different type. Required typespec: %s' %
                                 str(self._typespec))

            # Call Field's __set__() method, type checking is already performed
            Field.__set__(self, obj,
                          ScopedDict(value) if value is not None else value)
