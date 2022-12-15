# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

'''Dynamic recursive type checking of collections.

This module defines types for collections, such as lists, dictionaries etc.,
that you can use with the :py:func:`isinstance` builtin function to
recursively type check all the elements of the collection. Suppose you have a
list of integers, suchs as ``[1, 2, 3]``, the following checks should be true:

.. code-block:: python

    l = [1, 2, 3]
    assert isinstance(l, List[int]) == True
    assert isinstance(l, List[float]) == False


Aggregate types can be combined in an arbitrary depth, so that you can type
check any complex data strcture:

.. code-block:: python

    d = {'a': [1, 2], 'b': [3, 4]}
    assert isisntance(d, Dict) == True
    assert isisntance(d, Dict[str, List[int]]) == True


This module offers the following aggregate types:

.. py:data:: List[T]

   A list with elements of type :class:`T`.

.. py:data:: Set[T]

   A set with elements of type :class:`T`.

.. py:data:: Dict[K,V]

   A dictionary with keys of type :class:`K` and values of type :class:`V`.

.. py:data:: Tuple[T]

   A tuple with elements of type :class:`T`.

.. py:data:: Tuple[T1,T2,...,Tn]

   A tuple with ``n`` elements, whose types are exactly :class:`T1`,
   :class:`T2`, ..., :class:`Tn` in that order.


.. py:data:: Str[patt]

   A string type whose members are all the strings matching the regular
   expression ``patt``.


Implementation details
----------------------

Internally, this module leverages metaclasses and the
:py:func:`__isinstancecheck__` method to customize the behaviour of the
:py:func:`isinstance` builtin.

By implementing also the :py:func:`__getitem__` accessor method, this module
follows the look-and-feel of the type hints proposed in `PEP484
<https://www.python.org/dev/peps/pep-0484/>`__. This method returns a new type
that is a subtype of the base container type. Using the facilities of
:py:class:`abc.ABCMeta`, builtin types, such as :py:class:`list`,
:py:class:`str` etc. are registered as subtypes of the base container types
offered by this module. The type hierarchy of the types defined in this module
is the following (example shown for :class:`List`, but it is analogous for
the rest of the types):

.. code-block:: none

          type
            |
            |
            |
          List
        /   |
       /    |
      /     |
    list  List[T]


In the above example :class:`T` may refer to any type, so that
:class:`List[List[int]]` is an instance of :class:`List`, but not an instance
of :class:`List[int]`.

'''

import abc
import re


class ConvertibleType(abc.ABCMeta):
    '''A type that support conversions from other types.

    This is a metaclass that allows classes that use it to support arbitrary
    conversions from other types using a cast-like syntax without having to
    change their constructor:

    .. code-block:: python

       new_obj = convertible_type(another_type)

    For example, a class whose constructor accepts and :class:`int` may need
    to support a cast-from-string conversion. This is particular useful if you
    want a custom-typed test
    :attr:`~reframe.core.pipeline.RegressionMixin.variable` to be able to be
    set from the command line using the :option:`-S` option.

    In order to support such conversions, a class must use this metaclass and
    define a class method, named as :obj:`__rfm_cast_<type>__`, for each of
    the type conversion that needs to support .

    The following is an example of a class :class:`X` that its normal
    constructor accepts two arguments but it also allows conversions from
    string:

    .. code-block:: python

       class X(metaclass=ConvertibleType):
           def __init__(self, x, y):
               self.data = (x, y)

           @classmethod
           def __rfm_cast_str__(cls, s):
               return X(*(int(x) for x in s.split(',', maxsplit=1)))

        assert X(2, 3).data == X('2,3').data

    .. versionadded:: 3.8.0

    '''

    def __call__(cls, *args, **kwargs):
        if len(args) == 1:
            cast_fn_name = f'__rfm_cast_{type(args[0]).__name__}__'
            if hasattr(cls, cast_fn_name):
                cast_fn = getattr(cls, cast_fn_name)
                return cast_fn(args[0])

        return super().__call__(*args, **kwargs)


# Metaclasses that implement the isinstance logic for the different builtin
# container types


class _CompositeType(abc.ABCMeta):
    def __instancecheck__(cls, inst):
        assert hasattr(cls, '_types') and len(cls._types) == 2
        return (issubclass(type(inst), cls._types[0]) or
                issubclass(type(inst), cls._types[1]))


class _InvertedType(abc.ABCMeta):
    def __instancecheck__(cls, inst):
        assert hasattr(cls, '_xtype')
        return not issubclass(type(inst), cls._xtype)


class _BuiltinType(ConvertibleType):
    def __init__(cls, name, bases, namespace):
        # Make sure that the class defines `_type`
        cls._bases = bases
        cls._namespace = namespace
        assert hasattr(cls, '_type')
        cls.register(cls._type)

    def __instancecheck__(cls, inst):
        if hasattr(cls, '_types'):
            return (issubclass(type(inst), cls._types[0]) or
                    issubclass(type(inst), cls._types[1]))

        if hasattr(cls, '_xtype'):
            return not issubclass(type(inst), cls._xtype)

        return issubclass(type(inst), cls)

    def __or__(cls, other):
        new_type = _BuiltinType(f'{cls.__name__}|{other.__name__}',
                                cls._bases, cls._namespace)
        new_type._types = (cls, other)
        return new_type

    def __invert__(cls):
        new_type = _BuiltinType(f'~{cls.__name__}',
                                cls._bases, cls._namespace)
        new_type._xtype = cls
        return new_type


class _SequenceType(_BuiltinType):
    '''A metaclass for containers with uniformly typed elements.'''

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        cls._elem_type = None

    def __instancecheck__(cls, inst):
        if not super().__instancecheck__(inst):
            return False

        if cls._elem_type is None:
            return True

        return all(isinstance(c, cls._elem_type) for c in inst)

    def __getitem__(cls, elem_type):
        if not isinstance(elem_type, type):
            raise TypeError('{0} is not a valid type'.format(elem_type))

        if isinstance(elem_type, tuple):
            raise TypeError('invalid type specification for container type: '
                            'expected ContainerType[elem_type]')

        ret = _SequenceType('%s[%s]' % (cls.__name__, elem_type.__name__),
                            cls._bases, cls._namespace)
        ret._elem_type = elem_type
        cls.register(ret)
        return ret

    def __rfm_cast_str__(cls, s):
        container_type = cls._type
        elem_type = cls._elem_type
        return container_type(elem_type(e) for e in s.split(','))


class _TupleType(_SequenceType):
    '''A metaclass for tuples.

    Tuples may contain uniformly-typed elements or non-uniformly typed ones.
    '''

    def __instancecheck__(cls, inst):
        if not issubclass(type(inst), cls):
            return False

        if cls._elem_type is None:
            return True

        if len(cls._elem_type) == 1:
            # tuple with elements of the same type
            return all(isinstance(c, cls._elem_type[0]) for c in inst)

        # Non-uniformly typed tuple
        if len(inst) != len(cls._elem_type):
            return False

        return all(isinstance(elem, req_type)
                   for req_type, elem in zip(cls._elem_type, inst))

    def __getitem__(cls, elem_types):
        if not isinstance(elem_types, tuple):
            elem_types = (elem_types,)

        for t in elem_types:
            if not isinstance(t, type):
                raise TypeError('{0} is not a valid type'.format(t))

        cls_name = '%s[%s]' % (
            cls.__name__, ','.join(c.__name__ for c in elem_types)
        )
        ret = _TupleType(cls_name, cls._bases, cls._namespace)
        ret._elem_type = elem_types
        cls.register(ret)
        return ret

    def __rfm_cast_str__(cls, s):
        container_type = cls._type
        elem_types = cls._elem_type
        elems = s.split(',')
        if len(elem_types) == 1:
            elem_t = elem_types[0]
            return container_type(elem_t(e) for e in elems)
        elif len(elem_types) != len(elems):
            raise TypeError(f'cannot convert string {s!r} to {cls.__name__!r}')
        else:
            return container_type(
                elem_t(e) for elem_t, e in zip(elem_types, elems)
            )


class _MappingType(_BuiltinType):
    '''A metaclass for type checking mapping types.'''

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        cls._key_type = None
        cls._value_type = None
        cls._bases = bases
        cls._namespace = namespace

    def __instancecheck__(cls, inst):
        if not issubclass(type(inst), cls):
            return False

        if cls._key_type is None and cls._key_type is None:
            return True

        assert cls._key_type is not None and cls._value_type is not None
        has_valid_keys = all(isinstance(k, cls._key_type)
                             for k in inst.keys())
        has_valid_values = all(isinstance(v, cls._value_type)
                               for v in inst.values())
        return has_valid_keys and has_valid_values

    def __getitem__(cls, typespec):
        try:
            key_type, value_type = typespec
        except ValueError:
            raise TypeError(
                'invalid type specification for mapping type: '
                'expected MappingType[key_type, value_type]') from None

        for t in typespec:
            if not isinstance(t, type):
                raise TypeError('{0} is not a valid type'.format(t))

        cls_name = '%s[%s,%s]' % (cls.__name__, key_type.__name__,
                                  value_type.__name__)
        ret = _MappingType(cls_name, cls._bases, cls._namespace)
        ret._key_type = key_type
        ret._value_type = value_type
        cls.register(ret)
        return ret

    def __rfm_cast_str__(cls, s):
        mappping_type = cls._type
        key_type = cls._key_type
        value_type = cls._value_type
        seq = []
        for key_datum in s.split(','):
            try:
                k, v = key_datum.split(':')
            except ValueError:
                # Re-raise as TypeError
                raise TypeError(
                    f'cannot convert string {s!r} to {cls.__name__!r}'
                ) from None

            seq.append((key_type(k), value_type(v)))

        return mappping_type(seq)


class _StrType(_SequenceType):
    '''A metaclass for type checking string types.'''

    def __instancecheck__(cls, inst):
        if not issubclass(type(inst), cls):
            return False

        if cls._elem_type is None:
            return True

        # _elem_type is a regex
        return re.fullmatch(cls._elem_type, inst) is not None

    def __getitem__(cls, patt):
        if not isinstance(patt, str):
            raise TypeError('invalid type specification for string type: '
                            'expected _StrType[regex]')

        ret = _StrType("%s[r'%s']" % (cls.__name__, patt),
                       cls._bases, cls._namespace)
        ret._elem_type = patt
        cls.register(ret)
        return ret

    def __rfm_cast_str__(cls, s):
        if not isinstance(s, cls):
            raise TypeError(f'cannot convert string {s!r} to {cls.__name__!r}')

        return s


class Bool(metaclass=_BuiltinType):
    '''A boolean type accepting implicit conversions from strings.

    This type represents a boolean value but allows implicit conversions from
    :class:`str`. More specifically, the following conversions are supported:

    - The strings ``'yes'``, ``'true'`` and ``'1'`` are converted to ``True``.
    - The strings ``'no'``, ``'false'`` and ``'0'`` are converted to
      ``False``.

    The built-in :class:`bool` type is registered as a subclass of this type.

    Boolean test variables that are meant to be set properly from the command
    line must be declared of this type and not :class:`bool`.

    '''

    _type = bool

    @classmethod
    def __rfm_cast_str__(cls, s):
        if s in ('true', 'yes', '1'):
            return True
        elif s in ('false', 'no', '0'):
            return False

        raise TypeError(f'cannot convert {s!r} to bool')


def make_meta_type(name, cls, metacls=_BuiltinType):
    namespace = metacls.__prepare__(name, ())
    namespace['_type'] = cls
    ret = metacls(name, (), namespace)
    return ret


Dict    = make_meta_type('Dict', dict, _MappingType)
Float   = make_meta_type('Float', float)
Integer = make_meta_type('Integer', int)
List    = make_meta_type('List', list, _SequenceType)
Set     = make_meta_type('Set', set, _SequenceType)
Str     = make_meta_type('Str', str, _StrType)
Tuple   = make_meta_type('Tuple', tuple, _TupleType)
