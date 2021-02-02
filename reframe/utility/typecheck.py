# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
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


class _TypeFactory(abc.ABCMeta):
    def register_subtypes(cls):
        for t in cls._subtypes:
            cls.register(t)


# Metaclasses that implement the isinstance logic for the different aggregate
# types

class _ContainerType(_TypeFactory):
    '''A metaclass for containers with uniformly typed elements.'''

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        cls._elem_type = None
        cls._bases = bases
        cls._namespace = namespace
        cls.register_subtypes()

    def __instancecheck__(cls, inst):
        if not issubclass(type(inst), cls):
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

        ret = _ContainerType('%s[%s]' % (cls.__name__, elem_type.__name__),
                             cls._bases, cls._namespace)
        ret._elem_type = elem_type
        ret.register_subtypes()
        cls.register(ret)
        return ret


class _TupleType(_ContainerType):
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
        ret.register_subtypes()
        cls.register(ret)
        return ret


class _MappingType(_TypeFactory):
    '''A metaclass for type checking mapping types.'''

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        cls._key_type = None
        cls._value_type = None
        cls._bases = bases
        cls._namespace = namespace
        cls.register_subtypes()

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
        ret.register_subtypes()
        cls.register(ret)
        return ret


class _StrType(_ContainerType):
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
        ret.register_subtypes()
        cls.register(ret)
        return ret


class Dict(metaclass=_MappingType):
    _subtypes = (dict,)


class List(metaclass=_ContainerType):
    _subtypes = (list,)


class Set(metaclass=_ContainerType):
    _subtypes = (set,)


class Str(metaclass=_StrType):
    _subtypes = (str,)


class Tuple(metaclass=_TupleType):
    _subtypes = (tuple,)
