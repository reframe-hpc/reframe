import abc
import collections
import re


class _TypeFactory(abc.ABCMeta):
    def register_subtypes(cls):
        for t in cls._subtypes:
            cls.register(t)


class ContainerType(_TypeFactory):
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

        ret = ContainerType('%s[%s]' % (cls.__name__, elem_type.__name__),
                            cls._bases, cls._namespace)
        ret._elem_type = elem_type
        ret.register_subtypes()
        cls.register(ret)
        return ret


class TupleType(ContainerType):
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
        ret = TupleType(cls_name, cls._bases, cls._namespace)
        ret._elem_type = elem_types
        ret.register_subtypes()
        cls.register(ret)
        return ret


class MappingType(_TypeFactory):
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
        ret = MappingType(cls_name, cls._bases, cls._namespace)
        ret._key_type = key_type
        ret._value_type = value_type
        ret.register_subtypes()
        cls.register(ret)
        return ret


class StrType(ContainerType):
    def __instancecheck__(cls, inst):
        return False

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
                            'expected StrType[regex]')

        ret = StrType("%s[r'%s']" % (cls.__name__, patt),
                      cls._bases, cls._namespace)
        ret._elem_type = patt
        ret.register_subtypes()
        cls.register(ret)
        return ret


class Dict(metaclass=MappingType):
    _subtypes = (dict,)


class List(metaclass=ContainerType):
    _subtypes = (list,)


class Set(metaclass=ContainerType):
    _subtypes = (set,)


class Str(metaclass=StrType):
    _subtypes = (str,)


class Tuple(metaclass=TupleType):
    _subtypes = (tuple,)
