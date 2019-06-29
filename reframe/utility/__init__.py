import collections
import functools
import importlib
import importlib.util
import itertools
import os
import re
import sys
import types

from collections import UserDict


def _get_module_name(filename):
    barename, _ = os.path.splitext(filename)
    if os.path.basename(filename) == '__init__.py':
        barename = os.path.dirname(filename)

    if os.path.isabs(barename):
        raise AssertionError('BUG: _get_module_name() '
                             'accepts relative paths only')

    if filename.startswith('..'):
        return os.path.basename(barename)
    else:
        return barename.replace(os.sep, '.')


def _do_import_module_from_file(filename, module_name=None):
    module_name = module_name or _get_module_name(filename)
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, filename)
    if spec is None:
        raise ImportError("No module named '%s'" % module_name,
                          name=module_name, path=filename)

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def import_module_from_file(filename):
    """Import module from file."""

    # Expand and sanitize filename
    filename = os.path.abspath(os.path.expandvars(filename))
    if os.path.isdir(filename):
        filename = os.path.join(filename, '__init__.py')

    # Express filename relative to reframe
    rel_filename = os.path.relpath(filename, sys.path[0])
    module_name = _get_module_name(rel_filename)
    if rel_filename.startswith('..'):
        # We cannot use the standard Python import mechanism here, because the
        # module to import is outside the top-level package
        return _do_import_module_from_file(filename, module_name)

    return importlib.import_module(module_name)


def allx(iterable):
    """Same as the built-in all, except that it returns :class:`False` if
    ``iterable`` is empty.
    """

    # Generators must be treated specially, because there is no way to get
    # their size without consuming their elements.
    if isinstance(iterable, types.GeneratorType):
        try:
            head = next(iterable)
        except StopIteration:
            return False
        else:
            return all(itertools.chain([head], iterable))

    if not isinstance(iterable, collections.abc.Iterable):
        raise TypeError("'%s' object is not iterable" %
                        iterable.__class__.__name__)

    return all(iterable) if iterable else False


def decamelize(s, delim='_'):
    """Decamelize the string ``s``.

    For example, ``MyBaseClass`` will be converted to ``my_base_class``.
    The delimiter may be changed by setting the ``delim`` argument.
    """

    if not isinstance(s, str):
        raise TypeError('decamelize() requires a string argument')

    if not s:
        return ''

    return re.sub(r'([a-z])([A-Z])', r'\1%s\2' % delim, s).lower()


def toalphanum(s):
    """Convert string ``s`` be replacing any non-alphanumeric character with
    ``_``.
    """

    if not isinstance(s, str):
        raise TypeError('toalphanum() requires a string argument')

    if not s:
        return ''

    return re.sub(r'\W', '_', s)


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

        If key refers to both a scope and a key, the key will be deleted.
        If key refers to scope, the whole scope entry will be deleted.
        If not, the exact key requested will be deleted.
        No key resolution will be performed."""
        scope, lookup_key = self._keyinfo(key)
        if scope in self.data and lookup_key in self.data[scope]:
            del self.data[scope][lookup_key]
        elif key in self.data:
            # key is a scope
            del self.data[key]
        else:
            raise KeyError(str(key))

    def __missing__(self, key):
        raise KeyError(str(key))


@functools.total_ordering
class OrderedSet(collections.abc.MutableSet):
    """An ordered set."""

    def __init__(self, *args):
        # We need to allow construction without arguments
        if not args:
            iterable = []
        elif len(args) == 1:
            iterable = args[0]
        else:
            # We use the exact same error message as for the built-in set
            raise TypeError('%s expected at most 1 arguments, got %s' %
                            type(self).__name__, len(args))

        if not isinstance(iterable, collections.abc.Iterable):
            raise TypeError("'%s' object is not iterable" %
                            type(iterable).__name__)

        # We implement an ordered set through the keys of an OrderedDict;
        # its values are all set to None
        self.__data = collections.OrderedDict(
            itertools.zip_longest(iterable, [], fillvalue=None)
        )

    def __repr__(self):
        vals = self.__data.keys()
        if not vals:
            return type(self).__name__ + '()'
        else:
            return '{' + ', '.join(str(v) for v in vals) + '}'

    # Container i/face
    def __contains__(self, item):
        return item in self.__data

    def __iter__(self):
        return iter(self.__data)

    def __len__(self):
        return len(self.__data)

    # Set i/face
    #
    # Note on the complexity of the operators
    #
    # In every case below we first construct a set from the internal ordered
    # dictionary's keys and then apply the operator. This step's complexity is
    # O(len(self.__data.keys())). Since the complexity of the standard set
    # operators are at the order of magnitute of the lenghts of the operands
    # (ranging from O(min(len(a), len(b))) to O(len(a) + len(b))), this step
    # does not change the complexity class; it just changes the constant
    # factor.
    #
    def __eq__(self, other):
        if not isinstance(other, collections.abc.Set):
            return NotImplemented

        return set(self.__data.keys()) == other

    def __gt__(self, other):
        if not isinstance(other, collections.abc.Set):
            return NotImplemented

        return set(self.__data.keys()) > other

    def __and__(self, other):
        if not isinstance(other, collections.abc.Set):
            return NotImplemented

        return set(self.__data.keys()) & other

    def __or__(self, other):
        if not isinstance(other, collections.abc.Set):
            return NotImplemented

        return set(self.__data.keys()) | other

    def __sub__(self, other):
        if not isinstance(other, collections.abc.Set):
            return NotImplemented

        return set(self.__data.keys()) - other

    def __xor__(self, other):
        if not isinstance(other, collections.abc.Set):
            return NotImplemented

        return set(self.__data.keys()) ^ other

    def isdisjoint(self, other):
        if not isinstance(other, collections.abc.Set):
            return NotImplemented

        return set(self.__data.keys()).isdisjoint(other)

    def issubset(self, other):
        return self <= other

    def issuperset(self, other):
        return self >= other

    def symmetric_difference(self, other):
        return self ^ other

    def union(self, *others):
        ret = type(self)(self)
        for s in others:
            ret |= s

        return ret

    def intersection(self, *others):
        ret = type(self)(self)
        for s in others:
            ret &= s

        return ret

    def difference(self, *others):
        ret = type(self)(self)
        for s in others:
            ret -= s

        return ret

    # MutableSet i/face

    def add(self, elem):
        self.__data[elem] = None

    def remove(self, elem):
        del self.__data[elem]

    def discard(self, elem):
        try:
            self.remove(elem)
        except KeyError:
            pass

    def pop(self):
        return self.__data.popitem()[0]

    def clear(self):
        self.__data.clear()

    def __ior__(self, other):
        if not isinstance(other, collections.abc.Set):
            return NotImplemented

        for e in other:
            self.add(e)

        return self

    def __iand__(self, other):
        if not isinstance(other, collections.abc.Set):
            return NotImplemented

        discard_list = [e for e in self if e not in other]
        for e in discard_list:
            self.discard(e)

        return self

    def __isub__(self, other):
        if not isinstance(other, collections.abc.Set):
            return NotImplemented

        for e in other:
            self.discard(e)

        return self

    def __ixor__(self, other):
        if not isinstance(other, collections.abc.Set):
            return NotImplemented

        discard_list = [e for e in self if e in other]
        for e in discard_list:
            self.discard(e)

        return self

    # Other functions
    def __reversed__(self):
        return reversed(self.__data.keys())


class SequenceView(collections.abc.Sequence):
    """A read-only view of a sequence."""

    def __init__(self, container):
        if not isinstance(container, collections.abc.Sequence):
            raise TypeError('container must be of type Sequence')

        self.__container = container

    def count(self, *args, **kwargs):
        return self.__container.count(*args, **kwargs)

    def index(self, *args, **kwargs):
        return self.__container.index(*args, **kwargs)

    def __contains__(self, *args, **kwargs):
        return self.__container.__contains__(*args, **kwargs)

    def __getitem__(self, *args, **kwargs):
        return self.__container.__getitem__(*args, **kwargs)

    def __iter__(self, *args, **kwargs):
        return self.__container.__iter__(*args, **kwargs)

    def __len__(self, *args, **kwargs):
        return self.__container.__len__(*args, **kwargs)

    def __reversed__(self, *args, **kwargs):
        return self.__container.__reversed__(*args, **kwargs)

    def __add__(self, other):
        if not isinstance(other, collections.abc.Sequence):
            return NotImplemented

        return SequenceView(self.__container + other)

    def __iadd__(self, other):
        return NotImplemented

    def __eq__(self, other):
        if not isinstance(other, collections.abc.Sequence):
            return NotImplemented

        return self.__container == other


class MappingView(collections.abc.Mapping):
    """A read-only view of a mapping."""

    def __init__(self, mapping):
        if not isinstance(mapping, collections.abc.Mapping):
            raise TypeError('container must be of type Mapping')

        self.__mapping = mapping

    def get(self, *args, **kwargs):
        return self.__mapping.get(*args, **kwargs)

    def keys(self, *args, **kwargs):
        return self.__mapping.keys(*args, **kwargs)

    def items(self, *args, **kwargs):
        return self.__mapping.items(*args, **kwargs)

    def values(self, *args, **kwargs):
        return self.__mapping.values(*args, **kwargs)

    def __contains__(self, *args, **kwargs):
        return self.__mapping.__contains__(*args, **kwargs)

    def __getitem__(self, *args, **kwargs):
        return self.__mapping.__getitem__(*args, **kwargs)

    def __iter__(self, *args, **kwargs):
        return self.__mapping.__iter__(*args, **kwargs)

    def __len__(self, *args, **kwargs):
        return self.__mapping.__len__(*args, **kwargs)

    def __eq__(self, *args, **kwargs):
        return self.__mapping.__eq__(*args, **kwargs)

    def __ne__(self, *args, **kwargs):
        return self.__mapping.__ne__(*args, **kwargs)
