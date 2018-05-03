import collections.abc
import os
import importlib.util
from collections import UserDict


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
        scope, lookup_key = self._keyinfo(key)
        if scope in self.data and lookup_key != key:
            del self.data[scope][lookup_key]
        elif key in self.data:
            # key is a scope
            del self.data[key]
        else:
            raise KeyError(str(key))

    def __missing__(self, key):
        raise KeyError(str(key))


def import_module_from_file(filename):
    filename = os.path.expandvars(filename)

    # Figure out a reasonable module name
    # FIXME: we are not treating specially `__init__.py`
    barename, _ = os.path.splitext(filename)
    if os.path.isabs(barename):
        module_name = os.path.basename(barename)
    else:
        module_name = barename.replace('/', '.')

    spec = importlib.util.spec_from_file_location(module_name, filename)
    if spec is None:
        raise ImportError("No module named '%s'" % module_name,
                          name=module_name, path=filename)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
