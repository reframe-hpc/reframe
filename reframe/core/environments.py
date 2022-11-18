# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import collections
import os

import reframe.core.fields as fields
import reframe.utility as util
import reframe.utility.jsonext as jsonext
import reframe.utility.typecheck as typ
from reframe.core.warnings import user_deprecation_warning


def normalize_module_list(modules):
    '''Normalize module list.

    :meta private:
    '''
    ret = []
    for m in modules:
        if isinstance(m, str):
            ret.append({'name': m, 'collection': False, 'path': None})
        else:
            ret.append(m)

    return ret


class Environment(jsonext.JSONSerializable):
    '''This class abstracts away an environment to run regression tests.

    It is simply a collection of modules to be loaded and environment variables
    to be set when this environment is loaded by the framework.

    .. warning::
       Users may not create :class:`Environment` objects directly.
    '''

    def __init__(self, name, modules=None, env_vars=None,
                 extras=None, features=None):
        modules = modules or []
        env_vars = env_vars or []
        self._name = name
        self._modules = normalize_module_list(modules)
        self._module_names = [m['name'] for m in self._modules]

        # Convert values of env_vars to strings before storing
        if isinstance(env_vars, dict):
            env_vars = env_vars.items()

        self._env_vars = collections.OrderedDict()
        for k, v in env_vars:
            self._env_vars[k] = str(v)

        self._extras = extras or {}
        self._features = features or []

    @property
    def name(self):
        '''The name of this environment.

        :type: :class:`str`
        '''
        return self._name

    @property
    def modules(self):
        '''The modules associated with this environment.

        :type: :class:`List[str]`
        '''
        return util.SequenceView(self._module_names)

    @property
    def modules_detailed(self):
        '''A view of the modules associated with this environment in a detailed
        format.

        Each module is represented as a dictionary with the following
        attributes:

        - ``name``: the name of the module.
        - ``collection``: :class:`True` if the module name refers to a module
          collection.

        :type: :class:`List[Dict[str, object]]`

        .. versionadded:: 3.3

        '''

        return util.SequenceView(self._modules)

    @property
    def env_vars(self):
        '''The environment variables associated with this environment.

        :type: :class:`OrderedDict[str, str]`

        .. versionadded:: 4.0.0
        '''
        return util.MappingView(self._env_vars)

    @property
    def variables(self):
        '''The environment variables associated with this environment.

        .. deprecated:: 4.0.0
           Please :attr:`env_vars` instead.
        '''
        user_deprecation_warning("the 'variables' attribute is deprecated; "
                                 "please use the 'env_vars' instead")
        return util.MappingView(self._env_vars)

    @property
    def extras(self):
        '''User defined properties specified in the configuration.

        .. versionadded:: 3.9.1

        :type: :class:`Dict[str, object]`
        '''

        return self._extras

    @property
    def features(self):
        '''Used defined features specified in the configuration.

        .. versionadded:: 3.11.0

        :type: :class:`List[str]`
        '''
        return self._features

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        if (self.name != other.name or
            set(self.modules) != set(other.modules)):
            return False

        # Env. variables are checked against their string representation
        for kv0, kv1 in zip(self.env_vars.items(),
                            other.env_vars.items()):
            k0, v0 = kv0
            k1, v1 = kv1
            if k0 != k1 or str(v0) != str(v1):
                return False

        return True

    def __str__(self):
        return self.name

    def __repr__(self):
        return (f'{type(self).__name__}('
                f'name={self._name!r}, '
                f'modules={self._modules!r}, '
                f'env_vars={list(self._env_vars.items())!r}, '
                f'extras={self._extras!r}, features={self._features!r})')


class _EnvironmentSnapshot(Environment):
    '''An environment snapshot.'''

    def __init__(self, name='env_snapshot'):
        super().__init__(name, [], os.environ.items())

    def restore(self):
        '''Restore this environment snapshot.'''
        os.environ.clear()
        os.environ.update(self._env_vars)

    def __eq__(self, other):
        if not isinstance(other, Environment):
            return NotImplemented

        # Order of env. variables is not important when comparing snapshots
        for k, v in self.env_vars.items():
            if other.env_vars[k] != v:
                return False

        return self.name == other.name


def snapshot():
    '''Create an environment snapshot

    :returns: An instance of :class:`_EnvironmentSnapshot`.
    '''
    return _EnvironmentSnapshot()


class ProgEnvironment(Environment):
    '''A class representing a programming environment.

    This type of environment adds also properties for retrieving the compiler
    and compilation flags.

    .. warning::
       Users may not create :class:`ProgEnvironment` objects directly.
    '''

    _cc = fields.TypedField(str)
    _cxx = fields.TypedField(str)
    _ftn = fields.TypedField(str)
    _cppflags = fields.TypedField(typ.List[str])
    _cflags = fields.TypedField(typ.List[str])
    _cxxflags = fields.TypedField(typ.List[str])
    _fflags = fields.TypedField(typ.List[str])
    _ldflags = fields.TypedField(typ.List[str])

    def __init__(self,
                 name,
                 modules=None,
                 env_vars=None,
                 extras=None,
                 features=None,
                 cc='cc',
                 cxx='CC',
                 ftn='ftn',
                 nvcc='nvcc',
                 cppflags=None,
                 cflags=None,
                 cxxflags=None,
                 fflags=None,
                 ldflags=None,
                 **kwargs):
        super().__init__(name, modules, env_vars, extras, features)
        self._cc = cc
        self._cxx = cxx
        self._ftn = ftn
        self._nvcc = nvcc
        self._cppflags = cppflags or []
        self._cflags   = cflags   or []
        self._cxxflags = cxxflags or []
        self._fflags   = fflags   or []
        self._ldflags  = ldflags  or []

    @property
    def cc(self):
        '''The C compiler of this programming environment.

        :type: :class:`str`
        '''
        return self._cc

    @property
    def cxx(self):
        '''The C++ compiler of this programming environment.

        :type: :class:`str`
        '''
        return self._cxx

    @property
    def ftn(self):
        '''The Fortran compiler of this programming environment.

        :type: :class:`str`
        '''
        return self._ftn

    @property
    def cppflags(self):
        '''The preprocessor flags of this programming environment.

        :type: :class:`List[str]`
        '''
        return self._cppflags

    @property
    def cflags(self):
        '''The C compiler flags of this programming environment.

        :type: :class:`List[str]`
        '''
        return self._cflags

    @property
    def cxxflags(self):
        '''The C++ compiler flags of this programming environment.

        :type: :class:`List[str]`
        '''
        return self._cxxflags

    @property
    def fflags(self):
        '''The Fortran compiler flags of this programming environment.

        :type: :class:`List[str]`
        '''
        return self._fflags

    @property
    def ldflags(self):
        '''The linker flags of this programming environment.

        :type: :class:`List[str]`
        '''
        return self._ldflags

    @property
    def nvcc(self):
        return self._nvcc
