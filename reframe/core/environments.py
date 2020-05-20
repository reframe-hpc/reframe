# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import collections
import os

import reframe.core.fields as fields
import reframe.utility as util
import reframe.utility.os_ext as os_ext
import reframe.utility.typecheck as typ


class Environment:
    '''This class abstracts away an environment to run regression tests.

    It is simply a collection of modules to be loaded and environment variables
    to be set when this environment is loaded by the framework.

    .. warning::
       Users may not create :class:`Environment` objects directly.
    '''

    def __init__(self, name, modules=None, variables=None):
        modules = modules or []
        variables = variables or []
        self._name = name
        self._modules = list(modules)
        self._variables = collections.OrderedDict(variables)

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
        return util.SequenceView(self._modules)

    @property
    def variables(self):
        '''The environment variables associated with this environment.

        :type: :class:`OrderedDict[str, str]`
        '''
        return util.MappingView(self._variables)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        return (self.name == other.name and
                set(self.modules) == set(other.modules) and
                self.variables == other.variables)

    def __str__(self):
        return self.name

    def __repr__(self):
        return (f'{type(self).__name__}('
                f'name={self._name!r}, '
                f'modules={self._modules!r}, '
                f'variables={list(self._variables.items())!r})')


class _EnvironmentSnapshot(Environment):
    '''An environment snapshot.'''

    def __init__(self, name='env_snapshot'):
        super().__init__(name, [], os.environ.items())

    def restore(self):
        '''Restore this environment snapshot.'''
        os.environ.clear()
        os.environ.update(self._variables)

    def __eq__(self, other):
        if not isinstance(other, Environment):
            return NotImplemented

        # Order of variables is not important when comparing snapshots
        for k, v in self.variables.items():
            if other.variables[k] != v:
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

    _cc = fields.TypedField('_cc', str)
    _cxx = fields.TypedField('_cxx', str)
    _ftn = fields.TypedField('_ftn', str)
    _cppflags = fields.TypedField('_cppflags', typ.List[str])
    _cflags = fields.TypedField('_cflags', typ.List[str])
    _cxxflags = fields.TypedField('_cxxflags', typ.List[str])
    _fflags = fields.TypedField('_fflags', typ.List[str])
    _ldflags = fields.TypedField('_ldflags', typ.List[str])

    def __init__(self,
                 name,
                 modules=None,
                 variables=None,
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
        super().__init__(name, modules, variables)
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
