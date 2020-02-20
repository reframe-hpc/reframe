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
from reframe.core.runtime import runtime


class Environment:
    '''This class abstracts away an environment to run regression tests.

    It is simply a collection of modules to be loaded and environment variables
    to be set when this environment is loaded by the framework.
    '''
    name = fields.TypedField('name', typ.Str[r'(\w|-)+'])
    modules = fields.TypedField('modules', typ.List[str])
    variables = fields.TypedField('variables', typ.Dict[str, str])

    def __init__(self, name, modules=[], variables=[]):
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

        :type: :class:`list` of :class:`str`
        '''
        return util.SequenceView(self._modules)

    @property
    def variables(self):
        '''The environment variables associated with this environment.

        :type: dictionary of :class:`str` keys/values.
        '''
        return util.MappingView(self._variables)

    @property
    def is_loaded(self):
        ''':class:`True` if this environment is loaded,
        :class:`False` otherwise.
        '''
        is_module_loaded = runtime().modules_system.is_module_loaded
        return (all(map(is_module_loaded, self._modules)) and
                all(os.environ.get(k, None) == os_ext.expandvars(v)
                    for k, v in self._variables.items()))

    def details(self):
        '''Return a detailed description of this environment.'''
        variables = '\n'.join(' '*8 + '- %s=%s' % (k, v)
                              for k, v in self.variables.items())
        lines = [
            self._name + ':',
            '    modules: ' + ', '.join(self.modules),
            '    variables:' + ('\n' if variables else '') + variables
        ]
        return '\n'.join(lines)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        return (self.name == other.name and
                set(self.modules) == set(other.modules) and
                self.variables == other.variables)

    def __str__(self):
        return self.name

    def __repr__(self):
        ret = "{0}(name='{1}', modules={2}, variables={3})"
        return ret.format(type(self).__name__, self.name,
                          self.modules, self.variables)


class _EnvironmentSnapshot(Environment):
    def __init__(self, name='env_snapshot'):
        super().__init__(name,
                         runtime().modules_system.loaded_modules(),
                         os.environ.items())

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

        return (self.name == other.name and
                set(self.modules) == set(other.modules))


def snapshot():
    '''Create an environment snapshot'''
    return _EnvironmentSnapshot()


def load(*environs):
    '''Load environments in the current Python context.

    Returns a tuple containing a snapshot of the environment at entry to this
    function and a list of shell commands required to load ``environs``.
    '''
    env_snapshot = snapshot()
    commands = []
    rt = runtime()
    for env in environs:
        for m in env.modules:
            conflicted = rt.modules_system.load_module(m, force=True)
            for c in conflicted:
                commands += rt.modules_system.emit_unload_commands(c)

            commands += rt.modules_system.emit_load_commands(m)

        for k, v in env.variables.items():
            os.environ[k] = os_ext.expandvars(v)
            commands.append('export %s=%s' % (k, v))

    return env_snapshot, commands


def emit_load_commands(*environs):
    env_snapshot, commands = load(*environs)
    env_snapshot.restore()
    return commands


class temp_environment:
    '''Context manager to temporarily change the environment.'''

    def __init__(self, modules=[], variables=[]):
        self._modules = modules
        self._variables = variables

    def __enter__(self):
        new_env = Environment('_rfm_temp_env', self._modules, self._variables)
        self._environ_save, _ = load(new_env)
        return new_env

    def __exit__(self, exc_type, exc_value, traceback):
        self._environ_save.restore()


class ProgEnvironment(Environment):
    '''A class representing a programming environment.

    This type of environment adds also attributes for setting the compiler and
    compilation flags.

    If compilation flags are set to :class:`None` (the default, if not set
    otherwise in ReFrame's `configuration
    <configure.html#environments-configuration>`__), they are not passed to the
    ``make`` invocation.

    If you want to disable completely the propagation of the compilation flags
    to the ``make`` invocation, even if they are set, you should set the
    :attr:`propagate` attribute to :class:`False`.
    '''

    _cc = fields.TypedField('_cc', str)
    _cxx = fields.TypedField('_cxx', str)
    _ftn = fields.TypedField('_ftn', str)
    _cppflags = fields.TypedField('_cppflags', typ.List[str], type(None))
    _cflags = fields.TypedField('_cflags', typ.List[str], type(None))
    _cxxflags = fields.TypedField('_cxxflags', typ.List[str], type(None))
    _fflags = fields.TypedField('_fflags', typ.List[str], type(None))
    _ldflags = fields.TypedField('_ldflags', typ.List[str], type(None))

    def __init__(self,
                 name,
                 modules=[],
                 variables={},
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
        self._cppflags = cppflags
        self._cflags = cflags
        self._cxxflags = cxxflags
        self._fflags = fflags
        self._ldflags = ldflags

    @property
    def cc(self):
        '''The C compiler of this programming environment.

        :type: :class:`str`
        '''
        return self._cc

    @property
    def cxx(self):
        '''The C++ compiler of this programming environment.

        :type: :class:`str` or :class:`None`
        '''
        return self._cxx

    @property
    def ftn(self):
        '''The Fortran compiler of this programming environment.

        :type: :class:`str` or :class:`None`
        '''
        return self._ftn

    @property
    def cppflags(self):
        '''The preprocessor flags of this programming environment.

        :type: :class:`str` or :class:`None`
        '''
        return self._cppflags

    @property
    def cflags(self):
        '''The C compiler flags of this programming environment.

        :type: :class:`str` or :class:`None`
        '''
        return self._cflags

    @property
    def cxxflags(self):
        '''The C++ compiler flags of this programming environment.

        :type: :class:`str` or :class:`None`
        '''
        return self._cxxflags

    @property
    def fflags(self):
        '''The Fortran compiler flags of this programming environment.

        :type: :class:`str` or :class:`None`
        '''
        return self._fflags

    @property
    def ldflags(self):
        '''The linker flags of this programming environment.

        :type: :class:`str` or :class:`None`
        '''
        return self._ldflags

    @property
    def nvcc(self):
        return self._nvcc

    def details(self):
        def format_flags(flags):
            if flags is None:
                return '<None>'
            elif len(flags) == 0:
                return "''"
            else:
                return ' '.join(flags)

        base_details = super().details()
        extra_details = [
            '    CC: %s' % self.cc,
            '    CXX: %s' % self.cxx,
            '    FTN: %s' % self.ftn,
            '    NVCC: %s' % self.nvcc,
            '    CFLAGS: %s' % format_flags(self.cflags),
            '    CXXFLAGS: %s' % format_flags(self.cxxflags),
            '    FFLAGS: %s' % format_flags(self.fflags),
            '    CPPFLAGS: %s' % format_flags(self.cppflags),
            '    LDFLAGS: %s' % format_flags(self.ldflags)
        ]

        return '\n'.join([base_details, '\n'.join(extra_details)])
