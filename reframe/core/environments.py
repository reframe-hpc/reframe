import collections
import os

import reframe.core.fields as fields
import reframe.utility as util
import reframe.utility.os_ext as os_ext
import reframe.utility.typecheck as typ
from reframe.core.runtime import runtime


class Environment:
    """This class abstracts away an environment to run regression tests.

    It is simply a collection of modules to be loaded and environment variables
    to be set when this environment is loaded by the framework.
    Users may not create or modify directly environments.
    """
    name = fields.TypedField('name', typ.Str[r'(\w|-)+'])
    modules = fields.TypedField('modules', typ.List[str])
    variables = fields.TypedField('variables', typ.Dict[str, str])

    def __init__(self, name, modules=[], variables=[]):
        self._name = name
        self._modules = list(modules)
        self._variables = collections.OrderedDict(variables)
        self._loaded = False
        self._saved_variables = {}
        self._conflicted = []
        self._preloaded = set()
        self._module_ops = []

    @property
    def name(self):
        """The name of this environment.

        :type: :class:`str`
        """
        return self._name

    @property
    def modules(self):
        """The modules associated with this environment.

        :type: :class:`list` of :class:`str`
        """
        return util.SequenceView(self._modules)

    @property
    def variables(self):
        """The environment variables associated with this environment.

        :type: dictionary of :class:`str` keys/values.
        """
        return util.MappingView(self._variables)

    @property
    def is_loaded(self):
        """:class:`True` if this environment is loaded,
        :class:`False` otherwise.
        """
        is_module_loaded = runtime().modules_system.is_module_loaded
        return (all(map(is_module_loaded, self._modules)) and
                all(os.environ.get(k, None) == os_ext.expandvars(v)
                    for k, v in self._variables.items()))

    def load(self):
        # conflicted module list must be filled at the time of load
        rt = runtime()
        for m in self._modules:
            if rt.modules_system.is_module_loaded(m):
                self._preloaded.add(m)

            conflicted = rt.modules_system.load_module(m, force=True)
            for c in conflicted:
                self._module_ops.append(('u', c))

            self._module_ops.append(('l', m))
            self._conflicted += conflicted

        for k, v in self._variables.items():
            if k in os.environ:
                self._saved_variables[k] = os.environ[k]

            os.environ[k] = os_ext.expandvars(v)

        self._loaded = True

    def unload(self):
        if not self._loaded:
            return

        for k, v in self._variables.items():
            if k in self._saved_variables:
                os.environ[k] = self._saved_variables[k]
            elif k in os.environ:
                del os.environ[k]

        # Unload modules in reverse order
        for m in reversed(self._modules):
            if m not in self._preloaded:
                runtime().modules_system.unload_module(m)

        # Reload the conflicted packages, previously removed
        for m in self._conflicted:
            runtime().modules_system.load_module(m)

        self._loaded = False

    def emit_load_commands(self):
        rt = runtime()
        emit_fn = {
            'l': rt.modules_system.emit_load_commands,
            'u': rt.modules_system.emit_unload_commands
        }
        module_ops = self._module_ops or [('l', m) for m in self._modules]

        # Emit module commands
        ret = []
        for op, m in module_ops:
            ret += emit_fn[op](m)

        # Emit variable set commands
        for k, v in self._variables.items():
            ret.append('export %s=%s' % (k, v))

        return ret

    def emit_unload_commands(self):
        rt = runtime()

        # Invert the logic of module operations, since we are unloading the
        # environment
        emit_fn = {
            'l': rt.modules_system.emit_unload_commands,
            'u': rt.modules_system.emit_load_commands
        }

        ret = []
        for var in self._variables.keys():
            ret.append('unset %s' % var)

        if self._module_ops:
            module_ops = reversed(self._module_ops)
        else:
            module_ops = (('l', m) for m in reversed(self._modules))

        for op, m in module_ops:
            ret += emit_fn[op](m)

        return ret

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        return (self._name == other._name and
                set(self._modules) == set(other._modules) and
                self._variables == other._variables)

    def details(self):
        """Return a detailed description of this environment."""
        variables = '\n'.join(' '*8 + '- %s=%s' % (k, v)
                              for k, v in self.variables.items())
        lines = [
            self._name + ':',
            '    modules: ' + ', '.join(self.modules),
            '    variables:' + ('\n' if variables else '') + variables
        ]
        return '\n'.join(lines)

    def __str__(self):
        return self.name

    def __repr__(self):
        ret = "{0}(name='{1}', modules={2}, variables={3})"
        return ret.format(type(self).__name__, self.name,
                          self.modules, self.variables)


def swap_environments(src, dst):
    src.unload()
    dst.load()


class EnvironmentSnapshot(Environment):
    def __init__(self, name='env_snapshot'):
        self._name = name
        self._modules = runtime().modules_system.loaded_modules()
        self._variables = dict(os.environ)
        self._conflicted = []

    def load(self):
        os.environ.clear()
        os.environ.update(self._variables)
        self._loaded = True

    @property
    def is_loaded(self):
        raise NotImplementedError('is_loaded is not a valid property '
                                  'of an environment snapshot')

    def unload(self):
        raise NotImplementedError('cannot unload an environment snapshot')


class save_environment:
    """A context manager for saving and restoring the current environment."""

    def __init__(self):
        self.environ_save = EnvironmentSnapshot()

    def __enter__(self):
        return self.environ_save

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore the environment and propagate any exception thrown
        self.environ_save.load()


class ProgEnvironment(Environment):
    """A class representing a programming environment.

    This type of environment adds also attributes for setting the compiler and
    compilation flags.

    If compilation flags are set to :class:`None` (the default, if not set
    otherwise in ReFrame's `configuration
    <configure.html#environments-configuration>`__), they are not passed to the
    ``make`` invocation.

    If you want to disable completely the propagation of the compilation flags
    to the ``make`` invocation, even if they are set, you should set the
    :attr:`propagate` attribute to :class:`False`.
    """

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
        """The C compiler of this programming environment.

        :type: :class:`str`
        """
        return self._cc

    @property
    def cxx(self):
        """The C++ compiler of this programming environment.

        :type: :class:`str` or :class:`None`
        """
        return self._cxx

    @property
    def ftn(self):
        """The Fortran compiler of this programming environment.

        :type: :class:`str` or :class:`None`
        """
        return self._ftn

    @property
    def cppflags(self):
        """The preprocessor flags of this programming environment.

        :type: :class:`str` or :class:`None`
        """
        return self._cppflags

    @property
    def cflags(self):
        """The C compiler flags of this programming environment.

        :type: :class:`str` or :class:`None`
        """
        return self._cflags

    @property
    def cxxflags(self):
        """The C++ compiler flags of this programming environment.

        :type: :class:`str` or :class:`None`
        """
        return self._cxxflags

    @property
    def fflags(self):
        """The Fortran compiler flags of this programming environment.

        :type: :class:`str` or :class:`None`
        """
        return self._fflags

    @property
    def ldflags(self):
        """The linker flags of this programming environment.

        :type: :class:`str` or :class:`None`
        """
        return self._ldflags

    @property
    def nvcc(self):
        return self._nvcc
