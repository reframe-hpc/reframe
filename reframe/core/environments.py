import collections
import errno
import itertools
import os

import reframe.core.fields as fields
import reframe.utility.os_ext as os_ext
import reframe.utility.typecheck as typ
from reframe.core.exceptions import EnvironError, SpawnedProcessError
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

    def __init__(self, name, modules=[], variables={}, **kwargs):
        self._name = name
        self._modules = list(modules)
        self._variables = collections.OrderedDict()
        self._variables.update(variables)
        self._loaded = False
        self._saved_variables = {}
        self._conflicted = []
        self._preloaded = set()
        self._load_stmts = []

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
        return self._modules

    @property
    def variables(self):
        """The environment variables associated with this environment.

        :type: dictionary of :class:`str` keys/values.
        """
        return self._variables

    @property
    def is_loaded(self):
        """:class:`True` if this environment is loaded,
        :class:`False` otherwise.
        """
        return self._loaded

    # Add module to the list of modules to be loaded.
    def add_module(self, name):
        self._modules.append(name)

    # Set environment variable to name.
    #
    # If variable exists, its value will be saved internally and restored
    # during unloading.
    def set_variable(self, name, value):
        self._variables[name] = value

    def load(self):
        # conflicted module list must be filled at the time of load
        rt = runtime()
        for m in self._modules:
            if rt.modules_system.is_module_loaded(m):
                self._preloaded.add(m)

            conflicted = rt.modules_system.load_module(m, force=True)
            for c in conflicted:
                stmts = rt.modules_system.emit_unload_commands(c)
                self._load_stmts += stmts

            self._load_stmts += rt.modules_system.emit_load_commands(m)
            self._conflicted += conflicted

        for k, v in self._variables.items():
            if k in os.environ:
                self._saved_variables[k] = os.environ[k]

            os.environ[k] = os.path.expandvars(v)

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
        if self.is_loaded:
            # FIXME: This is a workaround for issue #423; the environment
            #        interface must be revisited (see issue #456)
            ret = list(self._load_stmts)
        else:
            ret = list(
                itertools.chain(*(rt.modules_system.emit_load_commands(m)
                                  for m in self.modules))
            )

        for k, v in self._variables.items():
            ret.append('export %s=%s' % (k, v))

        return ret

    def emit_unload_commands(self):
        rt = runtime()
        ret = []
        for var in self._variables.keys():
            ret.append('unset %s' % var)

        ret += list(
            itertools.chain(*(rt.modules_system.emit_unload_commands(m)
                              for m in reversed(self._modules)))
        )
        ret += list(
            itertools.chain(*(rt.modules_system.emit_load_commands(m)
                              for m in self._conflicted))
        )
        return ret

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        return (self._name == other._name and
                set(self._modules) == set(other._modules) and
                self._variables == other._variables)

    def __str__(self):
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

    def add_module(self, name):
        raise EnvironError('environment snapshot is read-only')

    def set_variable(self, name, value):
        raise EnvironError('environment snapshot is read-only')

    def load(self):
        os.environ.clear()
        os.environ.update(self._variables)
        self._loaded = True

    def unload(self):
        raise EnvironError('cannot unload an environment snapshot')


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

    #: The C compiler of this programming environment.
    #:
    #: :type: :class:`str`
    cc = fields.DeprecatedField(fields.TypedField('cc', str),
                                'setting this field is deprecated; '
                                'please set it through a build system',
                                fields.DeprecatedField.OP_SET)
    _cc = fields.TypedField('cc', str)

    #: The C++ compiler of this programming environment.
    #:
    #: :type: :class:`str` or :class:`None`
    cxx = fields.DeprecatedField(fields.TypedField('cxx', str, type(None)),
                                 'setting this field is deprecated; '
                                 'please set it through a build system',
                                 fields.DeprecatedField.OP_SET)
    _cxx = fields.TypedField('cxx', str, type(None))

    #: The Fortran compiler of this programming environment.
    #:
    #: :type: :class:`str` or :class:`None`
    ftn = fields.DeprecatedField(fields.TypedField('ftn', str, type(None)),
                                 'setting this field is deprecated; '
                                 'please set it through a build system',
                                 fields.DeprecatedField.OP_SET)
    _ftn = fields.TypedField('ftn', str, type(None))

    #: The preprocessor flags of this programming environment.
    #:
    #: :type: :class:`str` or :class:`None`
    cppflags = fields.DeprecatedField(
        fields.TypedField('cppflags', str, type(None)),
        'setting this field is deprecated; '
        'please set it through a build system',
        fields.DeprecatedField.OP_SET)
    _cppflags = fields.TypedField('cppflags', str, type(None))

    #: The C compiler flags of this programming environment.
    #:
    #: :type: :class:`str` or :class:`None`
    cflags = fields.DeprecatedField(
        fields.TypedField('cflags', str, type(None)),
        'setting this field is deprecated; '
        'please set it through a build system',
        fields.DeprecatedField.OP_SET)
    _cflags = fields.TypedField('cflags', str, type(None))

    #: The C++ compiler flags of this programming environment.
    #:
    #: :type: :class:`str` or :class:`None`
    cxxflags = fields.DeprecatedField(
        fields.TypedField('cxxflags', str, type(None)),
        'setting this field is deprecated; '
        'please set it through a build system',
        fields.DeprecatedField.OP_SET)
    _cxxflags = fields.TypedField('cxxflags', str, type(None))

    #: The Fortran compiler flags of this programming environment.
    #:
    #: :type: :class:`str` or :class:`None`
    fflags = fields.DeprecatedField(
        fields.TypedField('fflags', str, type(None)),
        'setting this field is deprecated; '
        'please set it through a build system',
        fields.DeprecatedField.OP_SET)
    _fflags = fields.TypedField('fflags', str, type(None))

    #: The linker flags of this programming environment.
    #:
    #: :type: :class:`str` or :class:`None`
    ldflags = fields.DeprecatedField(
        fields.TypedField('ldflags', str, type(None)),
        'setting this field is deprecated; '
        'please set it through a build system',
        fields.DeprecatedField.OP_SET)
    _ldflags = fields.TypedField('ldflags', str, type(None))

    #: The include search path of this programming environment.
    #:
    #: :type: :class:`list` of :class:`str`
    #: :default: ``[]``
    include_search_path = fields.DeprecatedField(
        fields.TypedField('include_search_path', typ.List[str]),
        'setting this field is deprecated; '
        'please set it through a build system',
        fields.DeprecatedField.OP_SET)
    _include_search_path = fields.TypedField('include_search_path',
                                             typ.List[str])

    #: Propagate the compilation flags to the ``make`` invocation.
    #:
    #: :type: :class:`bool`
    #: :default: :class:`True`
    propagate = fields.DeprecatedField(fields.TypedField('propagate', bool),
                                       'setting this field is deprecated; '
                                       'please set it through a build system',
                                       fields.DeprecatedField.OP_SET)
    _propagate = fields.TypedField('propagate', bool)

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
        self._include_search_path = []
        self._propagate = True

    @property
    def nvcc(self):
        return self._nvcc
