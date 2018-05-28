import errno
import os

import reframe.core.debug as debug
import reframe.core.fields as fields
import reframe.utility.os_ext as os_ext
from reframe.core.exceptions import (EnvironError, SpawnedProcessError,
                                     CompilationError)
from reframe.core.runtime import runtime


class Environment:
    """This class abstracts away an environment to run regression tests.

    It is simply a collection of modules to be loaded and environment variables
    to be set when this environment is loaded by the framework.
    Users may not create or modify directly environments.
    """
    name = fields.StringPatternField('name', '(\w|-)+')
    modules = fields.TypedListField('modules', str)
    variables = fields.TypedDictField('variables', str, str)

    def __init__(self, name, modules=[], variables={}, **kwargs):
        self._name = name
        self._modules = list(modules)
        self._variables = dict(variables)
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

            self._conflicted += rt.modules_system.load_module(m, force=True)
            for conflict in self._conflicted:
                stmts = rt.modules_system.emit_unload_commands(conflict)
                self._load_stmts += stmts

            self._load_stmts += rt.modules_system.emit_load_commands(m)

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

    def emit_load_instructions(self, builder):
        for stmt in self._load_stmts:
            builder.verbatim(stmt)

        for k, v in self._variables.items():
            builder.set_variable(k, v, export=True)

    # FIXME: Does not correspond to the actual process in unload()
    def emit_unload_instructions(self, builder):
        for k, v in self._variables.items():
            builder.unset_variable(k)

        for m in self._modules:
            builder.verbatim('module unload %s' % m)

        for m in self._conflicted:
            builder.verbatim('module load %s' % m)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        return (self._name == other._name and
                set(self._modules) == set(other._modules) and
                self._variables == other._variables)

    def __repr__(self):
        return debug.repr(self)

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
    cc = fields.StringField('cc')

    #: The C++ compiler of this programming environment.
    #:
    #: :type: :class:`str` or :class:`None`
    cxx = fields.StringField('cxx', allow_none=True)

    #: The Fortran compiler of this programming environment.
    #:
    #: :type: :class:`str` or :class:`None`
    ftn = fields.StringField('ftn', allow_none=True)

    #: The preprocessor flags of this programming environment.
    #:
    #: :type: :class:`str` or :class:`None`
    cppflags = fields.StringField('cppflags', allow_none=True)

    #: The C compiler flags of this programming environment.
    #:
    #: :type: :class:`str` or :class:`None`
    cflags = fields.StringField('cflags', allow_none=True)

    #: The C++ compiler flags of this programming environment.
    #:
    #: :type: :class:`str` or :class:`None`
    cxxflags = fields.StringField('cxxflags', allow_none=True)

    #: The Fortran compiler flags of this programming environment.
    #:
    #: :type: :class:`str` or :class:`None`
    fflags = fields.StringField('fflags', allow_none=True)

    #: The linker flags of this programming environment.
    #:
    #: :type: :class:`str` or :class:`None`
    ldflags = fields.StringField('ldflags', allow_none=True)

    #: The include search path of this programming environment.
    #:
    #: :type: :class:`list` of :class:`str`
    #: :default: ``[]``
    include_search_path = fields.TypedListField('include_search_path', str)

    #: Propagate the compilation flags to the ``make`` invocation.
    #:
    #: :type: :class:`bool`
    #: :default: :class:`True`
    propagate = fields.BooleanField('propagate')

    def __init__(self,
                 name,
                 modules=[],
                 variables={},
                 cc='cc',
                 cxx='CC',
                 ftn='ftn',
                 cppflags=None,
                 cflags=None,
                 cxxflags=None,
                 fflags=None,
                 ldflags=None,
                 **kwargs):
        super().__init__(name, modules, variables)
        self.cc  = cc
        self.cxx = cxx
        self.ftn = ftn
        self.cppflags = cppflags
        self.cflags   = cflags
        self.cxxflags = cxxflags
        self.fflags   = fflags
        self.ldflags  = ldflags
        self.include_search_path = []
        self.propagate = True

    def guess_language(self, filename):
        ext = filename.split('.')[-1]
        if ext in ['c']:
            return 'C'

        if ext in ['cc', 'cp', 'cxx', 'cpp', 'CPP', 'c++', 'C']:
            return 'C++'

        if ext in ['f', 'for', 'ftn', 'F', 'FOR', 'fpp', 'FPP', 'FTN',
                   'f90', 'f95', 'f03', 'f08', 'F90', 'F95', 'F03', 'F08']:
            return 'Fortran'

        if ext in ['cu']:
            return 'CUDA'

    def compile(self, sourcepath, makefile=None, executable=None,
                lang=None, options=''):

        if not os.path.exists(sourcepath):
            raise FileNotFoundError(errno.ENOENT,
                                    os.strerror(errno.ENOENT), sourcepath)

        if os.path.isdir(sourcepath):
            return self._compile_dir(sourcepath, makefile, options)
        else:
            return self._compile_file(sourcepath, executable, lang, options)

    def _compile_file(self, source_file, executable, lang, options):
        if not executable:
            # default executable, same as source_file without the extension
            executable = os.path.join(os.path.dirname(source_file),
                                      source_file.rsplit('.')[:-1][0])

        if not lang:
            lang  = self.guess_language(source_file)

        # Replace None's with empty strings
        cppflags = self.cppflags or ''
        cflags   = self.cflags   or ''
        cxxflags = self.cxxflags or ''
        fflags   = self.fflags   or ''
        ldflags  = self.ldflags  or ''

        flags = [cppflags]
        if lang == 'C':
            compiler = self.cc
            flags.append(cflags)
        elif lang == 'C++':
            compiler = self.cxx
            flags.append(cxxflags)
        elif lang == 'Fortran':
            compiler = self.ftn
            flags.append(fflags)
        elif lang == 'CUDA':
            compiler = 'nvcc'
            flags.append(cxxflags)
        else:
            raise EnvironError('Unknown language: %s' % lang)

        # Append include search path
        flags += ['-I' + d for d in self.include_search_path]
        cmd = ('%s %s %s -o %s %s %s' % (compiler, ' '.join(flags),
                                         source_file, executable,
                                         ldflags, options))
        try:
            return os_ext.run_command(cmd, check=True)
        except SpawnedProcessError as e:
            # Re-raise as compilation error
            raise CompilationError(command=e.command,
                                   stdout=e.stdout,
                                   stderr=e.stderr,
                                   exitcode=e.exitcode) from None

    def _compile_dir(self, source_dir, makefile, options):
        if makefile:
            cmd = 'make -C %s -f %s %s ' % (source_dir, makefile, options)
        else:
            cmd = 'make -C %s %s ' % (source_dir, options)

        # Pass a set of predefined options to the Makefile
        if self.propagate:
            flags = ["CC='%s'"  % self.cc,
                     "CXX='%s'" % self.cxx,
                     "FC='%s'"  % self.ftn]

            # Explicitly check against None here; the user may explicitly want
            # to clear the flags
            if self.cppflags is not None:
                flags.append("CPPFLAGS='%s'" % self.cppflags)

            if self.cflags is not None:
                flags.append("CFLAGS='%s'" % self.cflags)

            if self.cxxflags is not None:
                flags.append("CXXFLAGS='%s'" % self.cxxflags)

            if self.fflags is not None:
                flags.append("FFLAGS='%s'" % self.fflags)

            if self.ldflags is not None:
                flags.append("LDFLAGS='%s'" % self.ldflags)

            cmd += ' '.join(flags)

        try:
            return os_ext.run_command(cmd, check=True)
        except SpawnedProcessError as e:
            # Re-raise as compilation error
            raise CompilationError(command=e.command,
                                   stdout=e.stdout,
                                   stderr=e.stderr,
                                   exitcode=e.exitcode) from None
