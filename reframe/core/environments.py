import os
import shutil
import subprocess
import reframe.utility.os as os_ext

from reframe.core.exceptions import ReframeError, CommandError, \
                                       CompilationError
from reframe.core.fields import *
from reframe.core.modules import *


class Environment:
    name      = NonWhitespaceField('name')
    modules   = TypedListField('modules', str)
    variables = TypedDictField('variables', str, str)

    def __init__(self, name, modules = [], variables = {}, **kwargs):
        self.name = name
        self.modules = copy.deepcopy(modules)
        self.variables = copy.deepcopy(variables)
        self.loaded = False

        self._saved_variables = {}
        self._conflicted = []
        self._preloaded = set()


    def add_module(self, name):
        """Add module to the list of modules to be loaded."""
        self.modules.append(name)


    def set_variable(self, name, value):
        """Set environment variable to name.

        If variable exists, its value will be
        saved internally and restored when Restore() is called."""
        self.variables[name] = value


    def load(self):
        """Load environment."""
        for k, v in self.variables.items():
            if k in os.environ:
                self._saved_variables[k] = os.environ[k]
            os.environ[k] = v

        # conlicted module list must be filled at the time of load
        for m in self.modules:
            if module_present(m):
                self._preloaded.add(m)

            self._conflicted += module_force_load(m)

        self.loaded = True


    def unload(self):
        """Restore environment to its previous state."""
        if not self.loaded:
            return

        for k, v in self.variables.items():
            if k in self._saved_variables:
                os.environ[k] = self._saved_variables[k]
            elif k in os.environ:
                del os.environ[k]

        # Unload modules in reverse order
        for m in reversed(self.modules):
            if not m in self._preloaded:
                module_unload(m)

        # Reload the conflicted packages, previously removed
        for m in self._conflicted:
            module_load(m)

        self.loaded = False


    # FIXME: Does not correspond to the actual process in load()
    def emit_load_instructions(self, builder):
        """Emit shell instructions for loading this environment."""
        for m in self._conflicted:
            builder.verbatim('module unload %s' % m)

        for m in self.modules:
            builder.verbatim('module load %s' % m)

        for k, v in self.variables.items():
            builder.set_variable(k, v, export=True)


    # FIXME: Does not correspond to the actual process in unload()
    def emit_unload_instructions(self, builder):
        """Emit shell instructions for loading this environment."""
        for m in self.modules:
            builder.verbatim('module unload %s' % m)

        for m in self._conflicted:
            builder.verbatim('module load %s' % m)

        for k, v in self.variables.items():
            builder.unset_variable(k)


    def __eq__(self, other):
        return \
            other != None and \
            self.name == other.name and \
            set(self.modules) == set(other.modules) and \
            self.variables    == other.variables


    def __ne__(self, other):
        return not self.__eq__(other)


    def __repr__(self):
        return self.__str__()


    def __hash__(self):
        return self.name.__hash__()


    def __str__(self):
        return \
            'Name: %s\n' % self.name + \
            'Modules: %s\n' % str(self.modules) + \
            'Environment: %s\n' % str(self.variables)


def swap_environments(src, dst):
    """Switch from src to dst environment."""
    src.unload()
    dst.load()


class EnvironmentSnapshot(Environment):
    def __init__(self, name='env_snapshot'):
        self.name = name
        self.modules = module_list()
        self.variables = dict(os.environ)
        self._conflicted = []


    def add_module(self, name):
        raise RuntimeError('environment snapshot is read-only')


    def set_variable(self, name, value):
        raise RuntimeError('environment snapshot is read-only')


    def load(self):
        os.environ.clear()
        os.environ.update(self.variables)
        self.loaded = True


    def unload(self):
        raise RuntimeError('cannot unload an environment snapshot')


class ProgEnvironment(Environment):
    def __init__(self,
                 name,
                 modules = [],
                 variables = {},
                 cc  = 'cc',
                 cxx = 'CC',
                 ftn = 'ftn',
                 cppflags = '',
                 cflags   = '',
                 cxxflags = '',
                 fflags   = '',
                 ldflags  = '',
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

    def guess_language(self, filename):
        ext = filename.split('.')[-1]
        if ext in [ 'c' ]:
            return 'C'

        if ext in [ 'cc', 'cp', 'cxx', 'cpp', 'CPP', 'c++', 'C' ]:
            return 'C++'

        if ext in [ 'f', 'for', 'ftn', 'F', 'FOR', 'fpp', 'FPP', 'FTN',
                    'f90', 'f95', 'f03', 'f08', 'F90', 'F95', 'F03', 'F08' ]:
            return 'Fortran'

        if ext in [ 'cu' ]:
            return 'CUDA'


    def compile(self, sourcepath, makefile = None, executable = None,
                lang = None, options = ''):

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

        flags = self.cppflags
        if lang == 'C':
            compiler = self.cc
            flags += ' ' + self.cflags
        elif lang == 'C++':
            compiler = self.cxx
            flags += ' ' + self.cxxflags
        elif lang == 'Fortran':
            compiler = self.ftn
            flags += ' ' + self.fflags
        elif lang == 'CUDA':
            compiler = 'nvcc'
            flags += ' ' + self.cxxflags
        else:
            raise ReframeError('Unknown language')

        # append include search path
        for d in self.include_search_path:
            flags += ' -I%s' % d

        cmd = '%s %s %s -o %s %s %s' % \
              (compiler, flags, source_file, executable, self.ldflags, options)
        try:
            return os_ext.run_command(cmd, check=True)
        except CommandError as e:
            raise CompilationError(command  = e.command,
                                   stdout   = e.stdout,
                                   stderr   = e.stderr,
                                   exitcode = e.exitcode,
                                   environ  = self)


    def _compile_dir(self, source_dir, makefile, options):
        if makefile:
            cmd = 'make -C %s -f %s %s' % (source_dir, makefile, options)
        else:
            cmd = 'make -C %s %s' % (source_dir, options)

        # pass a set of predefined options to the Makefile
        # Naming convetion for implicit make variables
        cmd = cmd + \
              " CC='%s'"  % self.cc + \
              " CXX='%s'" % self.cxx + \
              " FC='%s'"  % self.ftn + \
              " CFLAGS='%s'" % self.cflags + \
              " CXXFLAGS='%s'" % self.cxxflags + \
              " FFLAGS='%s'" % self.fflags + \
              " LDFLAGS='%s'" % self.ldflags

        try:
            return os_ext.run_command(cmd, check=True)
        except CommandError as e:
            raise CompilationError(command  = e.command,
                                   stdout   = e.stdout,
                                   stderr   = e.stderr,
                                   exitcode = e.exitcode,
                                   environ  = self)
