import os
import shutil
import subprocess
import reframe.utility.os as os_ext
import reframe.core.debug as debug

from reframe.core.exceptions import (ReframeError,
                                     CommandError,
                                     CompilationError)
from reframe.core.fields import *
from reframe.core.modules import *


class Environment:
    name      = NonWhitespaceField('name')
    modules   = TypedListField('modules', str)
    variables = TypedDictField('variables', str, str)

    def __init__(self, name, modules=[], variables={}, **kwargs):
        self.name = name
        self.modules = list(modules)
        self.variables = dict(variables)
        self.loaded = False

        self._saved_variables = {}
        self._conflicted = []
        self._preloaded = set()
        self._load_stmts = []

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

        # conflicted module list must be filled at the time of load
        for m in self.modules:
            if module_present(m):
                self._preloaded.add(m)

            self._conflicted += module_force_load(m)
            for conflict in self._conflicted:
                self._load_stmts += ['module unload %s' % conflict]

            self._load_stmts += ['module load %s' % m]

        for k, v in self.variables.items():
            if k in os.environ:
                self._saved_variables[k] = os.environ[k]

            os.environ[k] = os.path.expandvars(v)

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
            if m not in self._preloaded:
                module_unload(m)

        # Reload the conflicted packages, previously removed
        for m in self._conflicted:
            module_load(m)

        self.loaded = False

    def emit_load_instructions(self, builder):
        """Emit shell instructions for loading this environment."""
        for stmt in self._load_stmts:
            builder.verbatim(stmt)

        for k, v in self.variables.items():
            builder.set_variable(k, v, export=True)

    # FIXME: Does not correspond to the actual process in unload()
    def emit_unload_instructions(self, builder):
        """Emit shell instructions for loading this environment."""
        for k, v in self.variables.items():
            builder.unset_variable(k)

        for m in self.modules:
            builder.verbatim('module unload %s' % m)

        for m in self._conflicted:
            builder.verbatim('module load %s' % m)

    def __eq__(self, other):
        return (other is not None and
                self.name == other.name and
                set(self.modules) == set(other.modules) and
                self.variables == other.variables)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return debug.repr(self)

    def __str__(self):
        return ('Name: %s\nModules: %s\nEnvironment: %s' %
                (self.name, modules, self.variables))


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
            raise ReframeError('Unknown language')

        # Append include search path
        flags += ['-I' + d for d in self.include_search_path]
        cmd = ('%s %s %s -o %s %s %s' % (compiler, ' '.join(flags),
                                         source_file, executable,
                                         ldflags, options))
        try:
            return os_ext.run_command(cmd, check=True)
        except CommandError as e:
            raise CompilationError(command=e.command,
                                   stdout=e.stdout,
                                   stderr=e.stderr,
                                   exitcode=e.exitcode,
                                   environ=self)

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
        except CommandError as e:
            raise CompilationError(command=e.command,
                                   stdout=e.stdout,
                                   stderr=e.stderr,
                                   exitcode=e.exitcode,
                                   environ=self)
