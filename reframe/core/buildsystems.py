import abc
import os

import reframe.core.fields as fields
from reframe.core.exceptions import BuildSystemError


class BuildSystem:
    cc  = fields.StringField('cc', allow_none=True)
    cxx = fields.StringField('cxx', allow_none=True)
    ftn = fields.StringField('ftn', allow_none=True)
    nvcc = fields.StringField('nvcc', allow_none=True)
    cflags = fields.TypedListField('cflags', str, allow_none=True)
    cxxflags = fields.TypedListField('cxxflags', str, allow_none=True)
    cppflags = fields.TypedListField('cppflags', str, allow_none=True)
    fflags  = fields.TypedListField('fflags', str, allow_none=True)
    ldflags = fields.TypedListField('ldflags', str, allow_none=True)
    # Set compiler and compiler flags from the programming environment
    #
    # :type: :class:`bool`
    # :default: :class:`True`
    flags_from_environ = fields.BooleanField('flags_from_environ')

    def __init__(self):
        self.cc  = None
        self.cxx = None
        self.ftn = None
        self.nvcc = None
        self.cflags = None
        self.cxxflags = None
        self.cppflags = None
        self.fflags  = None
        self.ldflags = None
        self.flags_from_environ = True

    @abc.abstractmethod
    def emit_build_commands(self, environ):
        """Return a list of commands needed for building using this build system.

        The build commands may always assume to be issued from the top-level
        directory of the code that is to be built.

        :arg environ: The programming environment for which to emit the build
        instructions.
        """

    def _resolve_flags(self, flags, environ, allow_none=True):
        _flags = getattr(self, flags)
        if _flags is not None:
            return _flags

        if self.flags_from_environ:
            return getattr(environ, flags)

        return None

    def _fix_flags(self, flags):
        # FIXME: That's a necessary workaround to the fact the environment
        # defines flags as strings, but here as lists of strings. Should be
        # removed as soon as setting directly the flags in an environment will
        # be disabled.
        if isinstance(flags, str):
            return flags.split()
        else:
            return flags

    def _cc(self, environ):
        return self._resolve_flags('cc', environ, False)

    def _cxx(self, environ):
        return self._resolve_flags('cxx', environ, False)

    def _ftn(self, environ):
        return self._resolve_flags('ftn', environ, False)

    def _nvcc(self, environ):
        return self._resolve_flags('nvcc', environ, False)

    def _cppflags(self, environ):
        return self._fix_flags(self._resolve_flags('cppflags', environ))

    def _cflags(self, environ):
        return self._fix_flags(self._resolve_flags('cflags', environ))

    def _cxxflags(self, environ):
        return self._fix_flags(self._resolve_flags('cxxflags', environ))

    def _fflags(self, environ):
        return self._fix_flags(self._resolve_flags('fflags', environ))

    def _ldflags(self, environ):
        return self._fix_flags(self._resolve_flags('ldflags', environ))


class Make(BuildSystem):
    options = fields.TypedListField('options', str)
    makefile = fields.StringField('makefile', allow_none=True)
    srcdir = fields.StringField('srcdir', allow_none=True)
    max_concurrency = fields.IntegerField('max_concurrency', allow_none=True)

    def __init__(self):
        super().__init__()
        self.options = []
        self.makefile = None
        self.srcdir = None
        self.max_concurrency = None

    def emit_build_commands(self, environ):
        cmd_parts = ['make']
        if self.makefile:
            cmd_parts += ['-f %s' % self.makefile]

        if self.srcdir:
            cmd_parts += ['-C %s' % self.srcdir]

        cmd_parts += ['-j']
        if self.max_concurrency is not None:
            cmd_parts += [str(self.max_concurrency)]

        cc = self._cc(environ)
        cxx = self._cxx(environ)
        ftn = self._ftn(environ)
        nvcc = self._nvcc(environ)
        cppflags = self._cppflags(environ)
        cflags   = self._cflags(environ)
        cxxflags = self._cxxflags(environ)
        fflags   = self._fflags(environ)
        ldflags  = self._ldflags(environ)
        if cc is not None:
            cmd_parts += ["CC='%s'" % cc]

        if cxx is not None:
            cmd_parts += ["CXX='%s'" % cxx]

        if ftn is not None:
            cmd_parts += ["FC='%s'" % ftn]

        if nvcc is not None:
            cmd_parts += ["NVCC='%s'" % nvcc]

        if cppflags is not None:
            cmd_parts += ["CPPFLAGS='%s'" % ' '.join(cppflags)]

        if cflags is not None:
            cmd_parts += ["CFLAGS='%s'" % ' '.join(cflags)]

        if cxxflags is not None:
            cmd_parts += ["CXXFLAGS='%s'" % ' '.join(cxxflags)]

        if fflags is not None:
            cmd_parts += ["FFLAGS='%s'" % ' '.join(fflags)]

        if ldflags is not None:
            cmd_parts += ["LDFLAGS='%s'" % ' '.join(ldflags)]

        if self.options:
            cmd_parts += self.options

        # Cause script to exit immediately if compilation fails
        cmd_parts += ['|| exit 1']
        return [' '.join(cmd_parts)]


class SingleSource(BuildSystem):
    srcfile = fields.StringField('srcfile', allow_none=True)
    executable = fields.StringField('executable', allow_none=True)
    include_path = fields.TypedListField('include_path', str)
    lang = fields.StringField('lang', allow_none=True)

    def __init__(self):
        super().__init__()
        self.srcfile = None
        self.executable = None
        self.include_path = []
        self.lang = None

    def _auto_exec_name(self):
        return '%s.exe' % os.path.splitext(self.srcfile)[0]

    def emit_build_commands(self, environ):
        if not self.srcfile:
            raise BuildSystemError(
                'a source file is required when using the %s build system' %
                type(self).__name__)

        cc = self._cc(environ)
        cxx = self._cxx(environ)
        ftn = self._ftn(environ)
        nvcc = self._nvcc(environ)
        cppflags = self._cppflags(environ) or []
        cflags   = self._cflags(environ) or []
        cxxflags = self._cxxflags(environ) or []
        fflags   = self._fflags(environ)  or []
        ldflags  = self._ldflags(environ) or []

        # Adjust cppflags with the include directories
        # NOTE: We do not use the += operator on purpose, be cause we don't
        # want to change the original list passed by the user
        cppflags = cppflags + [*map(lambda d: '-I ' + d, self.include_path)]

        # Generate the executable
        executable = self.executable or self._auto_exec_name()

        # Prepare the compilation command
        lang = self.lang or self._guess_language(self.srcfile)
        cmd_parts = []
        if lang == 'C':
            if cc is None:
                raise BuildSystemError('I do not know how to compile a '
                                       'C program')

            cmd_parts += [cc, *cppflags, *cflags, self.srcfile,
                          '-o', executable, *ldflags]
        elif lang == 'C++':
            if cxx is None:
                raise BuildSystemError('I do not know how to compile a '
                                       'C++ program')

            cmd_parts += [cxx, *cppflags, *cxxflags, self.srcfile,
                          '-o', executable, *ldflags]
        elif lang == 'Fortran':
            if ftn is None:
                raise BuildSystemError('I do not know how to compile a '
                                       'Fortran program')

            cmd_parts += [ftn, *cppflags, *fflags, self.srcfile,
                          '-o', executable, *ldflags]
        elif lang == 'CUDA':
            if nvcc is None:
                raise BuildSystemError('I do not know how to compile a '
                                       'CUDA program')

            cmd_parts += [nvcc, *cppflags, *cxxflags, self.srcfile,
                          '-o', executable, *ldflags]
        else:
            BuildSystemError('could not guess language of file: %s' %
                             self.srcfile)

        # Cause script to exit immediately if compilation fails
        cmd_parts += ['|| exit 1']
        return [' '.join(cmd_parts)]

    def _guess_language(self, filename):
        _, ext = os.path.splitext(filename)
        if ext in ['.c']:
            return 'C'

        if ext in ['.cc', '.cp', '.cxx', '.cpp', '.CPP', '.c++', '.C']:
            return 'C++'

        if ext in ['.f', '.for', '.ftn', '.F', '.FOR', '.fpp',
                   '.FPP', '.FTN', '.f90', '.f95', '.f03', '.f08',
                   '.F90', '.F95', '.F03', '.F08']:
            return 'Fortran'

        if ext in ['.cu']:
            return 'CUDA'


class BuildSystemField(fields.TypedField):
    """A field representing a build system.

    You may either assign an instance of :class:`BuildSystem` or a string
    representing the name of the concrete class of a build system.
    """

    def __init__(self, fieldname, allow_none=False):
        super().__init__(fieldname, BuildSystem, allow_none)

    def __set__(self, obj, value):
        if isinstance(value, str):
            try:
                value = globals()[value]()
            except KeyError:
                raise ValueError('unknown build system: %s' % value) from None

        super().__set__(obj, value)
