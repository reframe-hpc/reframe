# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import abc
import os
import re

import reframe.core.fields as fields
import reframe.utility.typecheck as typ
from reframe.core.exceptions import BuildSystemError
from reframe.core.meta import RegressionTestMeta


class BuildSystemMeta(RegressionTestMeta, abc.ABCMeta):
    '''Build systems metaclass.'''


class BuildSystem(metaclass=BuildSystemMeta):
    '''The abstract base class of any build system.

    Concrete build systems inherit from this class and must override the
    :func:`emit_build_commands` abstract function.
    '''

    #: The C compiler to be used.
    #: If empty and :attr:`flags_from_environ` is :class:`True`,
    #: the compiler defined in the current programming environment will be
    #: used.
    #:
    #: :type: :class:`str`
    #: :default: ``''``
    cc  = variable(str, value='')

    #: The C++ compiler to be used.
    #: If empty and :attr:`flags_from_environ` is :class:`True`,
    #: the compiler defined in the current programming environment will be
    #: used.
    #:
    #: :type: :class:`str`
    #: :default: ``''``
    cxx = variable(str, value='')

    #: The Fortran compiler to be used.
    #: If empty and :attr:`flags_from_environ` is :class:`True`,
    #: the compiler defined in the current programming environment will be
    #: used.
    #:
    #: :type: :class:`str`
    #: :default: ``''``
    ftn = variable(str, value='')

    #: The CUDA compiler to be used.
    #: If empty and :attr:`flags_from_environ` is :class:`True`,
    #: the compiler defined in the current programming environment will be
    #: used.
    #:
    #: :type: :class:`str`
    #: :default: ``''``
    nvcc = variable(str, value='')

    #: The C compiler flags to be used.
    #: If empty and :attr:`flags_from_environ` is :class:`True`,
    #: the corresponding flags defined in the current programming environment
    #: will be used.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    cflags = variable(typ.List[str], value=[])

    #: The preprocessor flags to be used.
    #: If empty and :attr:`flags_from_environ` is :class:`True`,
    #: the corresponding flags defined in the current programming environment
    #: will be used.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    cppflags = variable(typ.List[str], value=[])

    #: The C++ compiler flags to be used.
    #: If empty and :attr:`flags_from_environ` is :class:`True`,
    #: the corresponding flags defined in the current programming environment
    #: will be used.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    cxxflags = variable(typ.List[str], value=[])

    #: The Fortran compiler flags to be used.
    #: If empty and :attr:`flags_from_environ` is :class:`True`,
    #: the corresponding flags defined in the current programming environment
    #: will be used.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    fflags  = variable(typ.List[str], value=[])

    #: The linker flags to be used.
    #: If empty and :attr:`flags_from_environ` is :class:`True`,
    #: the corresponding flags defined in the current programming environment
    #: will be used.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    ldflags = variable(typ.List[str], value=[])

    #: Set compiler and compiler flags from the current programming environment
    #: if not specified otherwise.
    #:
    #: :type: :class:`bool`
    #: :default: :class:`True`
    flags_from_environ = variable(bool, value=True)

    @abc.abstractmethod
    def emit_build_commands(self, environ):
        '''Return the list of commands for building using this build system.

        The build commands, as well as this function, will always be executed
        from the test's stage directory.

        :arg environ: The programming environment for which to emit the build
           instructions.
           The framework passes here the current programming environment.
        :type environ: :class:`reframe.core.environments.ProgEnvironment`
        :raises: :class:`BuildSystemError` in case of errors when generating
          the build instructions.

        .. versionchanged:: 3.5.0
           This function executes from the test stage directory.

        :meta private:

        '''

    def post_build(self, buildjob):
        '''Callback function that the framework will call when the compilation
        is done.

        Build systems may use this information to do some post processing and
        provide additional, build system-specific, functionality to the users.

        This function will always be executed from the test's stage directory.

        .. versionadded:: 3.5.0
        .. versionchanged:: 3.5.2
           The function is executed from the stage directory.

        :meta private:

        '''

    def prepare_cmds(self):
        '''Callback function that the framework will call before run.

        Build systems may use this information to add commands to the run
        script before anything set by the user.

        :meta private:
        '''
        return []

    def _resolve_flags(self, flags, environ):
        _flags = getattr(self, flags)
        if _flags:
            return _flags

        if self.flags_from_environ:
            return getattr(environ, flags)

        return None

    def _cc(self, environ):
        return self._resolve_flags('cc', environ)

    def _cxx(self, environ):
        return self._resolve_flags('cxx', environ)

    def _ftn(self, environ):
        return self._resolve_flags('ftn', environ)

    def _nvcc(self, environ):
        return self._resolve_flags('nvcc', environ)

    def _cppflags(self, environ):
        return self._resolve_flags('cppflags', environ)

    def _cflags(self, environ):
        return self._resolve_flags('cflags', environ)

    def _cxxflags(self, environ):
        return self._resolve_flags('cxxflags', environ)

    def _fflags(self, environ):
        return self._resolve_flags('fflags', environ)

    def _ldflags(self, environ):
        return self._resolve_flags('ldflags', environ)

    def __str__(self):
        return type(self).__name__

    def __rfm_json_encode__(self):
        return str(self)


class Make(BuildSystem):
    '''A build system for compiling codes using ``make``.

    The generated build command has the following form:

    .. code::

      make -j [N] [-f MAKEFILE] [-C SRCDIR] CC="X" CXX="X" FC="X" NVCC="X" CPPFLAGS="X" CFLAGS="X" CXXFLAGS="X" FCFLAGS="X" LDFLAGS="X" OPTIONS

    The compiler and compiler flags variables will only be passed if they are
    not :class:`None`.
    Their value is determined by the corresponding attributes of
    :class:`BuildSystem`.
    If you want to completely disable passing these variables to the ``make``
    invocation, you should make sure not to set any of the correspoding
    attributes and set also the :attr:`BuildSystem.flags_from_environ` flag to
    :class:`False`.
    '''

    #: Append these options to the ``make`` invocation.
    #: This variable is also useful for passing variables or targets to
    #: ``make``.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    options = variable(typ.List[str], value=[])

    #: Instruct build system to use this Makefile.
    #: This option is useful when having non-standard Makefile names.
    #:
    #: :type: :class:`str`
    #: :default: :class:`None`
    makefile = variable(str, type(None), value=None)

    #: The top-level directory of the code.
    #:
    #: This is set automatically by the framework based on the
    #: :attr:`reframe.core.pipeline.RegressionTest.sourcepath` attribute.
    #:
    #: :type: :class:`str`
    #: :default: :class:`None`
    srcdir = variable(str, type(None), value=None)

    #: Limit concurrency for ``make`` jobs.
    #: This attribute controls the ``-j`` option passed to ``make``.
    #: If not :class:`None`, ``make`` will be invoked as ``make -j
    #: max_concurrency``.
    #: Otherwise, it will invoked as ``make -j``.
    #:
    #: :type: integer
    #: :default: ``1``
    #:
    #: .. note::
    #:     .. versionchanged:: 2.19
    #:        The default value is now ``1``
    max_concurrency = variable(int, type(None), value=1)

    def emit_build_commands(self, environ):
        cmd_parts = ['make']
        if self.makefile:
            cmd_parts += [f'-f {self.makefile}']

        if self.srcdir:
            cmd_parts += [f'-C {self.srcdir}']

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
        if cc:
            cmd_parts += [f'CC="{cc}"']

        if cxx:
            cmd_parts += [f'CXX="{cxx}"']

        if ftn:
            cmd_parts += [f'FC="{ftn}"']

        if nvcc:
            cmd_parts += [f'NVCC="{nvcc}"']

        if cppflags:
            flags = ' '.join(cppflags)
            cmd_parts += [f'CPPFLAGS="{flags}"']

        if cflags:
            flags = ' '.join(cflags)
            cmd_parts += [f'CFLAGS="{flags}"']

        if cxxflags:
            flags = ' '.join(cxxflags)
            cmd_parts += [f'CXXFLAGS="{flags}"']

        if fflags:
            flags = ' '.join(fflags)
            cmd_parts += [f'FCFLAGS="{flags}"']

        if ldflags:
            flags = ' '.join(ldflags)
            cmd_parts += [f'LDFLAGS="{flags}"']

        if self.options:
            cmd_parts += self.options

        return [' '.join(cmd_parts)]


class SingleSource(BuildSystem):
    '''A build system for compiling a single source file.

    The generated build command will have the following form:

    .. code::

      COMP CPPFLAGS XFLAGS SRCFILE -o EXEC LDFLAGS

    - ``COMP`` is the required compiler for compiling ``SRCFILE``.
      This build system will automatically detect the programming language of
      the source file and pick the correct compiler.
      See also the :attr:`SingleSource.lang` attribute.
    - ``CPPFLAGS`` are the preprocessor flags and are passed to any compiler.
    - ``XFLAGS`` is any of ``CFLAGS``, ``CXXFLAGS`` or ``FCFLAGS`` depending on
      the programming language of the source file.
    - ``SRCFILE`` is the source file to be compiled.
      This is set up automatically by the framework.
      See also the :attr:`SingleSource.srcfile` attribute.
    - ``EXEC`` is the executable to be generated.
      This is also set automatically by the framework.
      See also the :attr:`SingleSource.executable` attribute.
    - ``LDFLAGS`` are the linker flags.

    For CUDA codes, the language assumed is C++ (for the compilation flags) and
    the compiler used is :attr:`BuildSystem.nvcc`.

    '''

    #: The source file to compile.
    #: This is automatically set by the framework based on the
    #: :attr:`reframe.core.pipeline.RegressionTest.sourcepath` attribute.
    #:
    #: :type: :class:`str` or :class:`None`
    srcfile = variable(str, type(None), value=None)

    #: The executable file to be generated.
    #:
    #: This is set automatically by the framework based on the
    #: :attr:`reframe.core.pipeline.RegressionTest.executable` attribute.
    #:
    #: :type: :class:`str` or :class:`None`
    executable = variable(str, type(None), value=None)

    #: The include path to be used for this compilation.
    #:
    #: All the elements of this list will be appended to the
    #: :attr:`BuildSystem.cppflags`, by prepending to each of them the ``-I``
    #: option.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    include_path = variable(typ.List[str], value=[])

    #: The programming language of the file that needs to be compiled.
    #: If not specified, the build system will try to figure it out
    #: automatically based on the extension of the source file.
    #: The automatically detected extensions are the following:
    #:
    #:   - C: `.c` and `.upc`.
    #:   - C++: `.cc`, `.cp`, `.cxx`, `.cpp`, `.CPP`, `.c++` and `.C`.
    #:   - Fortran: `.f`, `.for`, `.ftn`, `.F`, `.FOR`, `.fpp`, `.FPP`, `.FTN`,
    #:     `.f90`, `.f95`, `.f03`, `.f08`, `.F90`, `.F95`, `.F03` and `.F08`.
    #:   - CUDA: `.cu`.
    #:
    #: :type: :class:`str` or :class:`None`
    lang = variable(str, type(None), value=None)

    def _auto_exec_name(self):
        return os.path.splitext(self.srcfile)[0] + '.x'

    def emit_build_commands(self, environ):
        if not self.srcfile:
            bname = type(self).__name__
            raise BuildSystemError(f'a source file is required when using '
                                   f'the {bname!r} build system')

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
            if not cc:
                raise BuildSystemError('I do not know how to compile a '
                                       'C program')

            cmd_parts += [cc, *cppflags, *cflags, self.srcfile,
                          '-o', executable, *ldflags]
        elif lang == 'C++':
            if not cxx:
                raise BuildSystemError('I do not know how to compile a '
                                       'C++ program')

            cmd_parts += [cxx, *cppflags, *cxxflags, self.srcfile,
                          '-o', executable, *ldflags]
        elif lang == 'Fortran':
            if not ftn:
                raise BuildSystemError('I do not know how to compile a '
                                       'Fortran program')

            cmd_parts += [ftn, *cppflags, *fflags, self.srcfile,
                          '-o', executable, *ldflags]
        elif lang == 'CUDA':
            if not nvcc:
                raise BuildSystemError('I do not know how to compile a '
                                       'CUDA program')

            cmd_parts += [nvcc, *cppflags, *cxxflags, self.srcfile,
                          '-o', executable, *ldflags]
        else:
            raise BuildSystemError(
                f'could not guess language of file: {self.srcfile}'
            )

        return [' '.join(cmd_parts)]

    def _guess_language(self, filename):
        _, ext = os.path.splitext(filename)
        if ext in ['.c', '.upc']:
            return 'C'

        if ext in ['.cc', '.cp', '.cxx', '.cpp', '.CPP', '.c++', '.C']:
            return 'C++'

        if ext in ['.f', '.for', '.ftn', '.F', '.FOR', '.fpp',
                   '.FPP', '.FTN', '.f90', '.f95', '.f03', '.f08',
                   '.F90', '.F95', '.F03', '.F08', '.cuf', '.CUF']:
            return 'Fortran'

        if ext in ['.cu']:
            return 'CUDA'


class ConfigureBasedBuildSystem(BuildSystem):
    '''Abstract base class for configured-based build systems.'''

    #: The top-level directory of the code.
    #:
    #: This is set automatically by the framework based on the
    #: :attr:`reframe.core.pipeline.RegressionTest.sourcepath` attribute.
    #:
    #: :type: :class:`str`
    #: :default: :class:`None`
    srcdir = variable(str, type(None), value=None)

    #: The CMake build directory, where all the generated files will be placed.
    #:
    #: :type: :class:`str`
    #: :default: :class:`None`
    builddir = variable(str, type(None), value=None)

    #: Additional configuration options to be passed to the CMake invocation.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    config_opts = variable(typ.List[str], value=[])

    #: Options to be passed to the subsequent ``make`` invocation.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    make_opts = variable(typ.List[str], value=[])

    #: Same as for the :attr:`Make` build system.
    #:
    #: :type: integer
    #: :default: ``1``
    max_concurrency = variable(int, type(None), value=1)


class CMake(ConfigureBasedBuildSystem):
    '''A build system for compiling CMake-based projects.

    This build system will emit the following commands:

    1. Create a build directory if :attr:`builddir` is not :class:`None` and
       change to it.
    2. Invoke ``cmake`` to configure the project by setting the corresponding
       CMake flags for compilers and compiler flags.
    3. Issue ``make`` to compile the code.
    '''

    def _combine_flags(self, cppflags, xflags):
        if not cppflags:
            return xflags

        ret = list(cppflags)
        if xflags:
            ret += xflags

        return ret

    def emit_build_commands(self, environ):
        prepare_cmd = []
        if self.srcdir:
            prepare_cmd += [f'cd {self.srcdir}']

        if self.builddir:
            prepare_cmd += [f'mkdir -p {self.builddir}',
                            f'cd {self.builddir}']

        cmake_cmd = ['cmake']
        cc = self._cc(environ)
        cxx = self._cxx(environ)
        ftn = self._ftn(environ)
        nvcc = self._nvcc(environ)
        cppflags = self._cppflags(environ)
        cflags   = self._combine_flags(cppflags, self._cflags(environ))
        cxxflags = self._combine_flags(cppflags, self._cxxflags(environ))
        fflags   = self._combine_flags(cppflags, self._fflags(environ))
        ldflags  = self._ldflags(environ)
        if cc:
            cmake_cmd += [f'-DCMAKE_C_COMPILER="{cc}"']

        if cxx:
            cmake_cmd += [f'-DCMAKE_CXX_COMPILER="{cxx}"']

        if ftn:
            cmake_cmd += [f'-DCMAKE_Fortran_COMPILER="{ftn}"']

        if nvcc:
            cmake_cmd += [f'-DCMAKE_CUDA_COMPILER="{nvcc}"']

        if cflags:
            flags = ' '.join(cflags)
            cmake_cmd += [f'-DCMAKE_C_FLAGS="{flags}"']

        if cxxflags:
            flags = ' '.join(cxxflags)
            cmake_cmd += [f'-DCMAKE_CXX_FLAGS="{flags}"']

        if fflags:
            flags = ' '.join(fflags)
            cmake_cmd += [f'-DCMAKE_Fortran_FLAGS="{flags}"']

        if ldflags:
            flags = ' '.join(ldflags)
            cmake_cmd += [f'-DCMAKE_EXE_LINKER_FLAGS="{flags}"']

        if self.config_opts:
            cmake_cmd += self.config_opts

        if self.builddir:
            cmake_cmd += [os.path.relpath('.', self.builddir)]
        else:
            cmake_cmd += ['.']

        make_cmd = ['make -j']
        if self.max_concurrency is not None:
            make_cmd += [str(self.max_concurrency)]

        if self.make_opts:
            make_cmd += self.make_opts

        return prepare_cmd + [' '.join(cmake_cmd), ' '.join(make_cmd)]


class Autotools(ConfigureBasedBuildSystem):
    '''A build system for compiling Autotools-based projects.

    This build system will emit the following commands:

    1. Create a build directory if :attr:`builddir` is not :class:`None` and
       change to it.
    2. Invoke ``configure`` to configure the project by setting the
       corresponding flags for compilers and compiler flags.
    3. Issue ``make`` to compile the code.
    '''

    #: The directory of the configure script.
    #:
    #: This can be changed to do an out of source build without copying the
    #: entire source tree.
    #:
    #: :type: :class:`str`
    #: :default: ``'.'``
    configuredir = variable(str, value='.')

    def emit_build_commands(self, environ):
        prepare_cmd = []
        if self.srcdir:
            prepare_cmd += [f'cd {self.srcdir}']

        if self.builddir:
            prepare_cmd += [f'mkdir -p {self.builddir}',
                            f'cd {self.builddir}']

        if self.builddir:
            configure_cmd = [os.path.join(
                os.path.relpath(self.configuredir, self.builddir), 'configure'
            )]
        else:
            configure_cmd = [os.path.join(self.configuredir, 'configure')]

        cc = self._cc(environ)
        cxx = self._cxx(environ)
        ftn = self._ftn(environ)
        cppflags = self._cppflags(environ)
        cflags   = self._cflags(environ)
        cxxflags = self._cxxflags(environ)
        fflags   = self._fflags(environ)
        ldflags  = self._ldflags(environ)
        if cc:
            configure_cmd += [f'CC="{cc}"']

        if cxx:
            configure_cmd += [f'CXX="{cxx}"']

        if ftn:
            configure_cmd += [f'FC="{ftn}"']

        if cppflags:
            flags = ' '.join(cppflags)
            configure_cmd += [f'CPPFLAGS="{flags}"']

        if cflags:
            flags = ' '.join(cflags)
            configure_cmd += [f'CFLAGS="{flags}"']

        if cxxflags:
            flags = ' '.join(cxxflags)
            configure_cmd += [f'CXXFLAGS="{flags}"']

        if fflags:
            flags = ' '.join(fflags)
            configure_cmd += [f'FCFLAGS="{flags}"']

        if ldflags:
            flags = ' '.join(ldflags)
            configure_cmd += [f'LDFLAGS="{flags}"']

        if self.config_opts:
            configure_cmd += self.config_opts

        make_cmd = ['make -j']
        if self.max_concurrency is not None:
            make_cmd += [str(self.max_concurrency)]

        if self.make_opts:
            make_cmd += self.make_opts

        return prepare_cmd + [' '.join(configure_cmd), ' '.join(make_cmd)]


class EasyBuild(BuildSystem):
    '''A build system for building test code using `EasyBuild
    <https://easybuild.io/>`__.

    ReFrame will use EasyBuild to build and install the code in the test's
    stage directory by default. ReFrame uses environment variables to
    configure EasyBuild for running, so users can pass additional options to
    the ``eb`` command and modify the default behaviour.

    .. versionadded:: 3.5.0

    '''

    #: The list of easyconfig files to build and install.
    #: This field is required.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    easyconfigs = variable(typ.List[str], value=[])

    #: Options to pass to the ``eb`` command.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    options = variable(typ.List[str], value=[])

    #: Instruct EasyBuild to emit a package for the built software.
    #: This will essentially pass the ``--package`` option to ``eb``.
    #:
    #: :type: :class:`bool`
    #: :default: ``False``
    emit_package = variable(bool, value=False)

    #: Options controlling the package creation from EasyBuild.
    #: For each key/value pair of this dictionary, ReFrame will pass
    #: ``--package-{key}={val}`` to the EasyBuild invocation.
    #:
    #: :type: :class:`Dict[str, str]`
    #: :default: ``{}``
    package_opts = variable(typ.Dict[str, str], value={})

    #: Default prefix for the EasyBuild installation.
    #:
    #: Relative paths will be appended to the stage directory of the test.
    #: ReFrame will set the following environment variables before running
    #: EasyBuild.
    #:
    #: .. code-block:: bash
    #:
    #:    export EASYBUILD_BUILDPATH={prefix}/build
    #:    export EASYBUILD_INSTALLPATH={prefix}
    #:    export EASYBUILD_PREFIX={prefix}
    #:    export EASYBUILD_SOURCEPATH={prefix}
    #:
    #: Users can change these defaults by passing specific options to the
    #: ``eb`` command.
    #:
    #: :type: :class:`str`
    #: :default: ``easybuild``
    prefix = variable(str, value='easybuild')

    def __init__(self):
        self._eb_modules = []

    def emit_build_commands(self, environ):
        if not self.easyconfigs:
            raise BuildSystemError(f"'easyconfigs' must not be empty")

        easyconfigs = ' '.join(self.easyconfigs)
        if self.emit_package:
            self.options.append('--package')
            for key, val in self.package_opts.items():
                self.options.append(f'--package-{key}={val}')

        prefix = os.path.join(os.getcwd(), self.prefix)
        options = ' '.join(self.options)
        return [f'export EASYBUILD_BUILDPATH={prefix}/build',
                f'export EASYBUILD_INSTALLPATH={prefix}',
                f'export EASYBUILD_PREFIX={prefix}',
                f'export EASYBUILD_SOURCEPATH={prefix}',
                f'eb {easyconfigs} {options}']

    def post_build(self, buildjob):
        # Store the modules generated by EasyBuild

        modulesdir = os.path.join(os.getcwd(), self.prefix,
                                  'modules', 'all')
        with open(buildjob.stdout) as fp:
            out = fp.read()

        self._eb_modules = [
            {'name': m, 'collection': False, 'path': modulesdir}
            for m in re.findall(r'building and installing (\S+)...', out)
        ]

    @property
    def generated_modules(self):
        return self._eb_modules


class Spack(BuildSystem):
    '''A build system for building test code using `Spack
    <https://spack.io/>`__.

    ReFrame will use a user-provided Spack environment in order to build and
    test a set of specs.

    .. versionadded:: 3.6.1

    '''

    #: The Spack environment to use for building this test.
    #:
    #: ReFrame will activate and install this environment.
    #: This environment will also be used to run the test.
    #:
    #: .. code-block:: bash
    #:
    #:    spack env activate -V -d <environment directory>
    #:
    #: ReFrame looks for environments in the test's
    #: :attr:`~reframe.core.pipeline.RegressionTest.sourcesdir`.
    #:
    #: If this field is `None`, the default, the environment name will
    #: be automatically set to `rfm_spack_env`.
    #:
    #: :type: :class:`str` or :class:`None`
    #: :default: :class:`None`
    #:
    #: .. note::
    #:     .. versionchanged:: 3.7.3
    #:        The field is no longer required and the Spack environment will be
    #:        automatically created if not provided.
    environment = variable(typ.Str[r'\S+'], type(None), value=None)

    #: The directory where Spack will install the packages requested by this
    #: test.
    #:
    #: After activating the Spack environment, ReFrame will set the
    #: `install_tree` Spack configuration in the given environment with the
    #: following command:
    #:
    #: .. code-block:: bash
    #:
    #:    spack config add "config:install_tree:root:<install tree>"
    #:
    #: Relative paths are resolved against the test's stage directory.  If this
    #: field and the Spack environment are both `None`, the default, the
    #: install directory will be automatically set to `opt/spack`.  If this
    #: field `None` but the Spack environment is not, then `install_tree` will
    #: not be set automatically and the install tree of the given environment
    #: will not be overridden.
    #:
    #: :type: :class:`str` or :class:`None`
    #: :default: :class:`None`
    #:
    #: .. versionadded:: 3.7.3
    install_tree = variable(typ.Str[r'\S+'], type(None), value=None)

    #: A list of additional specs to build and install within the given
    #: environment.
    #:
    #: ReFrame will add the specs to the active environment by emititing the
    #: following command:
    #:
    #: .. code-block:: bash
    #:
    #:    spack add spec1 spec2 ... specN
    #:
    #: If no spec is passed, ReFrame will simply install what is prescribed by
    #: the environment.
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    specs = variable(typ.List[str], value=[])

    #: Emit the necessary ``spack load`` commands before running the test.
    #:
    #: :type: :class:`bool`
    #: :default: :obj:`True`
    emit_load_cmds = variable(bool, value=True)

    #: Options to pass to ``spack install``
    #:
    #: :type: :class:`List[str]`
    #: :default: ``[]``
    install_opts = variable(typ.List[str], value=[])

    def __init__(self):
        # Set to True if the environment was auto-generated
        self._auto_env = False

    def emit_build_commands(self, environ):
        ret = self._create_env_cmds()

        if self._auto_env:
            install_tree = self.install_tree or 'opt/spack'
            ret.append(f'spack -e {self.environment} config add '
                       f'"config:install_tree:root:{install_tree}"')

        if self.specs:
            specs_str = ' '.join(self.specs)
            ret.append(f'spack -e {self.environment} add {specs_str}')

        install_cmd = f'spack -e {self.environment} install'
        if self.install_opts:
            install_cmd += ' ' + ' '.join(self.install_opts)

        ret.append(install_cmd)
        return ret

    def _create_env_cmds(self):
        if self.environment:
            return []

        self.environment = 'rfm_spack_env'
        self._auto_env = True
        return [f'spack env create -d {self.environment}']

    def prepare_cmds(self):
        cmds = self._create_env_cmds()
        if self.specs and self.emit_load_cmds:
            cmds.append(
                f'eval `spack -e {self.environment} load '
                f'--sh {" ".join(self.specs)}`'
            )

        return cmds


class CustomBuild(BuildSystem):
    '''Custom build system.

    This build system backend allows users to use custom build scripts to
    build the test code. It does not do any interpretation of the current test
    environment and it simply runs the supplied :attr:`commands`.

    Users should use this build system with caution, because environment
    management, reproducibility and any potential side effects are all
    controlled by the user's custom build system.

    .. versionadded:: 3.11.0
    '''

    #: The commands to run for building the test code.
    #:
    #: :type: :class:`List[str]`
    commands = variable(typ.List[str])

    def emit_build_commands(self, environ):
        return self.commands


class BuildSystemField(fields.TypedField):
    def __init__(self, fieldname, *other_types):
        super().__init__(fieldname, BuildSystem, *other_types)

    def __set__(self, obj, value):
        if isinstance(value, str):
            try:
                value = globals()[value]()
            except KeyError:
                raise ValueError(f'unknown build system: {value}') from None

        super().__set__(obj, value)
