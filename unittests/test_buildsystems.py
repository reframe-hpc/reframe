# Copyright 2016-2023 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import reframe.core.buildsystems as bs
import reframe.utility.osext as osext
import unittests.utility as test_util
from reframe.core.environments import ProgEnvironment
from reframe.core.exceptions import BuildSystemError


@pytest.fixture
def environ():
    return ProgEnvironment(name='test_env',
                           cc='gcc',
                           cxx='g++',
                           ftn='gfortran',
                           cppflags=['-DNDEBUG'],
                           cflags=['-Wall', '-std=c99'],
                           cxxflags=['-Wall', '-std=c++11'],
                           fflags=['-Wall'],
                           ldflags=['-dynamic'])


@pytest.fixture(params=['Autotools', 'CMake',
                        'CustomBuild', 'Make', 'SingleSource'])
def build_system(request):
    return bs.__dict__[request.param]()


@pytest.fixture
def build_system_with_flags(build_system):
    build_system.cc = 'cc'
    build_system.cxx = 'CC'
    build_system.ftn = 'ftn'
    build_system.nvcc = 'clang'
    build_system.cppflags = ['-DFOO']
    build_system.cflags = ['-Wall', '-std=c99', '-O2']
    build_system.cxxflags = ['-Wall', '-std=c++11', '-O2']
    build_system.fflags = ['-Wall', '-O2']
    build_system.ldflags = ['-static']
    return build_system


@test_util.dispatch('build_system')
def test_emit_from_env(build_system, environ):
    pytest.skip('unsupported for this build system')


@test_util.dispatch('build_system_with_flags')
def test_emit_from_buildsystem(build_system_with_flags, environ):
    pytest.skip('unsupported for this build system')


@test_util.dispatch('build_system')
def test_emit_no_env_defaults(build_system, environ):
    pytest.skip('unsupported for this build system')


def _emit_from_env_Make(build_system, environ):
    build_system.makefile = 'Makefile_foo'
    build_system.srcdir = 'foodir'
    build_system.options = ['FOO=1']
    build_system.max_concurrency = 32
    expected = [
        'make -f Makefile_foo -C foodir -j 32 CC="gcc" CXX="g++" '
        'FC="gfortran" NVCC="nvcc" CPPFLAGS="-DNDEBUG" '
        'CFLAGS="-Wall -std=c99" CXXFLAGS="-Wall -std=c++11" '
        'FCFLAGS="-Wall" LDFLAGS="-dynamic" FOO=1'
    ]
    assert expected == build_system.emit_build_commands(environ)


def _emit_from_env_CMake(build_system, environ):
    build_system.srcdir = 'src'
    build_system.builddir = 'build/foo'
    build_system.config_opts = ['-DFOO=1']
    build_system.make_opts = ['install']
    build_system.max_concurrency = 32
    expected = [
        'cd src',
        'mkdir -p build/foo',
        'cd build/foo',
        'cmake -DCMAKE_C_COMPILER="gcc" -DCMAKE_CXX_COMPILER="g++" '
        '-DCMAKE_Fortran_COMPILER="gfortran" '
        '-DCMAKE_CUDA_COMPILER="nvcc" '
        '-DCMAKE_C_FLAGS="-DNDEBUG -Wall -std=c99" '
        '-DCMAKE_CXX_FLAGS="-DNDEBUG -Wall -std=c++11" '
        '-DCMAKE_Fortran_FLAGS="-DNDEBUG -Wall" '
        '-DCMAKE_EXE_LINKER_FLAGS="-dynamic" -DFOO=1 ../..',
        'make -j 32 install'

    ]
    assert expected == build_system.emit_build_commands(environ)


def _emit_from_env_Autotools(build_system, environ):
    build_system.srcdir = 'src'
    build_system.builddir = 'build/foo'
    build_system.config_opts = ['FOO=1']
    build_system.make_opts = ['check']
    build_system.max_concurrency = 32
    expected = [
        'cd src',
        'mkdir -p build/foo',
        'cd build/foo',
        '../../configure CC="gcc" CXX="g++" FC="gfortran" '
        'CPPFLAGS="-DNDEBUG" CFLAGS="-Wall -std=c99" '
        'CXXFLAGS="-Wall -std=c++11" FCFLAGS="-Wall" '
        'LDFLAGS="-dynamic" FOO=1',
        'make -j 32 check'

    ]
    assert expected == build_system.emit_build_commands(environ)


def _emit_from_env_SingleSource(build_system, environ):
    build_system.include_path = ['foodir/include', 'bardir/include']
    build_system.executable = 'foo.x'
    build_system.srcfile = 'foo.c'
    expected = [
        'gcc -DNDEBUG -I foodir/include -I bardir/include '
        '-Wall -std=c99 foo.c -o foo.x -dynamic'
    ]
    assert expected == build_system.emit_build_commands(environ)


def _emit_from_env_CustomBuild(build_system, environ):
    build_system.commands = ['./custom-configure --foo', 'make']
    expected = ['./custom-configure --foo', 'make']
    assert expected == build_system.emit_build_commands(environ)


def _emit_from_buildsystem_Make(build_system_with_flags, environ):
    build_system_with_flags.makefile = 'Makefile_foo'
    build_system_with_flags.srcdir = 'foodir'
    build_system_with_flags.options = ['FOO=1']
    build_system_with_flags.max_concurrency = None
    expected = [
        'make -f Makefile_foo -C foodir -j CC="cc" CXX="CC" FC="ftn" '
        'NVCC="clang" CPPFLAGS="-DFOO" CFLAGS="-Wall -std=c99 -O2" '
        'CXXFLAGS="-Wall -std=c++11 -O2" FCFLAGS="-Wall -O2" '
        'LDFLAGS="-static" FOO=1'
    ]
    assert expected == build_system_with_flags.emit_build_commands(environ)


def _emit_from_buildsystem_CMake(build_system_with_flags, environ):
    build_system_with_flags.builddir = 'build/foo'
    build_system_with_flags.config_opts = ['-DFOO=1']
    build_system_with_flags.max_concurrency = None
    expected = [
        'mkdir -p build/foo',
        'cd build/foo',
        'cmake -DCMAKE_C_COMPILER="cc" -DCMAKE_CXX_COMPILER="CC" '
        '-DCMAKE_Fortran_COMPILER="ftn" -DCMAKE_CUDA_COMPILER="clang" '
        '-DCMAKE_C_FLAGS="-DFOO -Wall -std=c99 -O2" '
        '-DCMAKE_CXX_FLAGS="-DFOO -Wall -std=c++11 -O2" '
        '-DCMAKE_Fortran_FLAGS="-DFOO -Wall -O2" '
        '-DCMAKE_EXE_LINKER_FLAGS="-static" -DFOO=1 ../..',
        'make -j'

    ]
    assert expected == build_system_with_flags.emit_build_commands(environ)


def _emit_from_buildsystem_Autotools(build_system_with_flags, environ):
    build_system_with_flags.builddir = 'build/foo'
    build_system_with_flags.config_opts = ['FOO=1']
    build_system_with_flags.max_concurrency = None
    expected = [
        'mkdir -p build/foo',
        'cd build/foo',
        '../../configure CC="cc" CXX="CC" FC="ftn" '
        'CPPFLAGS="-DFOO" CFLAGS="-Wall -std=c99 -O2" '
        'CXXFLAGS="-Wall -std=c++11 -O2" FCFLAGS="-Wall -O2" '
        'LDFLAGS="-static" FOO=1',
        'make -j'

    ]
    assert expected == build_system_with_flags.emit_build_commands(environ)


def _emit_from_buildsystem_SingleSource(build_system_with_flags, environ):
    build_system_with_flags.include_path = ['foodir/include', 'bardir/include']
    build_system_with_flags.executable = 'foo.x'
    build_system_with_flags.srcfile = 'foo.c'
    expected = [
        'cc -DFOO -I foodir/include -I bardir/include '
        '-Wall -std=c99 -O2 foo.c -o foo.x -static'
    ]
    assert expected == build_system_with_flags.emit_build_commands(environ)


def _emit_from_buildsystem_CustomBuild(build_system_with_flags, environ):
    build_system_with_flags.commands = ['./custom-configure --foo', 'make']
    expected = ['./custom-configure --foo', 'make']
    assert expected == build_system_with_flags.emit_build_commands(environ)


def _emit_no_env_defaults_Make(build_system, environ):
    build_system.flags_from_environ = False
    assert ['make -j 1'] == build_system.emit_build_commands(environ)


def _emit_no_env_defaults_CMake(build_system, environ):
    build_system.flags_from_environ = False
    assert (['cmake .', 'make -j 1'] ==
            build_system.emit_build_commands(environ))


def _emit_no_env_defaults_Autotools(build_system, environ):
    build_system.flags_from_environ = False
    assert (['./configure', 'make -j 1'] ==
            build_system.emit_build_commands(environ))


def _emit_no_env_defaults_SingleSource(build_system, environ):
    build_system.cc = 'gcc'
    build_system.srcfile = 'foo.c'
    build_system.flags_from_environ = False
    assert (['gcc foo.c -o foo.x'] ==
            build_system.emit_build_commands(environ))


def _emit_no_env_defaults_CustomBuild(build_system, environ):
    build_system.commands = ['./custom-configure --foo', 'make']
    expected = ['./custom-configure --foo', 'make']
    assert expected == build_system.emit_build_commands(environ)


@pytest.fixture(params=['C', 'C++', 'Fortran', 'CUDA'])
def lang(request):
    return request.param


def test_compiler_pick(lang):
    ext = {'C': '.c', 'C++': '.cpp', 'Fortran': '.f90', 'CUDA': '.cu'}
    build_system = bs.SingleSource()
    build_system.cc = 'cc'
    build_system.cxx = 'CC'
    build_system.ftn = 'ftn'
    build_system.nvcc = 'nvcc'
    build_system.srcfile = 'foo' + ext[lang]
    compilers = {
        'C': build_system.cc,
        'C++': build_system.cxx,
        'Fortran': build_system.ftn,
        'CUDA': build_system.nvcc
    }
    assert ([f'{compilers[lang]} {build_system.srcfile} -o foo.x'] ==
            build_system.emit_build_commands(ProgEnvironment('testenv')))


def test_singlesource_unknown_language():
    build_system = bs.SingleSource()
    build_system.srcfile = 'foo.bar'
    with pytest.raises(BuildSystemError, match='could not guess language'):
        build_system.emit_build_commands(ProgEnvironment('testenv'))


def test_spack(environ, tmp_path):
    build_system = bs.Spack()
    build_system.environment = 'spack_env'
    build_system.install_opts = ['-j 10']
    with osext.change_dir(tmp_path):
        assert build_system.emit_build_commands(environ) == [
            f'spack -e {build_system.environment} install -j 10'
        ]
        assert build_system.prepare_cmds() == [
        ]


def test_spack_with_spec(environ, tmp_path):
    build_system = bs.Spack()
    build_system.environment = 'spack_env'
    build_system.specs = ['spec1@version1', 'spec2@version2']
    specs_str = ' '.join(build_system.specs)
    with osext.change_dir(tmp_path):
        assert build_system.emit_build_commands(environ) == [
            f'spack -e {build_system.environment} add {specs_str}',
            f'spack -e {build_system.environment} install'
        ]
        assert build_system.prepare_cmds() == [
            f'eval `spack -e {build_system.environment} load --sh {specs_str}`'
        ]


def test_spack_no_env(environ, tmp_path):
    build_system = bs.Spack()
    with osext.change_dir(tmp_path):
        assert build_system.emit_build_commands(environ) == [
            f'spack env create -d rfm_spack_env',
            f'spack -e rfm_spack_env config add '
            '"config:install_tree:root:opt/spack"',
            f'spack -e rfm_spack_env install'
        ]

    assert build_system.environment == 'rfm_spack_env'


def test_spack_env_config(environ, tmp_path):
    build_system = bs.Spack()
    build_system.env_create_opts = ['--without-view']
    build_system.config_opts = ['section1:header1:value1',
                                'section2:header2:value2']
    build_system.preinstall_cmds = ['echo hello', 'echo world']
    with osext.change_dir(tmp_path):
        assert build_system.emit_build_commands(environ) == [
            'spack env create -d rfm_spack_env --without-view',
            'spack -e rfm_spack_env config add "config:install_tree:root:opt/spack"',  # noqa: E501
            'spack -e rfm_spack_env config add "section1:header1:value1"',
            'spack -e rfm_spack_env config add "section2:header2:value2"',
            'echo hello',
            'echo world',
            'spack -e rfm_spack_env install',
        ]


def test_easybuild(environ, tmp_path):
    build_system = bs.EasyBuild()
    build_system.easyconfigs = ['ec1.eb', 'ec2.eb']
    build_system.options = ['-o1', '-o2']
    with osext.change_dir(tmp_path):
        assert build_system.emit_build_commands(environ) == [
            f'export EASYBUILD_BUILDPATH={tmp_path}/easybuild/build',
            f'export EASYBUILD_INSTALLPATH={tmp_path}/easybuild',
            f'export EASYBUILD_PREFIX={tmp_path}/easybuild',
            f'export EASYBUILD_SOURCEPATH={tmp_path}/easybuild',
            'eb ec1.eb ec2.eb -o1 -o2'
        ]


def test_easybuild_with_packaging(environ, tmp_path):
    build_system = bs.EasyBuild()
    build_system.easyconfigs = ['ec1.eb', 'ec2.eb']
    build_system.options = ['-o1', '-o2']
    build_system.emit_package = True
    build_system.package_opts = {
        'type': 'rpm',
        'tool-options': "'-o1 -o2'"
    }
    with osext.change_dir(tmp_path):
        assert build_system.emit_build_commands(environ) == [
            f'export EASYBUILD_BUILDPATH={tmp_path}/easybuild/build',
            f'export EASYBUILD_INSTALLPATH={tmp_path}/easybuild',
            f'export EASYBUILD_PREFIX={tmp_path}/easybuild',
            f'export EASYBUILD_SOURCEPATH={tmp_path}/easybuild',
            'eb ec1.eb ec2.eb -o1 -o2 --package --package-type=rpm '
            "--package-tool-options='-o1 -o2'"
        ]


def test_easybuild_no_easyconfigs(environ):
    build_system = bs.EasyBuild()
    with pytest.raises(BuildSystemError):
        build_system.emit_build_commands(environ)
