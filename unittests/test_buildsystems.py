# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import abc
import unittest
import pytest

import reframe.core.buildsystems as bs
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


@pytest.fixture
def setup_build_system():
    def _setup_build_system(build_system):
        build_system.cc = 'cc'
        build_system.cxx = 'CC'
        build_system.ftn = 'ftn'
        build_system.nvcc = 'clang'
        build_system.cppflags = ['-DFOO']
        build_system.cflags = ['-Wall', '-std=c99', '-O3']
        build_system.cxxflags = ['-Wall', '-std=c++11', '-O3']
        build_system.fflags = ['-Wall', '-O3']
        build_system.ldflags = ['-static']

    return _setup_build_system


@pytest.fixture
def make_build_system():
    return bs.Make()


@pytest.fixture
def cmake_build_system():
    return bs.CMake()


@pytest.fixture
def autotools_build_system():
    return bs.Autotools()


@pytest.fixture
def single_source_build_system():
    return bs.SingleSource()


def test_make_emit_from_env(environ, make_build_system):
    make_build_system.makefile = 'Makefile_foo'
    make_build_system.srcdir = 'foodir'
    make_build_system.options = ['FOO=1']
    make_build_system.max_concurrency = 32
    expected = [
        'make -f Makefile_foo -C foodir -j 32 CC="gcc" CXX="g++" '
        'FC="gfortran" NVCC="nvcc" CPPFLAGS="-DNDEBUG" '
        'CFLAGS="-Wall -std=c99" CXXFLAGS="-Wall -std=c++11" '
        'FCFLAGS="-Wall" LDFLAGS="-dynamic" FOO=1'
    ]
    assert expected == make_build_system.emit_build_commands(environ)


def test_make_emit_from_buildsystem(environ, make_build_system,
                                    setup_build_system):
    setup_build_system(make_build_system)
    make_build_system.makefile = 'Makefile_foo'
    make_build_system.srcdir = 'foodir'
    make_build_system.options = ['FOO=1']
    make_build_system.max_concurrency = None
    expected = [
        'make -f Makefile_foo -C foodir -j CC="cc" CXX="CC" FC="ftn" '
        'NVCC="clang" CPPFLAGS="-DFOO" CFLAGS="-Wall -std=c99 -O3" '
        'CXXFLAGS="-Wall -std=c++11 -O3" FCFLAGS="-Wall -O3" '
        'LDFLAGS="-static" FOO=1'
    ]
    assert expected == make_build_system.emit_build_commands(environ)


def test_make_emit_no_env_defaults(environ, make_build_system):
    make_build_system.flags_from_environ = False
    assert ['make -j 1'] ==  make_build_system.emit_build_commands(environ)


def test_cmake_emit_from_env(environ, cmake_build_system):
    cmake_build_system.srcdir = 'src'
    cmake_build_system.builddir = 'build/foo'
    cmake_build_system.config_opts = ['-DFOO=1']
    cmake_build_system.make_opts = ['install']
    cmake_build_system.max_concurrency = 32
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
    assert expected == cmake_build_system.emit_build_commands(environ)


def test_cmake_emit_from_buildsystem(environ, cmake_build_system,
                                     setup_build_system):
    setup_build_system(cmake_build_system)
    cmake_build_system.builddir = 'build/foo'
    cmake_build_system.config_opts = ['-DFOO=1']
    cmake_build_system.max_concurrency = None
    expected = [
        'mkdir -p build/foo',
        'cd build/foo',
        'cmake -DCMAKE_C_COMPILER="cc" -DCMAKE_CXX_COMPILER="CC" '
        '-DCMAKE_Fortran_COMPILER="ftn" -DCMAKE_CUDA_COMPILER="clang" '
        '-DCMAKE_C_FLAGS="-DFOO -Wall -std=c99 -O3" '
        '-DCMAKE_CXX_FLAGS="-DFOO -Wall -std=c++11 -O3" '
        '-DCMAKE_Fortran_FLAGS="-DFOO -Wall -O3" '
        '-DCMAKE_EXE_LINKER_FLAGS="-static" -DFOO=1 ../..',
        'make -j'

    ]
    assert expected == cmake_build_system.emit_build_commands(environ)


def test_cmake_emit_no_env_defaults(environ, cmake_build_system):
    cmake_build_system.flags_from_environ = False
    assert (['cmake .', 'make -j 1'] ==
            cmake_build_system.emit_build_commands(environ))


def test_autotools_emit_from_env(environ, autotools_build_system):
    autotools_build_system.srcdir = 'src'
    autotools_build_system.builddir = 'build/foo'
    autotools_build_system.config_opts = ['FOO=1']
    autotools_build_system.make_opts = ['check']
    autotools_build_system.max_concurrency = 32
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
    assert expected == autotools_build_system.emit_build_commands(environ)


def test_autotools_emit_from_buildsystem(environ, autotools_build_system,
                                         setup_build_system):
    setup_build_system(autotools_build_system)
    autotools_build_system.builddir = 'build/foo'
    autotools_build_system.config_opts = ['FOO=1']
    autotools_build_system.max_concurrency = None
    expected = [
        'mkdir -p build/foo',
        'cd build/foo',
        '../../configure CC="cc" CXX="CC" FC="ftn" '
        'CPPFLAGS="-DFOO" CFLAGS="-Wall -std=c99 -O3" '
        'CXXFLAGS="-Wall -std=c++11 -O3" FCFLAGS="-Wall -O3" '
        'LDFLAGS="-static" FOO=1',
        'make -j'

    ]
    assert expected == autotools_build_system.emit_build_commands(environ)

def test_autotools_emit_no_env_defaults(environ, autotools_build_system):
    autotools_build_system.flags_from_environ = False
    assert (['./configure', 'make -j 1'] ==
            autotools_build_system.emit_build_commands(environ))


@pytest.fixture
def compilers():
    return {
        'C': 'gcc',
        'C++': 'g++',
        'Fortran': 'gfortran',
        'CUDA': 'nvcc',
    }


@pytest.fixture
def flags():
    return {
        'C': '-Wall -std=c99',
        'C++': '-Wall -std=c++11',
        'Fortran': '-Wall',
    }


@pytest.fixture
def files():
    return {
        'C': 'foo.c',
        'C++': 'foo.cpp',
        'Fortran': 'foo.f90',
        'CUDA': 'foo.cu',
    }


def test_emit_from_env(environ, single_source_build_system, compilers, flags,
                       files):
    single_source_build_system.include_path = ['foodir/include',
                                               'bardir/include']
    single_source_build_system.executable = 'foo.e'
    for lang, comp in compilers.items():
        single_source_build_system.srcfile = files[lang]
        if lang == 'CUDA':
            lang_flags = flags['C++']
        else:
            lang_flags = flags[lang]

        ldflags = '-dynamic'
        expected = [
            f'{comp} -DNDEBUG -I foodir/include -I bardir/include '
            f'{lang_flags} {single_source_build_system.srcfile} '
            f'-o foo.e {ldflags}'
        ]
        assert (expected ==
                single_source_build_system.emit_build_commands(environ))


def test_emit_no_env(environ, setup_build_system, single_source_build_system,
                     files):
    setup_build_system(single_source_build_system)
    compilers = {
        'C': 'cc',
        'C++': 'CC',
        'Fortran': 'ftn',
        'CUDA': 'clang',
    }

    flags = {
        'C': '-Wall -std=c99 -O3',
        'C++': '-Wall -std=c++11 -O3',
        'Fortran': '-Wall -O3',
    }
    single_source_build_system.include_path = ['foodir/include',
                                              'bardir/include']
    single_source_build_system.executable = 'foo.e'
    for lang, comp in compilers.items():
        single_source_build_system.srcfile = files[lang]
        if lang == 'CUDA':
            lang_flags = flags['C++']
        else:
            lang_flags = flags[lang]

        ldflags = '-static'
        expected = [
            f'{comp} -DFOO -I foodir/include -I bardir/include {lang_flags} '
            f'{single_source_build_system.srcfile} -o foo.e {ldflags}'
        ]
        assert (expected ==
                single_source_build_system.emit_build_commands(environ))
