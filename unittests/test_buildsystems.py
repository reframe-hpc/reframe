import abc
import unittest

import reframe.core.buildsystems as bs
from reframe.core.environments import ProgEnvironment
from reframe.core.exceptions import BuildSystemError


class _BuildSystemTest:
    @abc.abstractmethod
    def create_build_system(self):
        pass

    def setUp(self):
        self.environ = ProgEnvironment(name='test_env',
                                       cc='gcc',
                                       cxx='g++',
                                       ftn='gfortran',
                                       cppflags='-DNDEBUG',
                                       cflags='-Wall -std=c99',
                                       cxxflags='-Wall -std=c++11',
                                       fflags='-Wall',
                                       ldflags='-dynamic')
        self.build_system = self.create_build_system()

    def setup_base_buildsystem(self):
        self.build_system.cc = 'cc'
        self.build_system.cxx = 'CC'
        self.build_system.ftn = 'ftn'
        self.build_system.nvcc = 'clang'
        self.build_system.cppflags = ['-DFOO']
        self.build_system.cflags = ['-Wall', '-std=c99', '-O3']
        self.build_system.cxxflags = ['-Wall', '-std=c++11', '-O3']
        self.build_system.fflags = ['-Wall', '-O3']
        self.build_system.ldflags = ['-static']


class TestMake(_BuildSystemTest, unittest.TestCase):
    def create_build_system(self):
        return bs.Make()

    def test_emit_from_env(self):
        self.build_system.makefile = 'Makefile_foo'
        self.build_system.srcdir = 'foodir'
        self.build_system.options = ['FOO=1']
        self.build_system.max_concurrency = 32
        expected = [
            "make -f Makefile_foo -C foodir -j 32 CC='gcc' CXX='g++' "
            "FC='gfortran' NVCC='nvcc' CPPFLAGS='-DNDEBUG' "
            "CFLAGS='-Wall -std=c99' CXXFLAGS='-Wall -std=c++11' "
            "FCFLAGS='-Wall' LDFLAGS='-dynamic' FOO=1"
        ]
        self.assertEqual(expected,
                         self.build_system.emit_build_commands(self.environ))

    def test_emit_from_buildsystem(self):
        super().setup_base_buildsystem()
        self.build_system.makefile = 'Makefile_foo'
        self.build_system.srcdir = 'foodir'
        self.build_system.options = ['FOO=1']
        self.build_system.max_concurrency = None
        expected = [
            "make -f Makefile_foo -C foodir -j CC='cc' CXX='CC' FC='ftn' "
            "NVCC='clang' CPPFLAGS='-DFOO' CFLAGS='-Wall -std=c99 -O3' "
            "CXXFLAGS='-Wall -std=c++11 -O3' FCFLAGS='-Wall -O3' "
            "LDFLAGS='-static' FOO=1"
        ]
        self.assertEqual(expected,
                         self.build_system.emit_build_commands(self.environ))

    def test_emit_no_env_defaults(self):
        self.build_system.flags_from_environ = False
        self.assertEqual(['make -j'],
                         self.build_system.emit_build_commands(self.environ))


class TestCMake(_BuildSystemTest, unittest.TestCase):
    def create_build_system(self):
        return bs.CMake()

    def test_emit_from_env(self):
        self.build_system.srcdir = 'src'
        self.build_system.builddir = 'build/foo'
        self.build_system.config_opts = ['-DFOO=1']
        self.build_system.make_opts = ['install']
        self.build_system.max_concurrency = 32
        expected = [
            "cd src",
            "mkdir -p build/foo",
            "cd build/foo",
            "cmake -DCMAKE_C_COMPILER='gcc' -DCMAKE_CXX_COMPILER='g++' "
            "-DCMAKE_Fortran_COMPILER='gfortran' "
            "-DCMAKE_CUDA_COMPILER='nvcc' "
            "-DCMAKE_C_FLAGS='-DNDEBUG -Wall -std=c99' "
            "-DCMAKE_CXX_FLAGS='-DNDEBUG -Wall -std=c++11' "
            "-DCMAKE_Fortran_FLAGS='-DNDEBUG -Wall' "
            "-DCMAKE_EXE_LINKER_FLAGS='-dynamic' -DFOO=1 ../..",
            "make -j 32 install"

        ]
        self.assertEqual(expected,
                         self.build_system.emit_build_commands(self.environ))

    def test_emit_from_buildsystem(self):
        super().setup_base_buildsystem()
        self.build_system.builddir = 'build/foo'
        self.build_system.config_opts = ['-DFOO=1']
        self.build_system.max_concurrency = None
        expected = [
            "mkdir -p build/foo",
            "cd build/foo",
            "cmake -DCMAKE_C_COMPILER='cc' -DCMAKE_CXX_COMPILER='CC' "
            "-DCMAKE_Fortran_COMPILER='ftn' -DCMAKE_CUDA_COMPILER='clang' "
            "-DCMAKE_C_FLAGS='-DFOO -Wall -std=c99 -O3' "
            "-DCMAKE_CXX_FLAGS='-DFOO -Wall -std=c++11 -O3' "
            "-DCMAKE_Fortran_FLAGS='-DFOO -Wall -O3' "
            "-DCMAKE_EXE_LINKER_FLAGS='-static' -DFOO=1 ../..",
            "make -j"

        ]
        print(self.build_system.emit_build_commands(self.environ))
        self.assertEqual(expected,
                         self.build_system.emit_build_commands(self.environ))

    def test_emit_no_env_defaults(self):
        self.build_system.flags_from_environ = False
        self.assertEqual(['cmake .', 'make -j'],
                         self.build_system.emit_build_commands(self.environ))


class TestAutotools(_BuildSystemTest, unittest.TestCase):
    def create_build_system(self):
        return bs.Autotools()

    def test_emit_from_env(self):
        self.build_system.srcdir = 'src'
        self.build_system.builddir = 'build/foo'
        self.build_system.config_opts = ['FOO=1']
        self.build_system.make_opts = ['check']
        self.build_system.max_concurrency = 32
        expected = [
            "cd src",
            "mkdir -p build/foo",
            "cd build/foo",
            "../../configure CC='gcc' CXX='g++' FC='gfortran' "
            "CPPFLAGS='-DNDEBUG' CFLAGS='-Wall -std=c99' "
            "CXXFLAGS='-Wall -std=c++11' FCFLAGS='-Wall' "
            "LDFLAGS='-dynamic' FOO=1",
            "make -j 32 check"

        ]
        self.assertEqual(expected,
                         self.build_system.emit_build_commands(self.environ))

    def test_emit_from_buildsystem(self):
        super().setup_base_buildsystem()
        self.build_system.builddir = 'build/foo'
        self.build_system.config_opts = ['FOO=1']
        self.build_system.max_concurrency = None
        expected = [
            "mkdir -p build/foo",
            "cd build/foo",
            "../../configure CC='cc' CXX='CC' FC='ftn' "
            "CPPFLAGS='-DFOO' CFLAGS='-Wall -std=c99 -O3' "
            "CXXFLAGS='-Wall -std=c++11 -O3' FCFLAGS='-Wall -O3' "
            "LDFLAGS='-static' FOO=1",
            "make -j"

        ]
        print(self.build_system.emit_build_commands(self.environ))
        self.assertEqual(expected,
                         self.build_system.emit_build_commands(self.environ))

    def test_emit_no_env_defaults(self):
        self.build_system.flags_from_environ = False
        self.assertEqual(['./configure', 'make -j'],
                         self.build_system.emit_build_commands(self.environ))


class TestSingleSource(_BuildSystemTest, unittest.TestCase):
    def create_build_system(self):
        return bs.SingleSource()

    def setUp(self):
        super().setUp()
        self.compilers = {
            'C': 'gcc',
            'C++': 'g++',
            'Fortran': 'gfortran',
            'CUDA': 'nvcc',
        }

        self.flags = {
            'C': '-Wall -std=c99',
            'C++': '-Wall -std=c++11',
            'Fortran': '-Wall',
        }

        self.files = {
            'C': 'foo.c',
            'C++': 'foo.cpp',
            'Fortran': 'foo.f90',
            'CUDA': 'foo.cu',
        }

    def test_emit_from_env(self):
        self.build_system.include_path = ['foodir/include', 'bardir/include']
        self.build_system.executable = 'foo.e'
        for lang, comp in self.compilers.items():
            self.build_system.srcfile = self.files[lang]
            if lang == 'CUDA':
                flags = self.flags['C++']
            else:
                flags = self.flags[lang]

            ldflags = '-dynamic'
            expected = [
                '%s -DNDEBUG -I foodir/include -I bardir/include %s %s '
                '-o foo.e %s' % (comp, flags,
                                 self.build_system.srcfile, ldflags)
            ]
            self.assertEqual(expected,
                             self.build_system.emit_build_commands(self.environ))

    def test_emit_no_env(self):
        super().setup_base_buildsystem()
        self.compilers = {
            'C': 'cc',
            'C++': 'CC',
            'Fortran': 'ftn',
            'CUDA': 'clang',
        }

        self.flags = {
            'C': '-Wall -std=c99 -O3',
            'C++': '-Wall -std=c++11 -O3',
            'Fortran': '-Wall -O3',
        }
        self.build_system.include_path = ['foodir/include', 'bardir/include']
        self.build_system.executable = 'foo.e'
        for lang, comp in self.compilers.items():
            self.build_system.srcfile = self.files[lang]
            if lang == 'CUDA':
                flags = self.flags['C++']
            else:
                flags = self.flags[lang]

            ldflags = '-static'
            expected = [
                '%s -DFOO -I foodir/include -I bardir/include %s %s '
                '-o foo.e %s' % (comp, flags,
                                 self.build_system.srcfile, ldflags)
            ]
            self.assertEqual(expected,
                             self.build_system.emit_build_commands(self.environ))
