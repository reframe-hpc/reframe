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
        self.build_system.flags_from_environ = False


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
            "FFLAGS='-Wall' LDFLAGS='-dynamic' FOO=1 || exit 1"
        ]
        self.assertEqual(expected,
                         self.build_system.emit_build_commands(self.environ))

    def test_emit_no_env(self):
        super().setup_base_buildsystem()
        self.build_system.makefile = 'Makefile_foo'
        self.build_system.srcdir = 'foodir'
        self.build_system.options = ['FOO=1']
        self.build_system.max_concurrency = None
        expected = [
            "make -f Makefile_foo -C foodir -j CC='cc' CXX='CC' FC='ftn' "
            "NVCC='clang' CPPFLAGS='-DFOO' CFLAGS='-Wall -std=c99 -O3' "
            "CXXFLAGS='-Wall -std=c++11 -O3' FFLAGS='-Wall -O3' "
            "LDFLAGS='-static' FOO=1 || exit 1"
        ]
        self.assertEqual(expected,
                         self.build_system.emit_build_commands(self.environ))

    def test_emit_no_env_defaults(self):
        self.build_system.flags_from_environ = False
        self.assertEqual(['make -j || exit 1'],
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
                '-o foo.e %s || exit 1' % (comp, flags,
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
                '-o foo.e %s || exit 1' % (comp, flags,
                                           self.build_system.srcfile, ldflags)
            ]
            self.assertEqual(expected,
                             self.build_system.emit_build_commands(self.environ))
