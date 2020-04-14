# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import pytest
import re
import tempfile
import unittest

import reframe as rfm
import reframe.core.runtime as rt
import reframe.utility.os_ext as os_ext
import reframe.utility.sanity as sn
import unittests.fixtures as fixtures
from reframe.core.exceptions import (BuildError, PipelineError, ReframeError,
                                     ReframeSyntaxError, PerformanceError,
                                     SanityError)
from reframe.frontend.loader import RegressionCheckLoader
from unittests.resources.checks.hellocheck import HelloTest


def _setup_local_execution():
    partition = rt.runtime().system.partition('login')
    environ = partition.environment('builtin-gcc')
    return partition, environ


def _setup_remote_execution(scheduler=None):
    partition = fixtures.partition_with_scheduler(scheduler)
    if partition is None:
        pytest.skip('job submission not supported')

    try:
        environ = partition.environs[0]
    except IndexError:
        pytest.skip('no environments configured for partition: %s' %
                    partition.fullname)

    return partition, environ


def _run(test, partition, prgenv):
    test.setup(partition, prgenv)
    test.compile()
    test.compile_wait()
    test.run()
    test.wait()
    test.check_sanity()
    test.check_performance()
    test.cleanup(remove_files=True)


def _cray_cle_version():
    completed = os_ext.run_command('cat /etc/opt/cray/release/cle-release')
    matched = re.match(r'^RELEASE=(\S+)', completed.stdout)
    if matched is None:
        return None

    return matched.group(1)


class TestRegressionTest(unittest.TestCase):
    def setUp(self):
        self.partition, self.prgenv = _setup_local_execution()
        self.loader = RegressionCheckLoader(['unittests/resources/checks'])

        # Set runtime prefix
        rt.runtime().resources.prefix = tempfile.mkdtemp(dir='unittests')

    def tearDown(self):
        os_ext.rmtree(rt.runtime().resources.prefix)
        os_ext.rmtree('.rfm_testing', ignore_errors=True)

    def replace_prefix(self, filename, new_prefix):
        basename = os.path.basename(filename)
        return os.path.join(new_prefix, basename)

    def keep_files_list(self, test, compile_only=False):
        ret = [self.replace_prefix(sn.evaluate(test.stdout), test.outputdir),
               self.replace_prefix(sn.evaluate(test.stderr), test.outputdir)]

        if not compile_only:
            ret.append(self.replace_prefix(test.job.script_filename,
                                           test.outputdir))

        ret.extend([self.replace_prefix(f, test.outputdir)
                    for f in test.keep_files])
        return ret

    def test_environ_setup(self):
        test = self.loader.load_from_file(
            'unittests/resources/checks/hellocheck.py')[0]

        # Use test environment for the regression check
        test.valid_prog_environs = [self.prgenv.name]
        test.modules = ['testmod_foo']
        test.variables = {'_FOO_': '1', '_BAR_': '2'}
        test.local = True

        test.setup(self.partition, self.prgenv)

        for k in test.variables.keys():
            assert k not in os.environ

    def _run_test(self, test, compile_only=False):
        _run(test, self.partition, self.prgenv)
        assert not os.path.exists(test.stagedir)
        for f in self.keep_files_list(test, compile_only):
            assert os.path.exists(f)

    @fixtures.switch_to_user_runtime
    def test_hellocheck(self):
        self.partition, self.prgenv = _setup_remote_execution()
        test = self.loader.load_from_file(
            'unittests/resources/checks/hellocheck.py')[0]

        # Use test environment for the regression check
        test.valid_prog_environs = [self.prgenv.name]
        self._run_test(test)

    @fixtures.switch_to_user_runtime
    def test_hellocheck_make(self):
        self.partition, self.prgenv = _setup_remote_execution()
        test = self.loader.load_from_file(
            'unittests/resources/checks/hellocheck_make.py')[0]

        # Use test environment for the regression check
        test.valid_prog_environs = [self.prgenv.name]
        self._run_test(test)

    def test_hellocheck_local(self):
        test = self.loader.load_from_file(
            'unittests/resources/checks/hellocheck.py')[0]

        # Use test environment for the regression check
        test.valid_prog_environs = [self.prgenv.name]

        # Test also the prebuild/postbuild functionality
        test.prebuild_cmd  = ['touch prebuild', 'mkdir prebuild_dir']
        test.postbuild_cmd = ['touch postbuild', 'mkdir postbuild_dir']
        test.keep_files = ['prebuild', 'postbuild',
                           'prebuild_dir', 'postbuild_dir']

        # Force local execution of the test
        test.local = True
        self._run_test(test)

    def test_hellocheck_local_prepost_run(self):
        @sn.sanity_function
        def stagedir(test):
            return test.stagedir

        test = self.loader.load_from_file(
            'unittests/resources/checks/hellocheck.py')[0]

        # Use test environment for the regression check
        test.valid_prog_environs = [self.prgenv.name]

        # Test also the prebuild/postbuild functionality
        test.pre_run  = ['echo prerun: `pwd`']
        test.post_run = ['echo postrun: `pwd`']
        pre_run_path = sn.extractsingle(r'^prerun: (\S+)', test.stdout, 1)
        post_run_path = sn.extractsingle(r'^postrun: (\S+)', test.stdout, 1)
        test.sanity_patterns = sn.all([
            sn.assert_eq(stagedir(test), pre_run_path),
            sn.assert_eq(stagedir(test), post_run_path),
        ])

        # Force local execution of the test
        test.local = True
        self._run_test(test)

    def test_hellocheck_local_prepost_run_in_setup(self):
        def custom_setup(obj, partition, environ, **job_opts):
            super(obj.__class__, obj).setup(partition, environ, **job_opts)
            obj.pre_run  = ['echo Prerunning cmd from setup phase']
            obj.post_run = ['echo Postruning cmd from setup phase']

        test = self.loader.load_from_file(
            'unittests/resources/checks/hellocheck.py')[0]

        # Monkey patch the setup method of the test
        test.setup = custom_setup.__get__(test)

        # Use test environment for the regression check
        test.valid_prog_environs = ['*']

        test.sanity_patterns = sn.all([
            sn.assert_found(r'^Prerunning cmd from setup phase', test.stdout),
            sn.assert_found(r'Hello, World\!', test.stdout),
            sn.assert_found(r'^Postruning cmd from setup phase', test.stdout)
        ])

        # Force local execution of the test
        test.local = True
        self._run_test(test)

    def test_run_only_sanity(self):
        @fixtures.custom_prefix('unittests/resources/checks')
        class MyTest(rfm.RunOnlyRegressionTest):
            def __init__(self):
                self.executable = './hello.sh'
                self.executable_opts = ['Hello, World!']
                self.local = True
                self.valid_prog_environs = ['*']
                self.valid_systems = ['*']
                self.sanity_patterns = sn.assert_found(
                    r'Hello, World\!', self.stdout)

        self._run_test(MyTest())

    def test_run_only_no_srcdir(self):
        @fixtures.custom_prefix('foo/bar/')
        class MyTest(rfm.RunOnlyRegressionTest):
            def __init__(self):
                self.executable = 'echo'
                self.executable_opts = ['hello']
                self.valid_prog_environs = ['*']
                self.valid_systems = ['*']
                self.sanity_patterns = sn.assert_found(r'hello', self.stdout)

        test = MyTest()
        assert test.sourcesdir is None
        self._run_test(MyTest())

    def test_compile_only_failure(self):
        @fixtures.custom_prefix('unittests/resources/checks')
        class MyTest(rfm.CompileOnlyRegressionTest):
            def __init__(self):
                self.sourcepath = 'compiler_failure.c'
                self.valid_prog_environs = ['*']
                self.valid_systems = ['*']

        test = MyTest()
        test.setup(self.partition, self.prgenv)
        test.compile()
        with pytest.raises(BuildError):
            test.compile_wait()

    def test_compile_only_warning(self):
        @fixtures.custom_prefix('unittests/resources/checks')
        class MyTest(rfm.RunOnlyRegressionTest):
            def __init__(self):
                self.build_system = 'SingleSource'
                self.build_system.srcfile = 'compiler_warning.c'
                self.build_system.cflags = ['-Wall']
                self.valid_prog_environs = ['*']
                self.valid_systems = ['*']
                self.sanity_patterns = sn.assert_found(r'warning', self.stderr)

        self._run_test(MyTest(), compile_only=True)

    @rt.switch_runtime(fixtures.TEST_SITE_CONFIG, 'testsys')
    def test_supports_system(self):
        test = self.loader.load_from_file(
            'unittests/resources/checks/hellocheck.py')[0]

        test.valid_systems = ['*']
        assert test.supports_system('gpu')
        assert test.supports_system('login')
        assert test.supports_system('testsys:gpu')
        assert test.supports_system('testsys:login')

        test.valid_systems = ['*:*']
        assert test.supports_system('gpu')
        assert test.supports_system('login')
        assert test.supports_system('testsys:gpu')
        assert test.supports_system('testsys:login')

        test.valid_systems = ['testsys']
        assert test.supports_system('gpu')
        assert test.supports_system('login')
        assert test.supports_system('testsys:gpu')
        assert test.supports_system('testsys:login')

        test.valid_systems = ['testsys:gpu']
        assert test.supports_system('gpu')
        assert not test.supports_system('login')
        assert test.supports_system('testsys:gpu')
        assert not test.supports_system('testsys:login')

        test.valid_systems = ['testsys:login']
        assert not test.supports_system('gpu')
        assert test.supports_system('login')
        assert not test.supports_system('testsys:gpu')
        assert test.supports_system('testsys:login')

        test.valid_systems = ['foo']
        assert not test.supports_system('gpu')
        assert not test.supports_system('login')
        assert not test.supports_system('testsys:gpu')
        assert not test.supports_system('testsys:login')

        test.valid_systems = ['*:gpu']
        assert test.supports_system('testsys:gpu')
        assert test.supports_system('foo:gpu')
        assert not test.supports_system('testsys:cpu')
        assert not test.supports_system('testsys:login')

        test.valid_systems = ['testsys:*']
        assert test.supports_system('testsys:login')
        assert test.supports_system('gpu')
        assert not test.supports_system('foo:gpu')

    def test_supports_environ(self):
        test = self.loader.load_from_file(
            'unittests/resources/checks/hellocheck.py')[0]

        test.valid_prog_environs = ['*']
        assert test.supports_environ('foo1')
        assert test.supports_environ('foo-env')
        assert test.supports_environ('*')

    def test_sourcesdir_none(self):
        @fixtures.custom_prefix('unittests/resources/checks')
        class MyTest(rfm.RegressionTest):
            def __init__(self):
                self.sourcesdir = None
                self.valid_prog_environs = ['*']
                self.valid_systems = ['*']

        with pytest.raises(ReframeError):
            self._run_test(MyTest())

    def test_sourcesdir_build_system(self):
        @fixtures.custom_prefix('unittests/resources/checks')
        class MyTest(rfm.RegressionTest):
            def __init__(self):
                self.build_system = 'Make'
                self.sourcepath = 'code'
                self.executable = './code/hello'
                self.local = True
                self.valid_systems = ['*']
                self.valid_prog_environs = ['*']
                self.sanity_patterns = sn.assert_found(r'Hello, World\!',
                                                       self.stdout)

        self._run_test(MyTest())

    def test_sourcesdir_none_generated_sources(self):
        @fixtures.custom_prefix('unittests/resources/checks')
        class MyTest(rfm.RegressionTest):
            def __init__(self):
                self.sourcesdir = None
                self.prebuild_cmd = [
                    "printf '#include <stdio.h>\\n int main(){ "
                    "printf(\"Hello, World!\\\\n\"); return 0; }' > hello.c"
                ]
                self.executable = './hello'
                self.sourcepath = 'hello.c'
                self.local = True
                self.valid_systems = ['*']
                self.valid_prog_environs = ['*']
                self.sanity_patterns = sn.assert_found(r'Hello, World\!',
                                                       self.stdout)

        self._run_test(MyTest())

    def test_sourcesdir_none_compile_only(self):
        @fixtures.custom_prefix('unittests/resources/checks')
        class MyTest(rfm.CompileOnlyRegressionTest):
            def __init__(self):
                self.sourcesdir = None
                self.valid_prog_environs = ['*']
                self.valid_systems = ['*']

        with pytest.raises(BuildError):
            self._run_test(MyTest())

    def test_sourcesdir_none_run_only(self):
        @fixtures.custom_prefix('unittests/resources/checks')
        class MyTest(rfm.RunOnlyRegressionTest):
            def __init__(self):
                self.sourcesdir = None
                self.executable = 'echo'
                self.executable_opts = ["Hello, World!"]
                self.local = True
                self.valid_prog_environs = ['*']
                self.valid_systems = ['*']
                self.sanity_patterns = sn.assert_found(r'Hello, World\!',
                                                       self.stdout)

        self._run_test(MyTest())

    def test_sourcepath_abs(self):
        @fixtures.custom_prefix('unittests/resources/checks')
        class MyTest(rfm.CompileOnlyRegressionTest):
            def __init__(self):
                self.valid_prog_environs = ['*']
                self.valid_systems = ['*']

        test = MyTest()
        test.setup(self.partition, self.prgenv)
        test.sourcepath = '/usr/src'
        with pytest.raises(PipelineError):
            test.compile()

    def test_sourcepath_upref(self):
        @fixtures.custom_prefix('unittests/resources/checks')
        class MyTest(rfm.CompileOnlyRegressionTest):
            def __init__(self):
                self.valid_prog_environs = ['*']
                self.valid_systems = ['*']

        test = MyTest()
        test.setup(self.partition, self.prgenv)
        test.sourcepath = '../hellosrc'
        with pytest.raises(PipelineError):
            test.compile()

    @rt.switch_runtime(fixtures.TEST_SITE_CONFIG, 'testsys')
    def test_extra_resources(self):
        @fixtures.custom_prefix('unittests/resources/checks')
        class MyTest(HelloTest):
            def __init__(self):
                super().__init__()
                self.name = type(self).__name__
                self.executable = os.path.join('.', self.name)
                self.local = True

            @rfm.run_after('setup')
            def set_resources(self):
                test.extra_resources = {
                    'gpu': {'num_gpus_per_node': 2},
                    'datawarp': {'capacity': '100GB',
                                 'stagein_src': test.stagedir}
                }
                test.job.options += ['--foo']

        test = MyTest()
        partition = rt.runtime().system.partition('gpu')
        environ = partition.environment('builtin-gcc')
        _run(test, partition, environ)
        expected_job_options = ['--gres=gpu:2',
                                '#DW jobdw capacity=100GB',
                                '#DW stage_in source=%s' % test.stagedir,
                                '--foo']
        self.assertCountEqual(expected_job_options, test.job.options)


class TestHooks(unittest.TestCase):
    def setUp(self):
        self.partition = rt.runtime().system.partition('login')
        self.prgenv = self.partition.environment('builtin-gcc')

        # Set runtime prefix
        rt.runtime().resources.prefix = tempfile.mkdtemp(dir='unittests')

    def tearDown(self):
        os_ext.rmtree(rt.runtime().resources.prefix)

    def test_setup_hooks(self):
        @fixtures.custom_prefix('unittests/resources/checks')
        class MyTest(HelloTest):
            def __init__(self):
                super().__init__()
                self.name = type(self).__name__
                self.executable = os.path.join('.', self.name)

            @rfm.run_before('setup')
            def prefoo(self):
                assert self.current_environ is None
                os.environ['_RFM_PRE_SETUP'] = 'foo'

            @rfm.run_after('setup')
            def postfoo(self):
                assert self.current_environ is not None
                os.environ['_RFM_POST_SETUP'] = 'foo'

        test = MyTest()
        _run(test, self.partition, self.prgenv)
        assert '_RFM_PRE_SETUP' in os.environ
        assert '_RFM_POST_SETUP' in os.environ

    def test_setup_hooks_in_compile_only_test(self):
        @fixtures.custom_prefix('unittests/resources/checks')
        class MyTest(rfm.CompileOnlyRegressionTest):
            def __init__(self):
                self.name = 'hellocheck_compile'
                self.valid_systems = ['*']
                self.valid_prog_environs = ['*']
                self.sourcepath = 'hello.c'
                self.executable = os.path.join('.', self.name)
                self.sanity_patterns = sn.assert_found('.*', self.stdout)
                self.count = 0

            @rfm.run_before('setup')
            def presetup(self):
                self.count += 1

            @rfm.run_after('setup')
            def postsetup(self):
                self.count += 1

        test = MyTest()
        _run(test, self.partition, self.prgenv)
        assert test.count == 2

    def test_compile_hooks(self):
        @fixtures.custom_prefix('unittests/resources/checks')
        class MyTest(HelloTest):
            def __init__(self):
                super().__init__()
                self.name = type(self).__name__
                self.executable = os.path.join('.', self.name)

            @rfm.run_before('compile')
            def setflags(self):
                os.environ['_RFM_PRE_COMPILE'] = 'FOO'

            @rfm.run_after('compile')
            def check_executable(self):
                exec_file = os.path.join(self.stagedir, self.executable)

                # Make sure that this hook is executed after compile_wait()
                assert os.path.exists(exec_file)

    def test_run_hooks(self):
        @fixtures.custom_prefix('unittests/resources/checks')
        class MyTest(HelloTest):
            def __init__(self):
                super().__init__()
                self.name = type(self).__name__
                self.executable = os.path.join('.', self.name)

            @rfm.run_before('run')
            def setflags(self):
                self.post_run = ['echo hello > greetings.txt']

            @rfm.run_after('run')
            def check_executable(self):
                outfile = os.path.join(self.stagedir, 'greetings.txt')

                # Make sure that this hook is executed after wait()
                assert os.path.exists(outfile)

        test = MyTest()
        _run(test, self.partition, self.prgenv)

    def test_run_hooks_in_run_only_test(self):
        @fixtures.custom_prefix('unittests/resources/checks')
        class MyTest(rfm.RunOnlyRegressionTest):
            def __init__(self):
                self.executable = 'echo'
                self.executable_opts = ['Hello, World!']
                self.local = True
                self.valid_prog_environs = ['*']
                self.valid_systems = ['*']
                self.sanity_patterns = sn.assert_found(
                    r'Hello, World\!', self.stdout)

            @rfm.run_before('run')
            def check_empty_stage(self):
                # Make sure nothing has been copied to the stage directory yet
                assert len(os.listdir(self.stagedir)) == 0

        test = MyTest()
        _run(test, self.partition, self.prgenv)

    def test_multiple_hooks(self):
        @fixtures.custom_prefix('unittests/resources/checks')
        class MyTest(HelloTest):
            def __init__(self):
                super().__init__()
                self.name = type(self).__name__
                self.executable = os.path.join('.', self.name)
                self.var = 0

            @rfm.run_after('setup')
            def x(self):
                self.var += 1

            @rfm.run_after('setup')
            def y(self):
                self.var += 1

            @rfm.run_after('setup')
            def z(self):
                self.var += 1

        test = MyTest()
        _run(test, self.partition, self.prgenv)
        assert test.var == 3

    def test_stacked_hooks(self):
        @fixtures.custom_prefix('unittests/resources/checks')
        class MyTest(HelloTest):
            def __init__(self):
                super().__init__()
                self.name = type(self).__name__
                self.executable = os.path.join('.', self.name)
                self.var = 0

            @rfm.run_before('setup')
            @rfm.run_after('setup')
            @rfm.run_after('compile')
            def x(self):
                self.var += 1

        test = MyTest()
        _run(test, self.partition, self.prgenv)
        assert test.var == 3

    def test_inherited_hooks(self):
        @fixtures.custom_prefix('unittests/resources/checks')
        class BaseTest(HelloTest):
            def __init__(self):
                super().__init__()
                self.name = type(self).__name__
                self.executable = os.path.join('.', self.name)
                self.var = 0

            @rfm.run_after('setup')
            def x(self):
                self.var += 1

        class C(rfm.RegressionTest):
            @rfm.run_before('run')
            def y(self):
                self.foo = 1

        class DerivedTest(BaseTest, C):
            @rfm.run_after('setup')
            def z(self):
                self.var += 1

        class MyTest(DerivedTest):
            pass

        test = MyTest()
        _run(test, self.partition, self.prgenv)
        assert test.var == 2
        assert test.foo == 1

    def test_overriden_hooks(self):
        @fixtures.custom_prefix('unittests/resources/checks')
        class BaseTest(HelloTest):
            def __init__(self):
                super().__init__()
                self.name = type(self).__name__
                self.executable = os.path.join('.', self.name)
                self.var = 0
                self.foo = 0

            @rfm.run_after('setup')
            def x(self):
                self.var += 1

            @rfm.run_before('setup')
            def y(self):
                self.foo += 1

        class DerivedTest(BaseTest):
            @rfm.run_after('setup')
            def x(self):
                self.var += 5

        class MyTest(DerivedTest):
            @rfm.run_before('setup')
            def y(self):
                self.foo += 10

        test = MyTest()
        _run(test, self.partition, self.prgenv)
        assert test.var == 5
        assert test.foo == 10

    def test_require_deps(self):
        import reframe.frontend.dependency as dependency
        import reframe.frontend.executors as executors

        @fixtures.custom_prefix('unittests/resources/checks')
        class T0(HelloTest):
            def __init__(self):
                super().__init__()
                self.name = type(self).__name__
                self.executable = os.path.join('.', self.name)
                self.x = 1

        @fixtures.custom_prefix('unittests/resources/checks')
        class T1(HelloTest):
            def __init__(self):
                super().__init__()
                self.name = type(self).__name__
                self.executable = os.path.join('.', self.name)
                self.depends_on('T0')

            @rfm.require_deps
            def sety(self, T0):
                self.y = T0().x + 1

            @rfm.run_before('run')
            @rfm.require_deps
            def setz(self, T0):
                self.z = T0().x + 2

        cases = executors.generate_testcases([T0(), T1()])
        deps = dependency.build_deps(cases)
        for c in dependency.toposort(deps):
            _run(*c)

        for c in cases:
            t = c.check
            if t.name == 'T0':
                assert t.x == 1
            elif t.name == 'T1':
                assert t.y == 2
                assert t.z == 3


class TestSyntax(unittest.TestCase):
    def test_regression_test(self):
        class MyTest(rfm.RegressionTest):
            def __init__(self, a, b):
                self.a = a
                self.b = b

        test = MyTest(1, 2)
        assert os.path.abspath(os.path.dirname(__file__)) == test.prefix
        assert ('TestSyntax.test_regression_test.<locals>.MyTest_1_2' ==
                test.name)

    def test_regression_test_strange_names(self):
        class C:
            def __init__(self, a):
                self.a = a

            def __repr__(self):
                return 'C(%s)' % self.a

        class MyTest(rfm.RegressionTest):
            def __init__(self, a, b):
                self.a = a
                self.b = b

        test = MyTest('(a*b+c)/12', C(33))
        assert ('TestSyntax.test_regression_test_strange_names.'
                '<locals>.MyTest__a_b_c__12_C_33_' == test.name)

    def test_user_inheritance(self):
        class MyBaseTest(rfm.RegressionTest):
            def __init__(self, a, b):
                self.a = a
                self.b = b

        class MyTest(MyBaseTest):
            def __init__(self):
                super().__init__(1, 2)

        test = MyTest()
        assert 'TestSyntax.test_user_inheritance.<locals>.MyTest' == test.name

    def test_runonly_test(self):
        class MyTest(rfm.RunOnlyRegressionTest):
            def __init__(self, a, b):
                self.a = a
                self.b = b

        test = MyTest(1, 2)
        assert os.path.abspath(os.path.dirname(__file__)) == test.prefix
        assert 'TestSyntax.test_runonly_test.<locals>.MyTest_1_2' == test.name

    def test_compileonly_test(self):
        class MyTest(rfm.CompileOnlyRegressionTest):
            def __init__(self, a, b):
                self.a = a
                self.b = b

        test = MyTest(1, 2)
        assert os.path.abspath(os.path.dirname(__file__)) == test.prefix
        assert ('TestSyntax.test_compileonly_test.<locals>.MyTest_1_2' ==
                test.name)

    def test_registration(self):
        import sys
        import unittests.resources.checks_unlisted.good as mod
        checks = mod._rfm_gettests()
        assert 13 == len(checks)
        assert [mod.MyBaseTest(0, 0),
                mod.MyBaseTest(0, 1),
                mod.MyBaseTest(1, 0),
                mod.MyBaseTest(1, 1),
                mod.MyBaseTest(2, 0),
                mod.MyBaseTest(2, 1),
                mod.AnotherBaseTest(0, 0),
                mod.AnotherBaseTest(0, 1),
                mod.AnotherBaseTest(1, 0),
                mod.AnotherBaseTest(1, 1),
                mod.AnotherBaseTest(2, 0),
                mod.AnotherBaseTest(2, 1),
                mod.MyBaseTest(10, 20)] == checks


class TestSanityPatterns(unittest.TestCase):
    @rt.switch_runtime(fixtures.TEST_SITE_CONFIG, 'testsys')
    def setUp(self):
        # Set up the test runtime
        self.resourcesdir = tempfile.mkdtemp(dir='unittests')
        rt.runtime().resources.prefix = self.resourcesdir

        # Set up regression test
        @fixtures.custom_prefix('unittests/resources/checks')
        class MyTest(rfm.RegressionTest):
            pass

        self.partition = rt.runtime().system.partition('gpu')
        self.prgenv = self.partition.environment('builtin-gcc')

        self.test = MyTest()
        self.test.setup(self.partition, self.prgenv)
        self.test.reference = {
            'testsys': {
                'value1': (1.4, -0.1, 0.1, None),
                'value2': (1.7, -0.1, 0.1, None),
            },
            'testsys:gpu': {
                'value3': (3.1, -0.1, 0.1, None),
            }
        }

        self.perf_file = tempfile.NamedTemporaryFile(mode='wt', delete=False)
        self.output_file = tempfile.NamedTemporaryFile(mode='wt', delete=False)
        self.test.perf_patterns = {
            'value1': sn.extractsingle(r'performance1 = (\S+)',
                                       self.perf_file.name, 1, float),
            'value2': sn.extractsingle(r'performance2 = (\S+)',
                                       self.perf_file.name, 1, float),
            'value3': sn.extractsingle(r'performance3 = (\S+)',
                                       self.perf_file.name, 1, float)
        }
        self.test.sanity_patterns = sn.assert_found(r'result = success',
                                                    self.output_file.name)

    def tearDown(self):
        self.perf_file.close()
        self.output_file.close()
        os.remove(self.perf_file.name)
        os.remove(self.output_file.name)
        os_ext.rmtree(self.resourcesdir)

    def write_performance_output(self, fp=None, **kwargs):
        if not fp:
            fp = self.perf_file

        for k, v in kwargs.items():
            fp.write('%s = %s\n' % (k, v))

        fp.close()

    def test_success(self):
        self.write_performance_output(performance1=1.3,
                                      performance2=1.8,
                                      performance3=3.3)
        self.output_file.write('result = success\n')
        self.output_file.close()
        self.test.check_sanity()
        self.test.check_performance()

    def test_sanity_failure(self):
        self.output_file.write('result = failure\n')
        self.output_file.close()
        with pytest.raises(SanityError):
            self.test.check_sanity()

    def test_sanity_failure_noassert(self):
        self.test.sanity_patterns = sn.findall(r'result = success',
                                               self.output_file.name)
        self.output_file.write('result = failure\n')
        self.output_file.close()
        with pytest.raises(SanityError):
            self.test.check_sanity()

    def test_sanity_multiple_patterns(self):
        self.output_file.write('result1 = success\n')
        self.output_file.write('result2 = success\n')
        self.output_file.close()

        # Simulate a pure sanity test; invalidate the reference values
        self.test.reference = {}
        self.test.sanity_patterns = sn.assert_eq(
            sn.count(sn.findall(r'result\d = success', self.output_file.name)),
            2)
        self.test.check_sanity()

        # Require more patterns to be present
        self.test.sanity_patterns = sn.assert_eq(
            sn.count(sn.findall(r'result\d = success', self.output_file.name)),
            3)
        with pytest.raises(SanityError):
            self.test.check_sanity()

    def test_sanity_multiple_files(self):
        files = [tempfile.NamedTemporaryFile(mode='wt', prefix='regtmp',
                                             dir=self.test.stagedir,
                                             delete=False)
                 for i in range(2)]

        for f in files:
            f.write('result = success\n')
            f.close()

        self.test.sanity_patterns = sn.all([
            sn.assert_found(r'result = success', files[0].name),
            sn.assert_found(r'result = success', files[1].name)
        ])
        self.test.check_sanity()
        for f in files:
            os.remove(f.name)

    def test_performance_failure(self):
        self.write_performance_output(performance1=1.0,
                                      performance2=1.8,
                                      performance3=3.3)
        self.output_file.write('result = success\n')
        self.output_file.close()
        self.test.check_sanity()
        with pytest.raises(PerformanceError):
            self.test.check_performance()

    def test_performance_no_units(self):
        with pytest.raises(TypeError):
            self.test.reference = {
                'testsys': {
                    'value1': (1.4, -0.1, 0.1),
                }
            }

    def test_unknown_tag(self):
        self.test.reference = {
            'testsys': {
                'value1': (1.4, -0.1, 0.1, None),
                'value2': (1.7, -0.1, 0.1, None),
                'foo': (3.1, -0.1, 0.1, None),
            }
        }

        self.write_performance_output(performance1=1.3,
                                      performance2=1.8,
                                      performance3=3.3)
        with pytest.raises(SanityError):
            self.test.check_performance()

    def test_unknown_system(self):
        self.write_performance_output(performance1=1.3,
                                      performance2=1.8,
                                      performance3=3.3)
        self.test.reference = {
            'testsys:login': {
                'value1': (1.4, -0.1, 0.1, None),
                'value3': (3.1, -0.1, 0.1, None),
            },
            'testsys:login2': {
                'value2': (1.7, -0.1, 0.1, None)
            }
        }
        self.test.check_performance()

    def test_empty_reference(self):
        self.write_performance_output(performance1=1.3,
                                      performance2=1.8,
                                      performance3=3.3)
        self.test.reference = {}
        self.test.check_performance()

    def test_default_reference(self):
        self.write_performance_output(performance1=1.3,
                                      performance2=1.8,
                                      performance3=3.3)
        self.test.reference = {
            '*': {
                'value1': (1.4, -0.1, 0.1, None),
                'value2': (1.7, -0.1, 0.1, None),
                'value3': (3.1, -0.1, 0.1, None),
            }
        }

        self.test.check_performance()

    def test_tag_resolution(self):
        self.write_performance_output(performance1=1.3,
                                      performance2=1.8,
                                      performance3=3.3)
        self.test.reference = {
            'testsys': {
                'value1': (1.4, -0.1, 0.1, None),
                'value2': (1.7, -0.1, 0.1, None),
            },
            '*': {
                'value3': (3.1, -0.1, 0.1, None),
            }
        }
        self.test.check_performance()

    def test_invalid_perf_value(self):
        self.test.perf_patterns = {
            'value1': sn.extractsingle(r'performance1 = (\S+)',
                                       self.perf_file.name, 1, float),
            'value2': sn.extractsingle(r'performance2 = (\S+)',
                                       self.perf_file.name, 1, str),
            'value3': sn.extractsingle(r'performance3 = (\S+)',
                                       self.perf_file.name, 1, float)
        }
        self.write_performance_output(performance1=1.3,
                                      performance2='foo',
                                      performance3=3.3)
        with pytest.raises(SanityError, match='not a number'):
            self.test.check_performance()

    def test_perf_var_evaluation(self):
        # All performance values must be evaluated, despite the first one
        # failing To test this, we need an extract function that will have a
        # side effect when evaluated, whose result we will check after calling
        # `check_performance()`.
        logfile = 'perf.log'

        @sn.sanity_function
        def extract_perf(patt, tag):
            val = sn.evaluate(
                sn.extractsingle(patt, self.perf_file.name, tag, float))

            with open('perf.log', 'a') as fp:
                fp.write('%s=%s' % (tag, val))

            return val

        self.test.perf_patterns = {
            'value1': extract_perf(r'performance1 = (?P<v1>\S+)', 'v1'),
            'value2': extract_perf(r'performance2 = (?P<v2>\S+)', 'v2'),
            'value3': extract_perf(r'performance3 = (?P<v3>\S+)', 'v3')
        }
        self.write_performance_output(performance1=1.0,
                                      performance2=1.8,
                                      performance3=3.3)
        with pytest.raises(PerformanceError) as cm:
            self.test.check_performance()

        logfile = os.path.join(self.test.stagedir, logfile)
        with open(logfile) as fp:
            log_output = fp.read()

        assert 'v1' in log_output
        assert 'v2' in log_output
        assert 'v3' in log_output


class TestRegressionTestWithContainer(unittest.TestCase):
    def temp_prefix(self):
        # Set runtime prefix
        rt.runtime().resources.prefix = tempfile.mkdtemp(dir='unittests')

    def create_test(self, platform, image):
        @fixtures.custom_prefix('unittests/resources/checks')
        class ContainerTest(rfm.RunOnlyRegressionTest):
            def __init__(self, platform):
                self.valid_prog_environs = ['*']
                self.valid_systems = ['*']
                self.container_platform = platform
                self.container_platform.image = image
                self.container_platform.commands = [
                    'pwd', 'ls', 'cat /etc/os-release'
                ]
                self.container_platform.workdir = '/workdir'
                self.sanity_patterns = sn.all([
                    sn.assert_found(
                        r'^' + self.container_platform.workdir, self.stdout),
                    sn.assert_found(r'^hello.c', self.stdout),
                    sn.assert_found(
                        r'18\.04\.\d+ LTS \(Bionic Beaver\)', self.stdout),
                ])

        test = ContainerTest(platform)
        return test

    def _skip_if_not_configured(self, partition, platform):
        if platform not in partition.container_environs.keys():
            pytest.skip('%s is not configured on the system' % platform)

    @fixtures.switch_to_user_runtime
    def test_singularity(self):
        cle_version = _cray_cle_version()
        if cle_version is not None and cle_version.startswith('6.0'):
            pytest.skip('test not supported on Cray CLE6')

        partition, environ = _setup_remote_execution()
        self._skip_if_not_configured(partition, 'Singularity')
        with tempfile.TemporaryDirectory(dir='unittests') as dirname:
            rt.runtime().resources.prefix = dirname
            _run(self.create_test('Singularity', 'docker://ubuntu:18.04'),
                 partition, environ)

    @fixtures.switch_to_user_runtime
    def test_docker(self):
        partition, environ = _setup_remote_execution('local')
        self._skip_if_not_configured(partition, 'Docker')
        with tempfile.TemporaryDirectory(dir='unittests') as dirname:
            rt.runtime().resources.prefix = dirname
            _run(self.create_test('Docker', 'ubuntu:18.04'),
                 partition, environ)

    @fixtures.switch_to_user_runtime
    def test_shifter(self):
        partition, environ = _setup_remote_execution()
        self._skip_if_not_configured(partition, 'ShifterNG')
        with tempfile.TemporaryDirectory(dir='unittests') as dirname:
            rt.runtime().resources.prefix = dirname
            _run(self.create_test('ShifterNG', 'ubuntu:18.04'),
                 partition, environ)

    @fixtures.switch_to_user_runtime
    def test_sarus(self):
        partition, environ = _setup_remote_execution()
        self._skip_if_not_configured(partition, 'Sarus')
        with tempfile.TemporaryDirectory(dir='unittests') as dirname:
            rt.runtime().resources.prefix = dirname
            _run(self.create_test('Sarus', 'ubuntu:18.04'),
                 partition, environ)

    def test_unknown_platform(self):
        partition, environ = _setup_local_execution()
        with pytest.raises(ValueError):
            with tempfile.TemporaryDirectory(dir='unittests') as dirname:
                rt.runtime().resources.prefix = dirname
                _run(self.create_test('foo', 'ubuntu:18.04'),
                     partition, environ)

    def test_not_configured_platform(self):
        partition, environ = _setup_local_execution()
        platform = None
        for cp in ['Docker', 'Singularity', 'Sarus', 'ShifterNG']:
            if cp not in partition.container_environs.keys():
                platform = cp
                break

        if platform is None:
            pytest.skip('cannot find a not configured supported platform')

        with pytest.raises(PipelineError):
            with tempfile.TemporaryDirectory(dir='unittests') as dirname:
                rt.runtime().resources.prefix = dirname
                _run(self.create_test(platform, 'ubuntu:18.04'),
                     partition, environ)
