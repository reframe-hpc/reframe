# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import pathlib
import pytest
import re

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


def _run(test, partition, prgenv):
    test.setup(partition, prgenv)
    test.compile()
    test.compile_wait()
    test.run()
    test.wait()
    test.check_sanity()
    test.check_performance()
    test.cleanup(remove_files=True)


def load_test(testfile):
    loader = RegressionCheckLoader(['unittests/resources/checks'])
    return loader.load_from_file(testfile)


@pytest.fixture
def temp_runtime(tmp_path):
    def _temp_runtime(config_file, system=None, options={}):
        options.update({'systems/prefix': str(tmp_path)})
        with rt.temp_runtime(config_file, system, options):
            yield rt.runtime()

    yield _temp_runtime


@pytest.fixture
def generic_system(temp_runtime):
    yield from temp_runtime(fixtures.TEST_CONFIG_FILE, 'generic')


@pytest.fixture
def testsys_system(temp_runtime):
    yield from temp_runtime(fixtures.TEST_CONFIG_FILE, 'testsys')


@pytest.fixture
def user_system(temp_runtime):
    if fixtures.USER_CONFIG_FILE:
        yield from temp_runtime(fixtures.USER_CONFIG_FILE,
                                fixtures.USER_SYSTEM)
    else:
        yield generic_system


@pytest.fixture
def hellotest():
    yield load_test('unittests/resources/checks/hellocheck.py')[0]


@pytest.fixture
def local_exec_ctx(generic_system):
    partition = fixtures.partition_by_name('default')
    environ = fixtures.environment_by_name('builtin-gcc', partition)
    yield partition, environ


@pytest.fixture
def local_user_exec_ctx(user_system):
    partition = fixtures.partition_by_scheduler('local')
    if partition is None:
        pytest.skip('no local jobs are supported')

    try:
        environ = partition.environs[0]
    except IndexError:
        pytest.skip('no environments configured for partition: %s' %
                    partition.fullname)

    yield partition, environ


@pytest.fixture
def remote_exec_ctx(user_system):
    partition = fixtures.partition_by_scheduler()
    if partition is None:
        pytest.skip('job submission not supported')

    try:
        environ = partition.environs[0]
    except IndexError:
        pytest.skip('no environments configured for partition: %s' %
                    partition.fullname)

    yield partition, environ


@pytest.fixture
def remote_exec_ctx(user_system):
    partition = fixtures.partition_by_scheduler()
    if partition is None:
        pytest.skip('job submission not supported')

    try:
        environ = partition.environs[0]
    except IndexError:
        pytest.skip('no environments configured for partition: %s' %
                    partition.fullname)

    yield partition, environ


@pytest.fixture
def container_remote_exec_ctx(remote_exec_ctx):
    def _container_exec_ctx(platform):
        partition = remote_exec_ctx[0]
        if platform not in partition.container_environs.keys():
            pytest.skip('%s is not configured on the system' % platform)

        yield from remote_exec_ctx

    return _container_exec_ctx


@pytest.fixture
def container_local_exec_ctx(local_user_exec_ctx):
    def _container_exec_ctx(platform):
        partition = local_user_exec_ctx[0]
        if platform not in partition.container_environs.keys():
            pytest.skip('%s is not configured on the system' % platform)

        yield from local_user_exec_ctx

    return _container_exec_ctx


def test_environ_setup(hellotest, local_exec_ctx):
    # Use test environment for the regression check
    hellotest.variables = {'_FOO_': '1', '_BAR_': '2'}
    hellotest.setup(*local_exec_ctx)
    for k in hellotest.variables.keys():
        assert k not in os.environ


def test_hellocheck(hellotest, remote_exec_ctx):
    _run(hellotest, *remote_exec_ctx)


def test_hellocheck_make(remote_exec_ctx):
    test = load_test('unittests/resources/checks/hellocheck_make.py')[0]
    _run(test, *remote_exec_ctx)


def test_hellocheck_local(hellotest, local_exec_ctx):
    # Test also the prebuild/postbuild functionality
    hellotest.prebuild_cmd = ['touch prebuild', 'mkdir prebuild_dir']
    hellotest.postbuild_cmd = ['touch postbuild', 'mkdir postbuild_dir']
    hellotest.keep_files = ['prebuild', 'postbuild',
                            'prebuild_dir', 'postbuild_dir']

    # Force local execution of the test; just for testing .local
    hellotest.local = True
    _run(hellotest, *local_exec_ctx)
    must_keep = [
        hellotest.stdout.evaluate(),
        hellotest.stderr.evaluate(),
        hellotest.build_stdout.evaluate(),
        hellotest.build_stderr.evaluate(),
        hellotest.job.script_filename,
        *hellotest.keep_files
    ]
    for f in must_keep:
        assert os.path.exists(os.path.join(hellotest.outputdir, f))


def test_hellocheck_local_prepost_run(hellotest, local_exec_ctx):
    @sn.sanity_function
    def stagedir(test):
        return test.stagedir

    # Test also the prebuild/postbuild functionality
    hellotest.pre_run = ['echo prerun: `pwd`']
    hellotest.post_run = ['echo postrun: `pwd`']
    pre_run_path = sn.extractsingle(r'^prerun: (\S+)', hellotest.stdout, 1)
    post_run_path = sn.extractsingle(r'^postrun: (\S+)', hellotest.stdout, 1)
    hellotest.sanity_patterns = sn.all([
        sn.assert_eq(stagedir(hellotest), pre_run_path),
        sn.assert_eq(stagedir(hellotest), post_run_path),
    ])
    _run(hellotest, *local_exec_ctx)


def test_run_only_sanity(local_exec_ctx):
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

    _run(MyTest(), *local_exec_ctx)


def test_run_only_no_srcdir(local_exec_ctx):
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
    _run(test, *local_exec_ctx)


def test_compile_only_failure(local_exec_ctx):
    @fixtures.custom_prefix('unittests/resources/checks')
    class MyTest(rfm.CompileOnlyRegressionTest):
        def __init__(self):
            self.sourcepath = 'compiler_failure.c'
            self.valid_prog_environs = ['*']
            self.valid_systems = ['*']

    test = MyTest()
    test.setup(*local_exec_ctx)
    test.compile()
    with pytest.raises(BuildError):
        test.compile_wait()


def test_compile_only_warning(local_exec_ctx):
    @fixtures.custom_prefix('unittests/resources/checks')
    class MyTest(rfm.RunOnlyRegressionTest):
        def __init__(self):
            self.build_system = 'SingleSource'
            self.build_system.srcfile = 'compiler_warning.c'
            self.build_system.cflags = ['-Wall']
            self.valid_prog_environs = ['*']
            self.valid_systems = ['*']
            self.sanity_patterns = sn.assert_found(r'warning', self.stderr)

    _run(MyTest(), *local_exec_ctx)


def test_supports_system(hellotest, testsys_system):
    hellotest.valid_systems = ['*']
    assert hellotest.supports_system('gpu')
    assert hellotest.supports_system('login')
    assert hellotest.supports_system('testsys:gpu')
    assert hellotest.supports_system('testsys:login')

    hellotest.valid_systems = ['*:*']
    assert hellotest.supports_system('gpu')
    assert hellotest.supports_system('login')
    assert hellotest.supports_system('testsys:gpu')
    assert hellotest.supports_system('testsys:login')

    hellotest.valid_systems = ['testsys']
    assert hellotest.supports_system('gpu')
    assert hellotest.supports_system('login')
    assert hellotest.supports_system('testsys:gpu')
    assert hellotest.supports_system('testsys:login')

    hellotest.valid_systems = ['testsys:gpu']
    assert hellotest.supports_system('gpu')
    assert not hellotest.supports_system('login')
    assert hellotest.supports_system('testsys:gpu')
    assert not hellotest.supports_system('testsys:login')

    hellotest.valid_systems = ['testsys:login']
    assert not hellotest.supports_system('gpu')
    assert hellotest.supports_system('login')
    assert not hellotest.supports_system('testsys:gpu')
    assert hellotest.supports_system('testsys:login')

    hellotest.valid_systems = ['foo']
    assert not hellotest.supports_system('gpu')
    assert not hellotest.supports_system('login')
    assert not hellotest.supports_system('testsys:gpu')
    assert not hellotest.supports_system('testsys:login')

    hellotest.valid_systems = ['*:gpu']
    assert hellotest.supports_system('testsys:gpu')
    assert hellotest.supports_system('foo:gpu')
    assert not hellotest.supports_system('testsys:cpu')
    assert not hellotest.supports_system('testsys:login')

    hellotest.valid_systems = ['testsys:*']
    assert hellotest.supports_system('testsys:login')
    assert hellotest.supports_system('gpu')
    assert not hellotest.supports_system('foo:gpu')


def test_supports_environ(hellotest, generic_system):
    hellotest.valid_prog_environs = ['*']
    assert hellotest.supports_environ('foo1')
    assert hellotest.supports_environ('foo-env')
    assert hellotest.supports_environ('*')


def test_sourcesdir_none(local_exec_ctx):
    @fixtures.custom_prefix('unittests/resources/checks')
    class MyTest(rfm.RegressionTest):
        def __init__(self):
            self.sourcesdir = None
            self.valid_prog_environs = ['*']
            self.valid_systems = ['*']

    with pytest.raises(ReframeError):
        _run(MyTest(), *local_exec_ctx)


def test_sourcesdir_build_system(local_exec_ctx):
    @fixtures.custom_prefix('unittests/resources/checks')
    class MyTest(rfm.RegressionTest):
        def __init__(self):
            self.build_system = 'Make'
            self.sourcepath = 'code'
            self.executable = './code/hello'
            self.valid_systems = ['*']
            self.valid_prog_environs = ['*']
            self.sanity_patterns = sn.assert_found(r'Hello, World\!',
                                                   self.stdout)

    _run(MyTest(), *local_exec_ctx)


def test_sourcesdir_none_generated_sources(local_exec_ctx):
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
            self.valid_systems = ['*']
            self.valid_prog_environs = ['*']
            self.sanity_patterns = sn.assert_found(r'Hello, World\!',
                                                   self.stdout)

    _run(MyTest(), *local_exec_ctx)


def test_sourcesdir_none_compile_only(local_exec_ctx):
    @fixtures.custom_prefix('unittests/resources/checks')
    class MyTest(rfm.CompileOnlyRegressionTest):
        def __init__(self):
            self.sourcesdir = None
            self.valid_prog_environs = ['*']
            self.valid_systems = ['*']

    with pytest.raises(BuildError):
        _run(MyTest(), *local_exec_ctx)


def test_sourcesdir_none_run_only(local_exec_ctx):
    @fixtures.custom_prefix('unittests/resources/checks')
    class MyTest(rfm.RunOnlyRegressionTest):
        def __init__(self):
            self.sourcesdir = None
            self.executable = 'echo'
            self.executable_opts = ["Hello, World!"]
            self.valid_prog_environs = ['*']
            self.valid_systems = ['*']
            self.sanity_patterns = sn.assert_found(r'Hello, World\!',
                                                   self.stdout)

    _run(MyTest(), *local_exec_ctx)


def test_sourcepath_abs(local_exec_ctx):
    @fixtures.custom_prefix('unittests/resources/checks')
    class MyTest(rfm.CompileOnlyRegressionTest):
        def __init__(self):
            self.valid_prog_environs = ['*']
            self.valid_systems = ['*']

    test = MyTest()
    test.setup(*local_exec_ctx)
    test.sourcepath = '/usr/src'
    with pytest.raises(PipelineError):
        test.compile()


def test_sourcepath_upref(local_exec_ctx):
    @fixtures.custom_prefix('unittests/resources/checks')
    class MyTest(rfm.CompileOnlyRegressionTest):
        def __init__(self):
            self.valid_prog_environs = ['*']
            self.valid_systems = ['*']

    test = MyTest()
    test.setup(*local_exec_ctx)
    test.sourcepath = '../hellosrc'
    with pytest.raises(PipelineError):
        test.compile()


def test_extra_resources(testsys_system):
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
    partition = fixtures.partition_by_name('gpu')
    environ = partition.environment('builtin-gcc')
    _run(test, partition, environ)
    expected_job_options = {'--gres=gpu:2',
                            '#DW jobdw capacity=100GB',
                            '#DW stage_in source=%s' % test.stagedir,
                            '--foo'}
    assert expected_job_options == set(test.job.options)


def test_setup_hooks(local_exec_ctx):
    @fixtures.custom_prefix('unittests/resources/checks')
    class MyTest(HelloTest):
        def __init__(self):
            super().__init__()
            self.name = type(self).__name__
            self.executable = os.path.join('.', self.name)
            self.count = 0

        @rfm.run_before('setup')
        def prefoo(self):
            assert self.current_environ is None
            self.count += 1

        @rfm.run_after('setup')
        def postfoo(self):
            assert self.current_environ is not None
            self.count += 1

    test = MyTest()
    _run(test, *local_exec_ctx)
    assert test.count == 2


def test_compile_hooks(local_exec_ctx):
    @fixtures.custom_prefix('unittests/resources/checks')
    class MyTest(HelloTest):
        def __init__(self):
            super().__init__()
            self.name = type(self).__name__
            self.executable = os.path.join('.', self.name)
            self.count = 0

        @rfm.run_before('compile')
        def setflags(self):
            self.count += 1

        @rfm.run_after('compile')
        def check_executable(self):
            exec_file = os.path.join(self.stagedir, self.executable)

            # Make sure that this hook is executed after compile_wait()
            assert os.path.exists(exec_file)

    test = MyTest()
    _run(test, *local_exec_ctx)
    assert test.count == 1


def test_run_hooks(local_exec_ctx):
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

    _run(MyTest(), *local_exec_ctx)


def test_multiple_hooks(local_exec_ctx):
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
    _run(test, *local_exec_ctx)
    assert test.var == 3


def test_stacked_hooks(local_exec_ctx):
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
    _run(test, *local_exec_ctx)
    assert test.var == 3


def test_inherited_hooks(local_exec_ctx):
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
    _run(test, *local_exec_ctx)
    assert test.var == 2
    assert test.foo == 1


def test_overriden_hooks(local_exec_ctx):
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
    _run(test, *local_exec_ctx)
    assert test.var == 5
    assert test.foo == 10


def test_require_deps(local_exec_ctx):
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


def test_regression_test_name():
    class MyTest(rfm.RegressionTest):
        def __init__(self, a, b):
            self.a = a
            self.b = b

    test = MyTest(1, 2)
    assert os.path.abspath(os.path.dirname(__file__)) == test.prefix
    assert 'test_regression_test_name.<locals>.MyTest_1_2' == test.name


def test_strange_test_names():
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
    assert ('test_strange_test_names.<locals>.MyTest__a_b_c__12_C_33_' ==
            test.name)


def test_name_user_inheritance():
    class MyBaseTest(rfm.RegressionTest):
        def __init__(self, a, b):
            self.a = a
            self.b = b

    class MyTest(MyBaseTest):
        def __init__(self):
            super().__init__(1, 2)

    test = MyTest()
    assert 'test_name_user_inheritance.<locals>.MyTest' == test.name


def test_name_runonly_test():
    class MyTest(rfm.RunOnlyRegressionTest):
        def __init__(self, a, b):
            self.a = a
            self.b = b

    test = MyTest(1, 2)
    assert os.path.abspath(os.path.dirname(__file__)) == test.prefix
    assert 'test_name_runonly_test.<locals>.MyTest_1_2' == test.name


def test_name_compileonly_test():
    class MyTest(rfm.CompileOnlyRegressionTest):
        def __init__(self, a, b):
            self.a = a
            self.b = b

    test = MyTest(1, 2)
    assert os.path.abspath(os.path.dirname(__file__)) == test.prefix
    assert 'test_name_compileonly_test.<locals>.MyTest_1_2' == test.name


def test_registration_of_tests():
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


def _run_sanity(test, *exec_ctx, skip_perf=False):
    test.setup(*exec_ctx)
    test.check_sanity()
    if not skip_perf:
        test.check_performance()


@pytest.fixture
def dummy_gpu_exec_ctx(testsys_system):
    partition = fixtures.partition_by_name('gpu')
    environ = fixtures.environment_by_name('builtin-gcc', partition)
    yield partition, environ


@pytest.fixture
def perf_file(tmp_path):
    yield tmp_path / 'perf.out'


@pytest.fixture
def sanity_file(tmp_path):
    yield tmp_path / 'sanity.out'


@pytest.fixture
def dummytest(testsys_system, perf_file, sanity_file):
    class MyTest(rfm.RunOnlyRegressionTest):
        def __init__(self):
            self.perf_file = perf_file
            self.sourcesdir = None
            self.reference = {
                'testsys': {
                    'value1': (1.4, -0.1, 0.1, None),
                    'value2': (1.7, -0.1, 0.1, None),
                },
                'testsys:gpu': {
                    'value3': (3.1, -0.1, 0.1, None),
                }
            }
            self.perf_patterns = {
                'value1': sn.extractsingle(
                    r'perf1 = (\S+)', perf_file, 1, float
                ),
                'value2': sn.extractsingle(
                    r'perf2 = (\S+)', perf_file, 1, float
                ),
                'value3': sn.extractsingle(
                    r'perf3 = (\S+)', perf_file, 1, float
                )
            }
            self.sanity_patterns = sn.assert_found(
                r'result = success', sanity_file
            )

    yield MyTest()


def test_sanity_success(dummytest, sanity_file, perf_file, dummy_gpu_exec_ctx):
    sanity_file.write_text('result = success\n')
    perf_file.write_text('perf1 = 1.3\n'
                         'perf2 = 1.8\n'
                         'perf3 = 3.3\n')
    _run_sanity(dummytest, *dummy_gpu_exec_ctx)


def test_sanity_failure(dummytest, sanity_file, dummy_gpu_exec_ctx):
    sanity_file.write_text('result = failure\n')
    with pytest.raises(SanityError):
        _run_sanity(dummytest, *dummy_gpu_exec_ctx, skip_perf=True)


def test_sanity_failure_noassert(dummytest, sanity_file, dummy_gpu_exec_ctx):
    dummytest.sanity_patterns = sn.findall(r'result = success', sanity_file)
    sanity_file.write_text('result = failure\n')
    with pytest.raises(SanityError):
        _run_sanity(dummytest, *dummy_gpu_exec_ctx, skip_perf=True)


def test_sanity_multiple_patterns(dummytest, sanity_file, dummy_gpu_exec_ctx):
    sanity_file.write_text('result1 = success\n'
                           'result2 = success\n')

    # Simulate a pure sanity test; reset the perf_patterns
    dummytest.perf_patterns = None
    dummytest.sanity_patterns = sn.assert_eq(
        sn.count(sn.findall(r'result\d = success', sanity_file)), 2
    )
    _run_sanity(dummytest, *dummy_gpu_exec_ctx, skip_perf=True)

    # Require more patterns to be present
    dummytest.sanity_patterns = sn.assert_eq(
        sn.count(sn.findall(r'result\d = success', sanity_file)), 3
    )
    with pytest.raises(SanityError):
        _run_sanity(dummytest, *dummy_gpu_exec_ctx, skip_perf=True)


def test_sanity_multiple_files(dummytest, tmp_path, dummy_gpu_exec_ctx):
    file0 = tmp_path / 'out1.txt'
    file1 = tmp_path / 'out2.txt'
    file0.write_text('result = success\n')
    file1.write_text('result = success\n')
    dummytest.sanity_patterns = sn.all([
        sn.assert_found(r'result = success', file0),
        sn.assert_found(r'result = success', file1)
    ])
    _run_sanity(dummytest, *dummy_gpu_exec_ctx, skip_perf=True)


def test_performance_failure(dummytest, sanity_file,
                             perf_file, dummy_gpu_exec_ctx):
    sanity_file.write_text('result = success\n')
    perf_file.write_text('perf1 = 1.0\n'
                         'perf2 = 1.8\n'
                         'perf3 = 3.3\n')
    with pytest.raises(PerformanceError):
        _run_sanity(dummytest, *dummy_gpu_exec_ctx)


def test_reference_unknown_tag(dummytest, sanity_file,
                               perf_file, dummy_gpu_exec_ctx):
    sanity_file.write_text('result = success\n')
    perf_file.write_text('perf1 = 1.3\n'
                         'perf2 = 1.8\n'
                         'perf3 = 3.3\n')
    dummytest.reference = {
        'testsys': {
            'value1': (1.4, -0.1, 0.1, None),
            'value2': (1.7, -0.1, 0.1, None),
            'foo': (3.1, -0.1, 0.1, None),
        }
    }
    with pytest.raises(SanityError):
        _run_sanity(dummytest, *dummy_gpu_exec_ctx)


def test_reference_unknown_system(dummytest, sanity_file,
                                  perf_file, dummy_gpu_exec_ctx):
    sanity_file.write_text('result = success\n')
    perf_file.write_text('perf1 = 1.3\n'
                         'perf2 = 1.8\n'
                         'perf3 = 3.3\n')
    dummytest.reference = {
        'testsys:login': {
            'value1': (1.4, -0.1, 0.1, None),
            'value3': (3.1, -0.1, 0.1, None),
        },
        'testsys:login2': {
            'value2': (1.7, -0.1, 0.1, None)
        }
    }
    _run_sanity(dummytest, *dummy_gpu_exec_ctx)


def test_reference_empty(dummytest, sanity_file,
                         perf_file, dummy_gpu_exec_ctx):
    sanity_file.write_text('result = success\n')
    perf_file.write_text('perf1 = 1.3\n'
                         'perf2 = 1.8\n'
                         'perf3 = 3.3\n')
    dummytest.reference = {}
    _run_sanity(dummytest, *dummy_gpu_exec_ctx)


def test_reference_default(dummytest, sanity_file,
                           perf_file, dummy_gpu_exec_ctx):
    sanity_file.write_text('result = success\n')
    perf_file.write_text('perf1 = 1.3\n'
                         'perf2 = 1.8\n'
                         'perf3 = 3.3\n')
    dummytest.reference = {
        '*': {
            'value1': (1.4, -0.1, 0.1, None),
            'value2': (1.7, -0.1, 0.1, None),
            'value3': (3.1, -0.1, 0.1, None),
        }
    }
    _run_sanity(dummytest, *dummy_gpu_exec_ctx)


def test_reference_tag_resolution(dummytest, sanity_file,
                                  perf_file, dummy_gpu_exec_ctx):
    sanity_file.write_text('result = success\n')
    perf_file.write_text('perf1 = 1.3\n'
                         'perf2 = 1.8\n'
                         'perf3 = 3.3\n')
    dummytest.reference = {
        'testsys': {
            'value1': (1.4, -0.1, 0.1, None),
            'value2': (1.7, -0.1, 0.1, None),
        },
        '*': {
            'value3': (3.1, -0.1, 0.1, None),
        }
    }
    _run_sanity(dummytest, *dummy_gpu_exec_ctx)


def test_performance_invalid_value(dummytest, sanity_file,
                                   perf_file, dummy_gpu_exec_ctx):
    sanity_file.write_text('result = success\n')
    perf_file.write_text('perf1 = 1.3\n'
                         'perf2 = foo\n'
                         'perf3 = 3.3\n')
    dummytest.perf_patterns = {
        'value1': sn.extractsingle(r'perf1 = (\S+)', perf_file, 1, float),
        'value2': sn.extractsingle(r'perf2 = (\S+)', perf_file, 1, str),
        'value3': sn.extractsingle(r'perf3 = (\S+)', perf_file, 1, float)
    }
    with pytest.raises(SanityError, match='not a number'):
        _run_sanity(dummytest, *dummy_gpu_exec_ctx)


def test_performance_var_evaluation(dummytest, sanity_file,
                                    perf_file, dummy_gpu_exec_ctx):
    # All performance values must be evaluated, despite the first one
    # failing To test this, we need an extract function that will have a
    # side effect when evaluated, whose result we will check after calling
    # `check_performance()`.
    logfile = 'perf.log'

    @sn.sanity_function
    def extract_perf(patt, tag):
        val = sn.evaluate(
            sn.extractsingle(patt, perf_file, tag, float)
        )
        with open('perf.log', 'a') as fp:
            fp.write('%s=%s' % (tag, val))

        return val

    sanity_file.write_text('result = success\n')
    perf_file.write_text('perf1 = 1.0\n'
                         'perf2 = 1.8\n'
                         'perf3 = 3.3\n')
    dummytest.perf_patterns = {
        'value1': extract_perf(r'perf1 = (?P<v1>\S+)', 'v1'),
        'value2': extract_perf(r'perf2 = (?P<v2>\S+)', 'v2'),
        'value3': extract_perf(r'perf3 = (?P<v3>\S+)', 'v3')
    }
    with pytest.raises(PerformanceError) as cm:
        _run_sanity(dummytest, *dummy_gpu_exec_ctx)

    logfile = os.path.join(dummytest.stagedir, logfile)
    with open(logfile) as fp:
        log_output = fp.read()

    assert 'v1' in log_output
    assert 'v2' in log_output
    assert 'v3' in log_output


@pytest.fixture
def container_test(tmp_path):
    def _container_test(platform, image):
        @fixtures.custom_prefix(tmp_path)
        class ContainerTest(rfm.RunOnlyRegressionTest):
            def __init__(self):
                self.name = 'container_test'
                self.valid_prog_environs = ['*']
                self.valid_systems = ['*']
                self.container_platform = platform
                self.container_platform.image = image
                self.container_platform.commands = [
                    'pwd', 'ls', 'cat /etc/os-release'
                ]
                self.container_platform.workdir = '/workdir'
                self.pre_run = ['touch foo']
                self.sanity_patterns = sn.all([
                    sn.assert_found(
                        r'^' + self.container_platform.workdir, self.stdout),
                    sn.assert_found(r'^foo', self.stdout),
                    sn.assert_found(
                        r'18\.04\.\d+ LTS \(Bionic Beaver\)', self.stdout),
                ])

        return ContainerTest()

    yield _container_test


def _cray_cle_version():
    completed = os_ext.run_command('cat /etc/opt/cray/release/cle-release')
    matched = re.match(r'^RELEASE=(\S+)', completed.stdout)
    if matched is None:
        return None

    return matched.group(1)


def test_with_singularity(container_test, container_remote_exec_ctx):
    cle_version = _cray_cle_version()
    if cle_version is not None and cle_version.startswith('6.0'):
        pytest.skip('test not supported on Cray CLE6')

    _run(container_test('Singularity', 'docker://ubuntu:18.04'),
         *container_remote_exec_ctx('Singularity'))


def test_with_shifter(container_test, container_remote_exec_ctx):
    _run(container_test('Shifter', 'ubuntu:18.04'),
         *container_remote_exec_ctx('Shifter'))


def test_with_sarus(container_test, container_remote_exec_ctx):
    _run(container_test('Sarus', 'ubuntu:18.04'),
         *container_remote_exec_ctx('Sarus'))


def test_with_docker(container_test, container_local_exec_ctx):
    _run(container_test('Docker', 'ubuntu:18.04'),
         *container_local_exec_ctx('Docker'))


def test_unknown_container_platform(container_test, local_exec_ctx):
    with pytest.raises(ValueError):
        _run(container_test('foo', 'ubuntu:18.04'), *local_exec_ctx)


def test_not_configured_container_platform(container_test, local_exec_ctx):
    partition, environ = local_exec_ctx
    platform = None
    for cp in ['Docker', 'Singularity', 'Sarus', 'ShifterNG']:
        if cp not in partition.container_environs.keys():
            platform = cp
            break

    if platform is None:
        pytest.skip('cannot find a supported platform that is not configured')

    with pytest.raises(PipelineError):
        _run(container_test(platform, 'ubuntu:18.04'), *local_exec_ctx)
