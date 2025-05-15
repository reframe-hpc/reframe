# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import pytest
import re
import sys

import reframe as rfm
import reframe.core.builtins as builtins
import reframe.core.runtime as rt
import reframe.utility.osext as osext
import reframe.utility.sanity as sn
import unittests.utility as test_util

from reframe.core.exceptions import (BuildError,
                                     ExpectedFailureError,
                                     PerformanceError,
                                     PipelineError,
                                     ReframeError,
                                     ReframeSyntaxError,
                                     SanityError,
                                     SkipTestError,
                                     UnexpectedSuccessError)
from reframe.core.meta import make_test
from reframe.core.warnings import ReframeDeprecationWarning


def _run(test, partition, prgenv):
    test.setup(partition, prgenv)
    test.compile()
    test.compile_wait()
    test.run()
    test.run_wait()
    test.check_sanity()
    test.check_performance()
    test.cleanup(remove_files=True)


@pytest.fixture
def HelloTest():
    from unittests.resources.checks.hellocheck import HelloTest
    yield HelloTest
    del sys.modules['unittests.resources.checks.hellocheck']


@pytest.fixture
def hellotest(HelloTest):
    yield HelloTest()


@pytest.fixture
def hellomaketest():
    from unittests.resources.checks.hellocheck_make import HelloMakeTest
    yield HelloMakeTest
    del sys.modules['unittests.resources.checks.hellocheck_make']


@pytest.fixture
def pinnedtest():
    from unittests.resources.checks.pinnedcheck import PinnedTest
    yield PinnedTest
    del sys.modules['unittests.resources.checks.pinnedcheck']


@pytest.fixture
def generic_system(make_exec_ctx_g):
    yield from make_exec_ctx_g(test_util.TEST_CONFIG_FILE, 'generic')


@pytest.fixture
def testsys_exec_ctx(make_exec_ctx_g):
    yield from make_exec_ctx_g(test_util.TEST_CONFIG_FILE, 'testsys')


@pytest.fixture
def user_system(make_exec_ctx_g):
    if test_util.USER_CONFIG_FILE:
        yield from make_exec_ctx_g(test_util.USER_CONFIG_FILE,
                                   test_util.USER_SYSTEM)
    else:
        yield generic_system


@pytest.fixture
def local_exec_ctx(generic_system):
    partition = test_util.partition_by_name('default')
    environ = test_util.environment_by_name('builtin', partition)
    yield partition, environ


@pytest.fixture
def local_user_exec_ctx(user_system):
    partition = test_util.partition_by_scheduler('local')
    if partition is None:
        pytest.skip('no local jobs are supported')

    try:
        environ = partition.environs[0]
    except IndexError:
        pytest.skip(
            f'no environments configured for partition: {partition.fullname}'
        )

    yield partition, environ


@pytest.fixture
def remote_exec_ctx(user_system):
    partition = test_util.partition_by_scheduler()
    if partition is None:
        pytest.skip('job submission not supported')

    try:
        environ = partition.environs[0]
    except IndexError:
        pytest.skip(
            f'no environments configured for partition: {partition.fullname}'
        )

    yield partition, environ


@pytest.fixture
def container_remote_exec_ctx(remote_exec_ctx):
    def _container_exec_ctx(platform):
        partition = remote_exec_ctx[0]
        if platform not in partition.container_environs.keys():
            pytest.skip(f'{platform} is not configured on the system')

        yield from remote_exec_ctx

    return _container_exec_ctx


@pytest.fixture
def container_local_exec_ctx(local_user_exec_ctx):
    def _container_exec_ctx(platform):
        partition = local_user_exec_ctx[0]
        if platform not in partition.container_environs.keys():
            pytest.skip(f'{platform} is not configured on the system')

        yield from local_user_exec_ctx

    return _container_exec_ctx


def test_eq():
    T0 = make_test('T0', (rfm.RegressionTest,), {})
    T1 = make_test('T1', (rfm.RegressionTest,), {})
    T2 = make_test('T1', (rfm.RegressionTest,), {})

    t0, t1, t2 = T0(), T1(), T2()
    assert t0 != t1
    assert hash(t0) != hash(t1)

    # T1 and T2 are different classes but have the same name, so the
    # corresponding tests should compare equal
    assert T1 is not T2
    assert t1 == t2
    assert hash(t1) == hash(t2)


def test_environ_setup(hellotest, local_exec_ctx):
    # Use test environment for the regression check
    hellotest.env_vars = {'_FOO_': 1, '_BAR_': 2}
    hellotest.setup(*local_exec_ctx)
    for k in hellotest.env_vars.keys():
        assert k not in os.environ


def test_hellocheck(hellotest, remote_exec_ctx):
    _run(hellotest, *remote_exec_ctx)


def test_hellocheck_make(hellomaketest, remote_exec_ctx):
    _run(hellomaketest(), *remote_exec_ctx)


def test_hellocheck_local(hellotest, local_exec_ctx):
    # Test also the prebuild/postbuild functionality
    hellotest.prebuild_cmds = ['touch prebuild', 'mkdir -p  prebuild_dir/foo']
    hellotest.postbuild_cmds = ['touch postbuild', 'mkdir postbuild_dir']
    hellotest.keep_files = ['prebuild', 'postbuild', '*dir']

    # Force local execution of the test; just for testing .local
    hellotest.local = True
    _run(hellotest, *local_exec_ctx)
    must_keep = [
        hellotest.stdout.evaluate(),
        hellotest.stderr.evaluate(),
        hellotest.build_stdout.evaluate(),
        hellotest.build_stderr.evaluate(),
        hellotest.job.script_filename,
        'prebuild', 'postbuild', 'prebuild_dir',
        'prebuild_dir/foo', 'postbuild_dir'
    ]
    for f in must_keep:
        assert os.path.exists(os.path.join(hellotest.outputdir, f))


def test_hellocheck_build_remotely(hellotest, remote_exec_ctx):
    hellotest.build_locally = False
    _run(hellotest, *remote_exec_ctx)
    assert not hellotest.build_job.scheduler.is_local


def test_hellocheck_local_prepost_run(HelloTest, local_exec_ctx):
    class _X(HelloTest):
        # Test also the prebuild/postbuild functionality
        prerun_cmds = ['echo prerun: `pwd`']
        postrun_cmds = ['echo postrun: `pwd`']

        @sanity_function
        def validate(self):
            pre_path  = sn.extractsingle(r'^prerun: (\S+)', self.stdout, 1)
            post_path = sn.extractsingle(r'^postrun: (\S+)', self.stdout, 1)
            return sn.all([
                sn.assert_eq(self.stagedir, pre_path),
                sn.assert_eq(self.stagedir, post_path),
            ])

    _run(_X(), *local_exec_ctx)


def test_run_only_set_sanity_in_a_hook(local_exec_ctx):
    @test_util.custom_prefix('unittests/resources/checks')
    class MyTest(rfm.RunOnlyRegressionTest):
        executable = './hello.sh'
        executable_opts = ['Hello, World!']
        local = True
        valid_prog_environs = ['*']
        valid_systems = ['*']

        @run_after('run')
        def set_sanity(self):
            self.sanity_patterns = sn.assert_found(
                r'Hello, World\!', self.stdout
            )

    _run(MyTest(), *local_exec_ctx)


def test_run_only_decorated_sanity(local_exec_ctx):
    @test_util.custom_prefix('unittests/resources/checks')
    class MyTest(rfm.RunOnlyRegressionTest):
        executable = './hello.sh'
        executable_opts = ['Hello, World!']
        local = True
        valid_prog_environs = ['*']
        valid_systems = ['*']

        @sanity_function
        def set_sanity(self):
            return sn.assert_found(r'Hello, World\!', self.stdout)

    _run(MyTest(), *local_exec_ctx)

    class MyOtherTest(MyTest):
        '''Test both syntaxes are incompatible.'''
        sanity_patterns = sn.assert_true(1)

    with pytest.raises(ReframeSyntaxError):
        _run(MyOtherTest(), *local_exec_ctx)


def test_run_only_no_srcdir(local_exec_ctx):
    @test_util.custom_prefix('foo/bar/')
    class MyTest(rfm.RunOnlyRegressionTest):
        valid_systems = ['*']
        valid_prog_environs = ['*']
        executable = 'echo'
        sanity_patterns = sn.assert_true(1)

    test = MyTest()
    assert test.sourcesdir is None
    _run(test, *local_exec_ctx)


def test_run_only_srcdir_set_to_none(local_exec_ctx):
    @test_util.custom_prefix('foo/bar/')
    class MyTest(rfm.RunOnlyRegressionTest):
        executable = 'echo'
        valid_prog_environs = ['*']
        valid_systems = ['*']
        sourcesdir = None
        sanity_patterns = sn.assert_true(1)

    test = MyTest()
    assert test.sourcesdir is None
    _run(test, *local_exec_ctx)


def test_executable_is_required(local_exec_ctx):
    class MyTest(rfm.RunOnlyRegressionTest):
        valid_prog_environs = ['*']
        valid_systems = ['*']

    with pytest.raises(AttributeError, match="'executable' has not been set"):
        _run(MyTest(), *local_exec_ctx)


def test_compile_only_failure(local_exec_ctx):
    @test_util.custom_prefix('unittests/resources/checks')
    class MyTest(rfm.CompileOnlyRegressionTest):
        sourcepath = 'compiler_failure.c'
        valid_prog_environs = ['*']
        valid_systems = ['*']

    test = MyTest()
    test.setup(*local_exec_ctx)
    test.compile()
    with pytest.raises(BuildError):
        test.compile_wait()


def test_compile_only_warning(local_exec_ctx):
    @test_util.custom_prefix('unittests/resources/checks')
    class MyTest(rfm.CompileOnlyRegressionTest):
        valid_prog_environs = ['*']
        valid_systems = ['*']
        build_system = 'SingleSource'
        sourcepath = 'compiler_warning.c'

        @run_before('compile')
        def setup_build(self):
            self.build_system.cflags = ['-Wall']

        @sanity_function
        def validate(self):
            return sn.assert_found(r'warning', self.stderr)

    _run(MyTest(), *local_exec_ctx)


def test_pinned_test(pinnedtest, local_exec_ctx):
    class MyTest(pinnedtest):
        pass

    pinned = MyTest()
    expected_prefix = os.path.join(os.getcwd(), 'unittests/resources/checks')
    assert pinned._prefix == expected_prefix


def test_valid_systems_syntax(hellotest):
    hellotest.valid_systems = ['*']
    hellotest.valid_systems = ['*:*']
    hellotest.valid_systems = ['sys:*']
    hellotest.valid_systems = ['*:part']
    hellotest.valid_systems = ['sys']
    hellotest.valid_systems = ['sys:part']
    hellotest.valid_systems = ['sys-0']
    hellotest.valid_systems = ['sys:part-0']
    hellotest.valid_systems = ['+x0']
    hellotest.valid_systems = ['-y0']
    hellotest.valid_systems = ['%z0=w0']
    hellotest.valid_systems = ['+x0 -y0 %z0=w0']
    hellotest.valid_systems = ['-y0 +x0 %z0=w0']
    hellotest.valid_systems = ['%z0=w0 +x0 -y0']

    with pytest.raises(TypeError):
        hellotest.valid_systems = ['']

    with pytest.raises(TypeError):
        hellotest.valid_systems = ['   sys:part']

    with pytest.raises(TypeError):
        hellotest.valid_systems = [' sys:part   ']

    with pytest.raises(TypeError):
        hellotest.valid_systems = [':']

    with pytest.raises(TypeError):
        hellotest.valid_systems = [':foo']

    with pytest.raises(TypeError):
        hellotest.valid_systems = ['foo:']

    with pytest.raises(TypeError):
        hellotest.valid_systems = ['+']

    with pytest.raises(TypeError):
        hellotest.valid_systems = ['-']

    with pytest.raises(TypeError):
        hellotest.valid_systems = ['%']

    with pytest.raises(TypeError):
        hellotest.valid_systems = ['%foo']

    with pytest.raises(TypeError):
        hellotest.valid_systems = ['%foo=']

    with pytest.raises(TypeError):
        hellotest.valid_systems = ['+x0 -y0 %z0']

    with pytest.raises(TypeError):
        hellotest.valid_systems = ['+x0 - %z0=w0']

    with pytest.raises(TypeError):
        hellotest.valid_systems = ['%']

    for sym in '!@#$^&()=<>':
        with pytest.raises(TypeError):
            hellotest.valid_systems = [f'{sym}foo']

    for sym in '!@#$%^&*()+=<>':
        with pytest.raises(TypeError):
            hellotest.valid_systems = [f'foo{sym}']


def test_valid_prog_environs_syntax(hellotest):
    hellotest.valid_prog_environs = ['*']
    hellotest.valid_prog_environs = ['env']
    hellotest.valid_prog_environs = ['env-0']
    hellotest.valid_prog_environs = ['env.0']
    hellotest.valid_prog_environs = ['+x0']
    hellotest.valid_prog_environs = ['-y0']
    hellotest.valid_prog_environs = ['%z0=w0']
    hellotest.valid_prog_environs = ['+x0 -y0 %z0=w0']
    hellotest.valid_prog_environs = ['-y0 +x0 %z0=w0']
    hellotest.valid_prog_environs = ['%z0=w0 +x0 -y0']
    hellotest.valid_prog_environs = ['+foo.bar']
    hellotest.valid_prog_environs = ['%foo.bar=a$xx']

    with pytest.raises(TypeError):
        hellotest.valid_prog_environs = ['']

    with pytest.raises(TypeError):
        hellotest.valid_prog_environs = ['  env0']

    with pytest.raises(TypeError):
        hellotest.valid_prog_environs = ['env0  ']

    with pytest.raises(TypeError):
        hellotest.valid_prog_environs = [':']

    with pytest.raises(TypeError):
        hellotest.valid_prog_environs = [':foo']

    with pytest.raises(TypeError):
        hellotest.valid_prog_environs = ['foo:']

    with pytest.raises(TypeError):
        hellotest.valid_prog_environs = ['+']

    with pytest.raises(TypeError):
        hellotest.valid_prog_environs = ['-']

    with pytest.raises(TypeError):
        hellotest.valid_prog_environs = ['%']

    with pytest.raises(TypeError):
        hellotest.valid_prog_environs = ['%foo']

    with pytest.raises(TypeError):
        hellotest.valid_prog_environs = ['%foo=']

    with pytest.raises(TypeError):
        hellotest.valid_prog_environs = ['+x0 -y0 %z0']

    with pytest.raises(TypeError):
        hellotest.valid_prog_environs = ['+x0 - %z0=w0']

    with pytest.raises(TypeError):
        hellotest.valid_prog_environs = ['%']

    for sym in '!@#$^&()=<>:':
        with pytest.raises(TypeError):
            hellotest.valid_prog_environs = [f'{sym}foo']

    for sym in '!@#$%^&*()+=<>:':
        with pytest.raises(TypeError):
            hellotest.valid_prog_environs = [f'foo{sym}']


def test_supports_sysenv(testsys_exec_ctx):
    def _named_comb(valid_sysenv):
        ret = {}
        for part, environs in valid_sysenv.items():
            ret[part.fullname] = [env.name for env in environs]

        return ret

    def _assert_supported(valid_systems, valid_prog_environs,
                          expected, **kwargs):
        valid_comb = _named_comb(
            rt.valid_sysenv_comb(valid_systems, valid_prog_environs, **kwargs)
        )
        assert expected == valid_comb

    _assert_supported(
        valid_systems=['*'],
        valid_prog_environs=['*'],
        expected={
            'testsys:login': ['PrgEnv-cray', 'PrgEnv-gnu'],
            'testsys:gpu': ['PrgEnv-gnu', 'builtin']
        }
    )
    _assert_supported(
        valid_systems=['*:*'],
        valid_prog_environs=['*'],
        expected={
            'testsys:login': ['PrgEnv-cray', 'PrgEnv-gnu'],
            'testsys:gpu': ['PrgEnv-gnu', 'builtin']
        }
    )
    _assert_supported(
        valid_systems=['testsys'],
        valid_prog_environs=['*'],
        expected={
            'testsys:login': ['PrgEnv-cray', 'PrgEnv-gnu'],
            'testsys:gpu': ['PrgEnv-gnu', 'builtin']
        }
    )
    _assert_supported(
        valid_systems=['testsys:*'],
        valid_prog_environs=['*'],
        expected={
            'testsys:login': ['PrgEnv-cray', 'PrgEnv-gnu'],
            'testsys:gpu': ['PrgEnv-gnu', 'builtin']
        }
    )
    _assert_supported(
        valid_systems=['testsys:gpu'],
        valid_prog_environs=['*'],
        expected={
            'testsys:gpu': ['PrgEnv-gnu', 'builtin']
        }
    )
    _assert_supported(
        valid_systems=['testsys:login'],
        valid_prog_environs=['*'],
        expected={
            'testsys:login': ['PrgEnv-cray', 'PrgEnv-gnu'],
        }
    )
    _assert_supported(
        valid_systems=['foo'],
        valid_prog_environs=['*'],
        expected={}
    )
    _assert_supported(
        valid_systems=['*:gpu'],
        valid_prog_environs=['*'],
        expected={
            'testsys:gpu': ['PrgEnv-gnu', 'builtin']
        }
    )

    # Check feature support
    _assert_supported(
        valid_systems=['+cuda'],
        valid_prog_environs=['*'],
        expected={
            'testsys:gpu': ['PrgEnv-gnu', 'builtin']
        }
    )

    # Check AND in features and extras
    _assert_supported(
        valid_systems=[r'+cuda +mpi %gpu_arch=v100'],
        valid_prog_environs=['*'],
        expected={}
    )
    _assert_supported(
        valid_systems=['+cuda -mpi'],
        valid_prog_environs=['*'],
        expected={}
    )

    # Check OR in features and extras
    _assert_supported(
        valid_systems=['+cuda +mpi', r'%gpu_arch=v100'],
        valid_prog_environs=['*'],
        expected={
            'testsys:gpu': ['PrgEnv-gnu', 'builtin']
        }
    )

    # Check that extra keys can used as features
    _assert_supported(
        valid_systems=['+cuda +mpi', '+gpu_arch'],
        valid_prog_environs=['*'],
        expected={
            'testsys:gpu': ['PrgEnv-gnu', 'builtin']
        }
    )

    # Check that resources are taken into account
    _assert_supported(
        valid_systems=['+gpu +datawarp'],
        valid_prog_environs=['*'],
        expected={
            'testsys:gpu': ['PrgEnv-gnu', 'builtin']
        }
    )

    # Check negation
    _assert_supported(
        valid_systems=['-mpi -gpu'],
        valid_prog_environs=['*'],
        expected={
            'testsys:login': ['PrgEnv-cray', 'PrgEnv-gnu']
        }
    )
    _assert_supported(
        valid_systems=['-mpi -foo'],
        valid_prog_environs=['*'],
        expected={
            'testsys:login': ['PrgEnv-cray', 'PrgEnv-gnu']
        }
    )
    _assert_supported(
        valid_systems=['+gpu -datawarp'],
        valid_prog_environs=['*'],
        expected={}
    )

    # Test environment scoping
    _assert_supported(
        valid_systems=['*'],
        valid_prog_environs=['PrgEnv-cray'],
        expected={
            'testsys:gpu': [],
            'testsys:login': ['PrgEnv-cray']
        }
    )
    _assert_supported(
        valid_systems=['*'],
        valid_prog_environs=['+cxx14'],
        expected={
            'testsys:gpu': [],
            'testsys:login': ['PrgEnv-cray', 'PrgEnv-gnu']
        }
    )
    _assert_supported(
        valid_systems=['*'],
        valid_prog_environs=['+cxx14 -cxx14'],
        expected={
            'testsys:gpu': [],
            'testsys:login': []
        }
    )
    _assert_supported(
        valid_systems=['*'],
        valid_prog_environs=['+cxx14', '-cxx14'],
        expected={
            'testsys:gpu': ['PrgEnv-gnu', 'builtin'],
            'testsys:login': ['PrgEnv-cray', 'PrgEnv-gnu']
        }
    )
    _assert_supported(
        valid_systems=['*'],
        valid_prog_environs=[r'%bar=x'],
        expected={
            'testsys:gpu': [],
            'testsys:login': ['PrgEnv-gnu']
        }
    )
    _assert_supported(
        valid_systems=['*'],
        valid_prog_environs=[r'%foo=2'],
        expected={
            'testsys:gpu': ['PrgEnv-gnu'],
            'testsys:login': []
        }
    )
    _assert_supported(
        valid_systems=['*'],
        valid_prog_environs=[r'%foo=bar'],
        expected={
            'testsys:gpu': [],
            'testsys:login': []
        }
    )
    _assert_supported(
        valid_systems=['*'],
        valid_prog_environs=['-cxx14'],
        expected={
            'testsys:gpu': ['PrgEnv-gnu', 'builtin'],
            'testsys:login': []
        }
    )

    # Check that extra keys can used as features
    _assert_supported(
        valid_systems=['*'],
        valid_prog_environs=['+foo +bar'],
        expected={
            'testsys:gpu': ['PrgEnv-gnu'],
            'testsys:login': ['PrgEnv-gnu']
        }
    )
    _assert_supported(
        valid_systems=['*'],
        valid_prog_environs=['+foo -bar'],
        expected={
            'testsys:gpu': [],
            'testsys:login': []
        }
    )

    # Check valid_systems / valid_prog_environs combinations
    _assert_supported(
        valid_systems=['testsys:login'],
        valid_prog_environs=['-cxx14'],
        expected={
            'testsys:login': []
        }
    )
    _assert_supported(
        valid_systems=['+cross_compile'],
        valid_prog_environs=['-cxx14'],
        expected={
            'testsys:login': []
        }
    )

    # Test skipping validity checks
    _assert_supported(
        valid_systems=['foo'],
        valid_prog_environs=['*'],
        expected={
            'testsys:login': ['PrgEnv-cray', 'PrgEnv-gnu'],
            'testsys:gpu': ['PrgEnv-gnu', 'builtin']
        },
        check_systems=False
    )
    _assert_supported(
        valid_systems=['foo'],
        valid_prog_environs=['xxx'],
        expected={
            'testsys:login': ['PrgEnv-cray', 'PrgEnv-gnu'],
            'testsys:gpu': ['PrgEnv-gnu', 'builtin']
        },
        check_systems=False,
        check_environs=False
    )


def test_sourcesdir_none(local_exec_ctx):
    @test_util.custom_prefix('unittests/resources/checks')
    class MyTest(rfm.RegressionTest):
        sourcesdir = None
        valid_prog_environs = ['*']
        valid_systems = ['*']

    with pytest.raises(ReframeError):
        _run(MyTest(), *local_exec_ctx)


def test_sourcesdir_build_system(local_exec_ctx):
    @test_util.custom_prefix('unittests/resources/checks')
    class MyTest(rfm.RegressionTest):
        build_system = 'Make'
        sourcepath = 'code'
        executable = './code/hello'
        valid_systems = ['*']
        valid_prog_environs = ['*']

        @sanity_function
        def validate(self):
            return sn.assert_found(r'Hello, World\!', self.stdout)

    _run(MyTest(), *local_exec_ctx)


def test_sourcesdir_git(local_exec_ctx):
    @test_util.custom_prefix('unittests/resources/checks')
    class MyTest(rfm.RunOnlyRegressionTest):
        sourcesdir = 'https://github.com/reframe-hpc/ci-hello-world.git'
        executable = 'true'
        valid_systems = ['*']
        valid_prog_environs = ['*']
        keep_files = ['README.md']

        @sanity_function
        def validate(self):
            print(self.stagedir)
            return sn.assert_true(os.path.exists('README.md'))

    _run(MyTest(), *local_exec_ctx)


def test_sourcesdir_none_generated_sources(local_exec_ctx):
    @test_util.custom_prefix('unittests/resources/checks')
    class MyTest(rfm.RegressionTest):
        sourcesdir = None
        prebuild_cmds = [
            "printf '#include <stdio.h>\\n int main(){ "
            "printf(\"Hello, World!\\\\n\"); return 0; }' > hello.c"
        ]
        executable = './hello'
        sourcepath = 'hello.c'
        valid_systems = ['*']
        valid_prog_environs = ['*']

        @sanity_function
        def validate(self):
            return sn.assert_found(r'Hello, World\!', self.stdout)

    _run(MyTest(), *local_exec_ctx)


def test_sourcesdir_none_compile_only(local_exec_ctx):
    @test_util.custom_prefix('unittests/resources/checks')
    class MyTest(rfm.CompileOnlyRegressionTest):
        sourcesdir = None
        valid_prog_environs = ['*']
        valid_systems = ['*']

    with pytest.raises(BuildError):
        _run(MyTest(), *local_exec_ctx)


def test_sourcesdir_none_run_only(local_exec_ctx):
    @test_util.custom_prefix('unittests/resources/checks')
    class MyTest(rfm.RunOnlyRegressionTest):
        sourcesdir = None
        executable = 'echo'
        executable_opts = ['Hello, World!']
        valid_prog_environs = ['*']
        valid_systems = ['*']

        @sanity_function
        def validate(self):
            return sn.assert_found(r'Hello, World\!', self.stdout)

    _run(MyTest(), *local_exec_ctx)


def test_sourcepath_abs(local_exec_ctx):
    @test_util.custom_prefix('unittests/resources/checks')
    class MyTest(rfm.CompileOnlyRegressionTest):
        valid_prog_environs = ['*']
        valid_systems = ['*']

    test = MyTest()
    test.setup(*local_exec_ctx)
    test.sourcepath = '/usr/src'
    with pytest.raises(PipelineError):
        test.compile()


def test_sourcepath_upref(local_exec_ctx):
    @test_util.custom_prefix('unittests/resources/checks')
    class MyTest(rfm.CompileOnlyRegressionTest):
        valid_prog_environs = ['*']
        valid_systems = ['*']

    test = MyTest()
    test.setup(*local_exec_ctx)
    test.sourcepath = '../hellosrc'
    with pytest.raises(PipelineError):
        test.compile()


def test_sourcepath_non_existent(local_exec_ctx):
    @test_util.custom_prefix('unittests/resources/checks')
    class MyTest(rfm.CompileOnlyRegressionTest):
        valid_prog_environs = ['*']
        valid_systems = ['*']

    test = MyTest()
    test.setup(*local_exec_ctx)
    test.sourcepath = 'non_existent.c'
    test.compile()
    with pytest.raises(BuildError):
        test.compile_wait()


def test_extra_resources(HelloTest, testsys_exec_ctx):
    @test_util.custom_prefix('unittests/resources/checks')
    class MyTest(HelloTest):
        local = True

        @run_after('setup')
        def set_resources(self):
            test.extra_resources = {
                'gpu': {'num_gpus_per_node': 2},
                'datawarp': {'capacity': '100GB',
                             'stagein_src': test.stagedir}
            }
            test.job.options += ['--foo']

    test = MyTest()
    partition = test_util.partition_by_name('gpu')
    environ = partition.environment('builtin')
    _run(test, partition, environ)
    expected_job_options = {'--gres=gpu:2',
                            '#DW jobdw capacity=100GB',
                            f'#DW stage_in source={test.stagedir}',
                            '--foo'}
    assert expected_job_options == set(test.job.options)


def test_unkown_pre_hook():
    class MyTest(rfm.RunOnlyRegressionTest):
        @run_before('foo')
        def prepare(self):
            self.x = 1

    with pytest.raises(ValueError):
        MyTest()


def test_unkown_post_hook():
    class MyTest(rfm.RunOnlyRegressionTest):
        @run_after('foo')
        def prepare(self):
            self.x = 1

    with pytest.raises(ValueError):
        MyTest()


def test_pre_init_hook():
    class MyTest(rfm.RunOnlyRegressionTest):
        @run_before('init')
        def prepare(self):
            self.x = 1

    with pytest.raises(ValueError):
        MyTest()


def test_post_init_hook(local_exec_ctx):
    class _T0(rfm.RunOnlyRegressionTest):
        x = variable(str, value='y')
        y = variable(str, value='x')

        def __init__(self):
            self.x = 'x'

        @run_after('init')
        def prepare(self):
            self.y += 'y'

    class _T1(_T0):
        def __init__(self):
            super().__init__()
            self.z = 'z'

    t0 = _T0()
    assert t0.x == 'x'
    assert t0.y == 'xy'

    t1 = _T1()
    assert t1.x == 'x'
    assert t1.y == 'xy'
    assert t1.z == 'z'


def test_setup_hooks(HelloTest, local_exec_ctx):
    @test_util.custom_prefix('unittests/resources/checks')
    class MyTest(HelloTest):
        count = variable(int, value=0)

        @run_before('setup')
        def prefoo(self):
            assert self.current_environ is None
            self.count += 1

        @run_after('setup')
        def postfoo(self):
            assert self.current_environ is not None
            self.count += 1

    test = MyTest()
    _run(test, *local_exec_ctx)
    assert test.count == 2


def test_compile_hooks(HelloTest, local_exec_ctx):
    @test_util.custom_prefix('unittests/resources/checks')
    class MyTest(HelloTest):
        count = variable(int, value=0)

        @run_before('compile')
        def setflags(self):
            self.count += 1

        @run_after('compile')
        def check_executable(self):
            exec_file = os.path.join(self.stagedir, self.executable)

            # Make sure that this hook is executed after compile_wait()
            assert os.path.exists(exec_file)

    test = MyTest()
    _run(test, *local_exec_ctx)
    assert test.count == 1


def test_run_hooks(HelloTest, local_exec_ctx):
    @test_util.custom_prefix('unittests/resources/checks')
    class MyTest(HelloTest):
        @run_before('run')
        def setflags(self):
            self.postrun_cmds = ['echo hello > greetings.txt']

        @run_after('run')
        def check_executable(self):
            outfile = os.path.join(self.stagedir, 'greetings.txt')

            # Make sure that this hook is executed after wait()
            assert os.path.exists(outfile)

    _run(MyTest(), *local_exec_ctx)


def test_multiple_hooks(HelloTest, local_exec_ctx):
    @test_util.custom_prefix('unittests/resources/checks')
    class MyTest(HelloTest):
        var = variable(int, value=0)

        @run_after('setup')
        def x(self):
            self.var += 1

        @run_after('setup')
        def y(self):
            self.var += 1

        @run_after('setup')
        def z(self):
            self.var += 1

    test = MyTest()
    _run(test, *local_exec_ctx)
    assert test.var == 3


def test_stacked_hooks(HelloTest, local_exec_ctx):
    @test_util.custom_prefix('unittests/resources/checks')
    class MyTest(HelloTest):
        var = variable(int, value=0)

        @run_before('setup')
        @run_after('setup')
        @run_after('compile')
        def x(self):
            self.var += 1

    test = MyTest()
    _run(test, *local_exec_ctx)
    assert test.var == 3


def test_multiple_inheritance(HelloTest):
    with pytest.raises(ReframeSyntaxError):
        class MyTest(rfm.RunOnlyRegressionTest, HelloTest):
            pass


def test_inherited_hooks(HelloTest, local_exec_ctx):
    @test_util.custom_prefix('unittests/resources/checks')
    class BaseTest(HelloTest):
        var = variable(int, value=0)

        @run_after('setup')
        def x(self):
            self.var += 1

    class C(rfm.RegressionMixin):
        @run_before('run')
        def y(self):
            self.foo = 1

    class DerivedTest(BaseTest, C):
        @run_after('setup')
        def z(self):
            self.var += 1

        @run_after('setup')
        def w(self):
            self.var += 1

    class MyTest(DerivedTest):
        pass

    test = MyTest()
    _run(test, *local_exec_ctx)
    assert test.var == 3
    assert test.foo == 1
    assert test.pipeline_hooks() == {
        'post_setup': [BaseTest.x, DerivedTest.z, DerivedTest.w],
        'pre_run': [C.y],
    }


@pytest.fixture
def weird_mro_test(HelloTest):
    # This returns a class with non-obvious MRO resolution.
    #
    # See example in https://www.python.org/download/releases/2.3/mro/
    #
    # The MRO of A is ABECDFX, which means that E is more specialized than C!
    class X(rfm.RegressionMixin):
        pass

    class D(X):
        @run_after('setup')
        def d(self):
            pass

    class E(X):
        @run_after('setup')
        def e(self):
            pass

    class F(X):
        @run_after('setup')
        def f(self):
            pass

    class C(D, F):
        @run_after('setup')
        def c(self):
            pass

    class B(E, D):
        @run_after('setup')
        def b(self):
            pass

    class A(B, C, HelloTest):
        @run_after('setup')
        def a(self):
            pass

    return A


def test_inherited_hooks_order(weird_mro_test, local_exec_ctx):
    t = weird_mro_test()
    hook_order = [fn.__name__ for fn in t.pipeline_hooks()['post_setup']]
    assert hook_order == ['f', 'd', 'c', 'e', 'b', 'a']


def test_inherited_hooks_from_instantiated_tests(HelloTest, local_exec_ctx):
    @test_util.custom_prefix('unittests/resources/checks')
    class T0(HelloTest):
        var = variable(int, value=0)

        @run_after('setup')
        def x(self):
            self.var += 1

    class T1(T0):
        @run_before('run')
        def y(self):
            self.foo = 1

    t0 = T0()
    t1 = T1()
    print('==> running t0')
    _run(t0, *local_exec_ctx)
    print('==> running t1')
    _run(t1, *local_exec_ctx)
    assert t0.var == 1
    assert t1.var == 1
    assert t1.foo == 1


def test_overriden_hooks(HelloTest, local_exec_ctx):
    @test_util.custom_prefix('unittests/resources/checks')
    class BaseTest(HelloTest):
        var = variable(int, value=0)
        foo = variable(int, value=0)

        @run_after('setup')
        def x(self):
            self.var += 1

        @run_before('setup')
        def y(self):
            self.foo += 1

    class DerivedTest(BaseTest):
        @run_after('setup')
        def x(self):
            self.var += 5

    class MyTest(DerivedTest):
        @run_before('setup')
        def y(self):
            self.foo += 10

    test = MyTest()
    _run(test, *local_exec_ctx)
    assert test.var == 5
    assert test.foo == 10


def test_overriden_hook_different_stages(HelloTest, local_exec_ctx):
    @test_util.custom_prefix('unittests/resources/checks')
    class MyTest(HelloTest):
        @run_after('init')
        def foo(self):
            pass

        @run_after('setup')
        def foo(self):
            pass

    test = MyTest()
    assert test.pipeline_hooks() == {'post_setup': [MyTest.foo]}


def test_overriden_hook_exec_order():
    @test_util.custom_prefix('unittests/resources/checks')
    class X(rfm.RunOnlyRegressionTest):
        @run_before('run')
        def foo(self):
            pass

        @run_before('run')
        def bar(self):
            pass

    class Y(X):
        @run_before('run')
        def foo(self):
            pass

    test = Y()
    assert test.pipeline_hooks() == {'pre_run': [Y.foo, X.bar]}


def test_pinned_hooks():
    class X(rfm.RunOnlyRegressionTest):
        @run_before('run', always_last=True)
        def foo(self):
            pass

        @run_after('sanity', always_last=True)
        def fooX(self):
            '''Check that a single `always_last` hook is registered
            correctly.'''

    class Y(X):
        @run_before('run')
        def bar(self):
            pass

    test = Y()
    assert test.pipeline_hooks() == {
        'pre_run': [Y.bar, X.foo],
        'post_sanity': [X.fooX]
    }


def test_pinned_hooks_multiple_last():
    class X(rfm.RunOnlyRegressionTest):
        @run_before('run', always_last=True)
        def hook_a(self):
            pass

        @run_before('run')
        def hook_b(self):
            pass

    class Y(X):
        @run_before('run', always_last=True)
        def hook_c(self):
            pass

        @run_before('run')
        def hook_d(self):
            pass

    test = Y()
    assert test.pipeline_hooks() == {
        'pre_run': [X.hook_b, Y.hook_d, Y.hook_c, X.hook_a]
    }


def test_pinned_hooks_multiple_last_inherited():
    class X(rfm.RunOnlyRegressionTest):
        @run_before('run', always_last=True)
        def foo(self):
            pass

    class Y(X):
        @run_before('run', always_last=True)
        def bar(self):
            pass

    test = Y()
    assert test.pipeline_hooks() == {'pre_run': [Y.bar, X.foo]}


def test_disabled_hooks(HelloTest, local_exec_ctx):
    @test_util.custom_prefix('unittests/resources/checks')
    class BaseTest(HelloTest):
        var = variable(int, value=0)
        foo = variable(int, value=0)

        @run_after('setup')
        def x(self):
            self.var += 1

        @run_before('setup')
        def y(self):
            self.foo += 1

    class MyTest(BaseTest):
        @run_after('setup')
        def x(self):
            self.var += 5

    test = MyTest()
    test.disable_hook('y')
    _run(test, *local_exec_ctx)
    assert test.var == 5
    assert test.foo == 0


def test_require_deps(HelloTest, local_exec_ctx):
    import reframe.frontend.dependencies as dependencies
    import reframe.frontend.executors as executors

    @test_util.custom_prefix('unittests/resources/checks')
    class T0(HelloTest):
        x = variable(int, value=1)

    @test_util.custom_prefix('unittests/resources/checks')
    class T1(HelloTest):
        @run_after('init')
        def setdeps(self):
            self.depends_on('T0')

        @require_deps
        def sety(self, T0):
            self.y = T0().x + 1

        @run_before('run')
        @require_deps
        def setz(self, T0):
            self.z = T0().x + 2

    cases = executors.generate_testcases([T0(), T1()], prepare=True)
    deps, _ = dependencies.build_deps(cases)
    for c in dependencies.toposort(deps):
        _run(*c)

    for c in cases:
        t = c.check
        if t.name == 'T0':
            assert t.x == 1
        elif t.name == 'T1':
            assert t.y == 2
            assert t.z == 3


def test_trap_job_errors_without_sanity_patterns(local_exec_ctx):
    rt.runtime().site_config.add_sticky_option('general/trap_job_errors', True)

    @test_util.custom_prefix('unittests/resources/checks')
    class MyTest(rfm.RunOnlyRegressionTest):
        valid_prog_environs = ['*']
        valid_systems = ['*']
        executable = 'exit 10'

    with pytest.raises(SanityError, match='job exited with exit code 10'):
        _run(MyTest(), *local_exec_ctx)


def test_trap_job_errors_with_sanity_patterns(local_exec_ctx):
    rt.runtime().site_config.add_sticky_option('general/trap_job_errors', True)

    @test_util.custom_prefix('unittests/resources/checks')
    class MyTest(rfm.RunOnlyRegressionTest):
        valid_prog_environs = ['*']
        valid_systems = ['*']
        prerun_cmds = ['echo hello']
        executable = 'true'

        @sanity_function
        def validate(self):
            return sn.assert_not_found(r'hello', self.stdout)

    with pytest.raises(SanityError):
        _run(MyTest(), *local_exec_ctx)


def _run_sanity(test, *exec_ctx, skip_perf=False):
    test.setup(*exec_ctx)
    test.check_sanity()
    if not skip_perf:
        test.check_performance()


@pytest.fixture
def dummy_gpu_exec_ctx(testsys_exec_ctx):
    partition = test_util.partition_by_name('gpu')
    environ = test_util.environment_by_name('builtin', partition)
    yield partition, environ


@pytest.fixture
def perf_file(tmp_path):
    yield tmp_path / 'perf.out'


@pytest.fixture
def sanity_file(tmp_path):
    yield tmp_path / 'sanity.out'


# NOTE: The following series of tests test the `perf_patterns` syntax, so they
# should not change to the `@performance_function` syntax`

@pytest.fixture
def dummytest(testsys_exec_ctx, perf_file, sanity_file):
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


@pytest.fixture
def dummytest_modern(testsys_exec_ctx, perf_file, sanity_file):
    '''Modern version of the dummytest above'''

    class MyTest(rfm.RunOnlyRegressionTest):
        perf_file = perf_file
        reference = {
            'testsys': {
                'value1': (1.4, -0.1, 0.1, None),
                'value2': (1.7, -0.1, 0.1, None),
            },
            'testsys:gpu': {
                'value3': (3.1, -0.1, 0.1, None),
            }
        }

        @sanity_function
        def validate(self):
            return sn.assert_found(r'result = success', sanity_file)

        @performance_function('unit')
        def value1(self):
            return sn.extractsingle(r'perf1 = (\S+)', perf_file, 1, float)

        @performance_function('unit')
        def value2(self):
            return sn.extractsingle(r'perf2 = (\S+)', perf_file, 1, float)

        @performance_function('unit')
        def value3(self):
            return sn.extractsingle(r'perf3 = (\S+)', perf_file, 1, float)

    yield MyTest()


@pytest.fixture(params=['classic', 'modern'])
def dummy_perftest(request, dummytest, dummytest_modern):
    if request.param == 'modern':
        return dummytest_modern
    else:
        return dummytest


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


def test_required_reference(dummy_perftest, sanity_file,
                            perf_file, dummy_gpu_exec_ctx):
    sanity_file.write_text('result = success\n')
    perf_file.write_text('perf1 = 1.3\n'
                         'perf2 = 1.8\n'
                         'perf3 = 3.3\n')

    dummy_perftest.require_reference = True
    dummy_perftest.reference = {
        'testsys:login': {
            'value1': (1.4, -0.1, 0.1, None),
            'value3': (3.1, -0.1, 0.1, None),
        },
        'foo': {
            'value2': (1.7, -0.1, 0.1, None)
        }
    }
    with pytest.raises(PerformanceError):
        _run_sanity(dummy_perftest, *dummy_gpu_exec_ctx)


def test_reference_deferrable(dummy_perftest):
    with pytest.raises(TypeError):
        dummy_perftest.reference = {'*': {'value1': (sn.defer(1), -0.1, -0.1)}}

    with pytest.raises(TypeError):
        class T(rfm.RegressionTest):
            reference = {'*': {'value1': (sn.defer(1), -0.1, -0.1)}}


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


def test_perf_patterns_evaluation(dummytest, sanity_file,
                                  perf_file, dummy_gpu_exec_ctx):
    # All performance values must be evaluated, despite the first one
    # failing To test this, we need an extract function that will have a
    # side effect when evaluated, whose result we will check after calling
    # `check_performance()`.
    logfile = 'perf.log'

    @sn.deferrable
    def extract_perf(patt, tag):
        val = sn.evaluate(
            sn.extractsingle(patt, perf_file, tag, float)
        )
        with open('perf.log', 'a') as fp:
            fp.write(f'{tag}={val}')

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
    with pytest.raises(PerformanceError):
        _run_sanity(dummytest, *dummy_gpu_exec_ctx)

    logfile = os.path.join(dummytest.stagedir, logfile)
    with open(logfile) as fp:
        log_output = fp.read()

    assert 'v1' in log_output
    assert 'v2' in log_output
    assert 'v3' in log_output


@pytest.fixture
def perftest(testsys_exec_ctx, perf_file, sanity_file):
    class MyTest(rfm.RunOnlyRegressionTest):
        sourcesdir = None

        @sanity_function
        def dummy_sanity(self):
            return sn.assert_found(r'success', sanity_file)

        @performance_function('unit')
        def value1(self):
            pass

        @performance_function('unit')
        def value2(self):
            pass

        @performance_function('unit', perf_key='value3')
        def value_3(self):
            pass

    yield MyTest()


def test_validate_default_perf_variables(perftest):
    assert len(perftest.perf_variables) == 3
    assert 'value1' in perftest.perf_variables
    assert 'value2' in perftest.perf_variables
    assert 'value3' in perftest.perf_variables


def test_perf_vars_without_reference(perftest, sanity_file,
                                     perf_file, dummy_gpu_exec_ctx):
    logfile = 'perf.log'

    @sn.deferrable
    def extract_perf(patt, tag):
        val = sn.evaluate(
            sn.extractsingle(patt, perf_file, tag, float)
        )
        with open(logfile, 'a') as fp:
            fp.write(f'{tag}={val}')

        return val

    sanity_file.write_text('result = success\n')
    perf_file.write_text('perf1 = 1.0\n'
                         'perf3 = 3.3\n')
    perftest.perf_variables = {
        'value1': sn.make_performance_function(
            extract_perf(r'perf1 = (?P<v1>\S+)', 'v1'), 'unit'
        ),
        'value3': sn.make_performance_function(
            extract_perf, 'unit', r'perf3 = (?P<v3>\S+)', 'v3'
        )
    }
    _run_sanity(perftest, *dummy_gpu_exec_ctx)

    logfile = os.path.join(perftest.stagedir, logfile)
    with open(logfile) as fp:
        log_output = fp.read()

    assert 'v1' in log_output
    assert 'v3' in log_output


def test_perf_vars_with_reference(perftest, sanity_file,
                                  perf_file, dummy_gpu_exec_ctx):
    # This test also checks that a performance function that raises an
    # exception is simply skipped.

    logfile = 'perf.log'

    @sn.deferrable
    def extract_perf(patt, tag):
        val = sn.evaluate(
            sn.extractsingle(patt, perf_file, tag, float)
        )
        with open(logfile, 'a') as fp:
            fp.write(f'{tag}={val}')

        return val

    def dummy_perf(x):
        # Dummy function to check that a performance variable is simply
        # skipped when the wrong number of arguments are passed to it.
        with open(logfile, 'a') as fp:
            fp.write('v2')

        return 1

    sanity_file.write_text('result = success\n')
    perf_file.write_text('perf1 = 1.0\n')

    # Make the unit in the reference different from the performance function
    perftest.reference = {
        '*': {
            'value1': (0, None, None, 'unit_')
        }
    }
    perftest.perf_variables = {
        'value1': sn.make_performance_function(
            extract_perf(r'perf1 = (?P<v1>\S+)', 'v1'), 'unit'
        ),
        'value2': sn.make_performance_function(
            dummy_perf, 'other_units', perftest, 'extra_arg'
        ),
    }
    _run_sanity(perftest, *dummy_gpu_exec_ctx)

    logfile = os.path.join(perftest.stagedir, logfile)
    with open(logfile) as fp:
        log_output = fp.read()

    assert 'v1' in log_output
    assert 'v2' not in log_output


def test_incompat_perf_syntax(perftest, sanity_file,
                              perf_file, dummy_gpu_exec_ctx):
    sanity_file.write_text('result = success\n')
    perf_file.write_text('perf1 = 1.0\n')
    perftest.perf_patterns = {}
    with pytest.raises(ReframeSyntaxError):
        _run_sanity(perftest, *dummy_gpu_exec_ctx)


@pytest.fixture(params=['pass', 'fail', 'xpass', 'xfail'])
def value1_status(request):
    return request.param


@pytest.fixture(params=['pass', 'fail', 'xpass', 'xfail'])
def value2_status(request):
    return request.param


def test_perf_expected_failures(perftest, sanity_file, perf_file,
                                value1_status, value2_status, local_exec_ctx):
    def reftuple(status):
        if status == 'pass':
            return (10, 0, 0)
        elif status == 'fail':
            return (5, 0, 0)
        elif status == 'xpass':
            return builtins.xfail('bug 123', (10, 0, 0))
        elif status == 'xfail':
            return builtins.xfail('bug 456', (5, 0, 0))

        assert 0, 'unknown perf status'

    perftest.perf_variables = {
        'value1': sn.make_performance_function(
            sn.extractsingle(r'value1 = (\S+)', perf_file, 1, float), 'unit'
        ),
        'value2': sn.make_performance_function(
            sn.extractsingle(r'value2 = (\S+)', perf_file, 1, float), 'unit'
        )
    }
    perftest.reference = {
        '*': {
            'value1': reftuple(value1_status),
            'value2': reftuple(value2_status)
        }
    }
    sanity_file.write_text('result = success\n')
    perf_file.write_text('value1 = 10\nvalue2 = 10')

    # Matrix of expected exceptions based on status of observed values
    # - Rows represent `value1_status`
    # - Columns represent `value2_status`
    index = {status: i for i, status in enumerate(['pass', 'fail',
                                                   'xpass', 'xfail'])}
    P  = PerformanceError
    US = UnexpectedSuccessError
    EF = ExpectedFailureError
    exceptions = [
        [None, P, US, EF],
        [   P, P,  P,  P],  # noqa: E201
        [  US, P, US, US],  # noqa: E201
        [  EF, P, US, EF]   # noqa: E201
    ]
    exctype = exceptions[index[value1_status]][index[value2_status]]
    if exctype is None:
        _run_sanity(perftest, *local_exec_ctx)
    else:
        with pytest.raises(exctype):
            _run_sanity(perftest, *local_exec_ctx)


@pytest.fixture
def container_test(tmp_path):
    def _container_test(platform, image):
        @test_util.custom_prefix(tmp_path)
        class ContainerTest(rfm.RunOnlyRegressionTest):
            valid_prog_environs = ['*']
            valid_systems = ['*']
            prerun_cmds = ['touch foo']

            @run_after('init')
            def setup_container_platf(self):
                if platform:
                    self.container_platform = platform

                self.container_platform.image = image
                self.container_platform.command = (
                    "bash -c 'cd /rfm_workdir; pwd; ls; "
                    "cat /etc/os-release'"
                )

            @sanity_function
            def assert_os_release(self):
                return sn.all([
                    sn.assert_found(r'^/rfm_workdir', self.stdout),
                    sn.assert_found(r'^foo', self.stdout),
                    sn.assert_found(
                        r'18\.04\.\d+ LTS \(Bionic Beaver\)', self.stdout),
                ])

        return ContainerTest()

    yield _container_test


def _cray_cle_version():
    completed = osext.run_command('cat /etc/opt/cray/release/cle-release')
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

    with pytest.raises(PipelineError):
        _run(container_test(None, 'ubuntu:18.04'), *local_exec_ctx)


def test_skip_if_no_topo(HelloTest, local_exec_ctx):
    class MyTest(HelloTest):
        skip_message = variable(str, type(None), value=None)

        @run_after('setup')
        def access_topo(self):
            self.skip_if_no_procinfo(self.skip_message)

    class EchoTest(rfm.RunOnlyRegressionTest):
        valid_systems = ['*']
        valid_prog_environs = ['*']
        executable = 'echo'
        sanity_patterns = sn.assert_true(1)

        @run_before('setup')
        def access_topo(self):
            self.skip_if_no_procinfo()

    # The test should be skipped, because the auto-detection has not run
    t = MyTest()
    with pytest.raises(SkipTestError, match='no topology.*information'):
        _run(t, *local_exec_ctx)

    # Re-run to test that the custom message is used
    t.skip_message = 'custom message'
    with pytest.raises(SkipTestError, match='custom message'):
        _run(t, *local_exec_ctx)

    # This test should run to completion without problems
    _run(EchoTest(), *local_exec_ctx)


def test_make_test_without_builtins(local_exec_ctx):
    hello_cls = make_test(
        'HelloTest', (rfm.RunOnlyRegressionTest,),
        {
            'valid_systems': ['*'],
            'valid_prog_environs': ['*'],
            'executable': 'echo',
            'sanity_patterns': sn.assert_true(1)
        }
    )

    assert hello_cls.__name__ == 'HelloTest'
    assert hello_cls.__module__ == 'unittests.test_pipeline'
    _run(hello_cls(), *local_exec_ctx)


def test_make_test_with_module():
    hello_cls = make_test(
        'HelloTest', (rfm.RunOnlyRegressionTest,),
        {
            'valid_systems': ['*'],
            'valid_prog_environs': ['*'],
            'executable': 'echo',
            'sanity_patterns': sn.assert_true(1)
        },
        module='foo'
    )

    assert hello_cls.__name__ == 'HelloTest'
    assert hello_cls.__module__ == 'foo'


def test_make_test_with_builtins(local_exec_ctx):
    class _X(rfm.RunOnlyRegressionTest):
        valid_systems = ['*']
        valid_prog_environs = ['*']
        executable = 'echo'
        message = variable(str)

        @run_before('run')
        def set_message(self):
            self.executable_opts = [self.message]

        @sanity_function
        def validate(self):
            return sn.assert_found(self.message, self.stdout)

    hello_cls = make_test('HelloTest', (_X,), {})
    hello_cls.setvar('message', 'hello')
    assert hello_cls.__name__ == 'HelloTest'
    _run(hello_cls(), *local_exec_ctx)


def test_make_test_with_builtins_inline(local_exec_ctx):
    def set_message(obj):
        obj.executable_opts = [obj.message]

    def validate(obj):
        return sn.assert_found(obj.message, obj.stdout)

    hello_cls = make_test(
        'HelloTest', (rfm.RunOnlyRegressionTest,),
        {
            'valid_systems': ['*'],
            'valid_prog_environs': ['*'],
            'executable': 'echo',
            'message': builtins.variable(str),
        },
        methods=[
            builtins.run_before('run')(set_message),
            builtins.sanity_function(validate)
        ]
    )
    hello_cls.setvar('message', 'hello')
    assert hello_cls.__name__ == 'HelloTest'
    _run(hello_cls(), *local_exec_ctx)


def test_set_var_default():
    class _X(rfm.RunOnlyRegressionTest):
        foo = variable(int, value=10)
        bar = variable(int)
        zoo = variable(int)

        @run_after('init')
        def set_defaults(self):
            self.set_var_default('foo', 100)
            self.set_var_default('bar', 100)
            self.zoo = 1
            self.set_var_default('zoo', 100)
            with pytest.raises(ValueError):
                self.set_var_default('foobar', 10)

    x = _X()
    assert x.foo == 10
    assert x.bar == 100
    assert x.zoo == 1


def test_hashcode():
    # We always redefine _X0 here so that the test gets always the same base
    # name (class name) and only the parameter values should change. We then
    # use aliases to access the various definitions for our assertions.

    class _X0(rfm.RunOnlyRegressionTest):
        p = parameter([1])

    class _X0(rfm.RunOnlyRegressionTest):   # noqa: F811
        p = parameter([2])

    _X1 = _X0

    class _X0(rfm.RunOnlyRegressionTest):
        p = parameter([1, 2])

    _X2 = _X0

    t0 = _X0(variant_num=0)
    t1 = _X1(variant_num=0)
    t2, t3 = (_X2(variant_num=i) for i in range(_X2.num_variants))

    assert t0.hashcode != t1.hashcode
    assert t2.hashcode == t0.hashcode
    assert t3.hashcode == t1.hashcode


def test_variables_deprecation():
    with pytest.warns(ReframeDeprecationWarning):
        class _X(rfm.RunOnlyRegressionTest):
            variables = {'FOO': 1}

    test = _X()
    assert test.env_vars['FOO'] == 1

    with pytest.warns(ReframeDeprecationWarning):
        test.variables['BAR'] = 2

    assert test.env_vars['BAR'] == 2
