# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import io
import itertools
import os
import pytest
import re
import sys

import reframe.core.environments as env
import reframe.frontend.runreport as runreport
import reframe.core.logging as logging
import reframe.core.runtime as rt
import unittests.fixtures as fixtures


def run_command_inline(argv, funct, *args, **kwargs):
    # Save current execution context
    argv_save = sys.argv
    environ_save = env.snapshot()
    sys.argv = argv
    exitcode = None

    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    print(*sys.argv)
    with contextlib.redirect_stdout(captured_stdout):
        with contextlib.redirect_stderr(captured_stderr):
            try:
                with rt.temp_runtime(None):
                    exitcode = funct(*args, **kwargs)
            except SystemExit as e:
                exitcode = e.code
            finally:
                # Restore execution context
                environ_save.restore()
                sys.argv = argv_save

    return (exitcode,
            captured_stdout.getvalue(),
            captured_stderr.getvalue())


@pytest.fixture
def perflogdir(tmp_path):
    dirname = tmp_path / '.rfm-perflogs'
    yield dirname


@pytest.fixture
def run_reframe(tmp_path, perflogdir):
    def _run_reframe(system='generic:default',
                     checkpath=['unittests/resources/checks/hellocheck.py'],
                     environs=['builtin'],
                     local=True,
                     action='run',
                     more_options=None,
                     mode=None,
                     config_file='unittests/resources/settings.py',
                     ignore_check_conflicts=True,
                     perflogdir=str(perflogdir)):
        import reframe.frontend.cli as cli

        # We always pass the --report-file option, because we don't want to
        # pollute the user's home directory
        argv = ['./bin/reframe', '--prefix', str(tmp_path), '--nocolor',
                f'--report-file={tmp_path / "report.json"}']
        if mode:
            argv += ['--mode', mode]

        if system:
            argv += ['--system', system]

        if config_file:
            argv += ['-C', config_file]

        argv += itertools.chain(*(['-c', c] for c in checkpath))
        argv += itertools.chain(*(['-p', e] for e in environs))
        if local:
            argv += ['--force-local']

        if action == 'run':
            argv += ['-r']
        elif action == 'list':
            argv += ['-l']
        elif action == 'list_detailed':
            argv += ['-L']
        elif action == 'help':
            argv += ['-h']

        if ignore_check_conflicts:
            argv += ['--ignore-check-conflicts']

        if perflogdir:
            argv += ['--perflogdir', perflogdir]

        if more_options:
            argv += more_options

        return run_command_inline(argv, cli.main)

    return _run_reframe


@pytest.fixture
def temp_runtime(tmp_path):
    def _temp_runtime(site_config, system=None, options=None):
        options = options or {}
        options.update({'systems/prefix': tmp_path})
        with rt.temp_runtime(site_config, system, options):
            yield

    yield _temp_runtime


@pytest.fixture
def user_exec_ctx(temp_runtime):
    if fixtures.USER_CONFIG_FILE is None:
        pytest.skip('no user configuration file supplied')

    yield from temp_runtime(fixtures.USER_CONFIG_FILE, fixtures.USER_SYSTEM)


@pytest.fixture
def remote_exec_ctx(user_exec_ctx):
    partition = fixtures.partition_by_scheduler()
    if not partition:
        pytest.skip('job submission not supported')

    return partition, partition.environs[0]


def test_check_success(run_reframe, tmp_path):
    returncode, stdout, _ = run_reframe(more_options=['--save-log-files'])
    assert 'PASSED' in stdout
    assert 'FAILED' not in stdout
    assert returncode == 0

    logfile = logging.log_files()[0]
    assert os.path.exists(tmp_path / 'output' / logfile)
    assert os.path.exists(tmp_path / 'report.json')


def test_check_restore_session_failed(run_reframe, tmp_path):
    run_reframe(
        checkpath=['unittests/resources/checks_unlisted/deps_complex.py'],
    )
    returncode, stdout, _ = run_reframe(
        checkpath=[],
        more_options=[
            f'--restore-session={tmp_path}/report.json', '--failed'
        ]
    )
    report = runreport.load_report(f'{tmp_path}/report.json')
    assert set(report.slice('name', when=('fail_phase', 'sanity'))) == {'T2'}
    assert set(report.slice('name',
                            when=('fail_phase', 'startup'))) == {'T7', 'T9'}
    assert set(report.slice('name', when=('fail_phase', 'setup'))) == {'T8'}
    assert report['runs'][-1]['num_cases'] == 4

    restored = {r['name'] for r in report['restored_cases']}
    assert restored == {'T1', 'T6'}


def test_check_restore_session_succeeded_test(run_reframe, tmp_path):
    run_reframe(
        checkpath=['unittests/resources/checks_unlisted/deps_complex.py'],
        more_options=['--keep-stage-files']
    )
    returncode, stdout, _ = run_reframe(
        checkpath=[],
        more_options=[
            f'--restore-session={tmp_path}/report.json', '-n', 'T1'
        ]
    )
    report = runreport.load_report(f'{tmp_path}/report.json')
    assert report['runs'][-1]['num_cases'] == 1
    assert report['runs'][-1]['testcases'][0]['name'] == 'T1'

    restored = {r['name'] for r in report['restored_cases']}
    assert restored == {'T4', 'T5'}


def test_check_restore_session_check_search_path(run_reframe, tmp_path):
    run_reframe(
        checkpath=['unittests/resources/checks_unlisted/deps_complex.py']
    )
    returncode, stdout, _ = run_reframe(
        checkpath=[f'{tmp_path}/foo'],
        more_options=[
            f'--restore-session={tmp_path}/report.json', '-n', 'T1', '-R'
        ],
        action='list'
    )
    assert returncode == 0
    assert 'Found 0 check(s)' in stdout


def test_check_success_force_local(run_reframe, tmp_path):
    # We explicitly use a system here with a non-local scheduler and pass the
    # `--force-local` option
    returncode, stdout, _ = run_reframe(system='testsys:gpu', local=True)
    assert 'PASSED' in stdout
    assert 'FAILED' not in stdout
    assert returncode == 0


def test_report_file_with_sessionid(run_reframe, tmp_path):
    returncode, *_ = run_reframe(
        more_options=[
            f'--report-file={tmp_path / "rfm-report-{sessionid}.json"}'
        ]
    )
    assert returncode == 0
    assert os.path.exists(tmp_path / 'rfm-report-0.json')


def test_report_ends_with_newline(run_reframe, tmp_path):
    returncode, stdout, _ = run_reframe(
        more_options=[
            f'--report-file={tmp_path / "rfm-report.json"}'
        ]
    )
    assert returncode == 0
    with open(tmp_path / 'rfm-report.json') as fp:
        assert fp.read()[-1] == '\n'


def test_check_submit_success(run_reframe, remote_exec_ctx):
    # This test will run on the auto-detected system
    partition, environ = remote_exec_ctx
    returncode, stdout, _ = run_reframe(
        config_file=fixtures.USER_CONFIG_FILE,
        local=False,
        system=partition.fullname,
        # Pick up the programming environment of the partition
        # Prepend ^ and append $ so as to much exactly the given name
        environs=[f'^{environ.name}$']
    )

    assert 'FAILED' not in stdout
    assert 'PASSED' in stdout

    # Assert that we have run only one test case
    assert 'Ran 2/2 test case(s)' in stdout
    assert 0 == returncode


def test_check_failure(run_reframe):
    returncode, stdout, _ = run_reframe(
        checkpath=['unittests/resources/checks/frontend_checks.py'],
        more_options=['-t', 'BadSetupCheck']
    )
    assert 'FAILED' in stdout
    assert returncode != 0


def test_check_setup_failure(run_reframe):
    returncode, stdout, stderr = run_reframe(
        checkpath=['unittests/resources/checks/frontend_checks.py'],
        more_options=['-t', 'BadSetupCheckEarly'],
        local=False,

    )
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert 'FAILED' in stdout
    assert returncode != 0


def test_check_kbd_interrupt(run_reframe):
    returncode, stdout, stderr = run_reframe(
        checkpath=[
            'unittests/resources/checks_unlisted/kbd_interrupt.py'
        ],
        more_options=['-t', 'KeyboardInterruptCheck'],
        local=False,
    )
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert 'FAILED' in stdout
    assert returncode != 0


def test_check_sanity_failure(run_reframe, tmp_path):
    returncode, stdout, stderr = run_reframe(
        checkpath=['unittests/resources/checks/frontend_checks.py'],
        more_options=['-t', 'SanityFailureCheck']
    )
    assert 'FAILED' in stdout

    # This is a normal failure, it should not raise any exception
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert returncode != 0
    assert os.path.exists(
        tmp_path / 'stage' / 'generic' / 'default' /
        'builtin' / 'SanityFailureCheck'
    )


def test_dont_restage(run_reframe, tmp_path):
    run_reframe(
        checkpath=['unittests/resources/checks/frontend_checks.py'],
        more_options=['-t', 'SanityFailureCheck']
    )

    # Place a random file in the test's stage directory and rerun with
    # `--dont-restage` and `--max-retries`
    stagedir = (tmp_path / 'stage' / 'generic' / 'default' /
                'builtin' / 'SanityFailureCheck')
    (stagedir / 'foobar').touch()
    returncode, stdout, stderr = run_reframe(
        checkpath=['unittests/resources/checks/frontend_checks.py'],
        more_options=['-t', 'SanityFailureCheck',
                      '--dont-restage', '--max-retries=1']
    )
    assert os.path.exists(stagedir / 'foobar')
    assert not os.path.exists(f'{stagedir}_retry1')

    # And some standard assertions
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert returncode != 0


def test_checkpath_symlink(run_reframe, tmp_path):
    # FIXME: This should move to test_loader.py
    checks_symlink = tmp_path / 'checks_symlink'
    os.symlink(os.path.abspath('unittests/resources/checks'),
               checks_symlink)

    returncode, stdout, _ = run_reframe(
        action='list',
        more_options=['-R'],
        checkpath=['unittests/resources/checks', str(checks_symlink)]
    )
    num_checks_default = re.search(
        r'Found (\d+) check', stdout, re.MULTILINE).group(1)
    num_checks_in_checkdir = re.search(
        r'Found (\d+) check', stdout, re.MULTILINE).group(1)
    assert num_checks_in_checkdir == num_checks_default


def test_performance_check_failure(run_reframe, tmp_path, perflogdir):
    returncode, stdout, stderr = run_reframe(
        checkpath=['unittests/resources/checks/frontend_checks.py'],
        more_options=['-t', 'PerformanceFailureCheck']
    )
    assert 'FAILED' in stdout

    # This is a normal failure, it should not raise any exception
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert returncode != 0
    assert os.path.exists(
        tmp_path / 'stage' / 'generic' / 'default' /
        'builtin' / 'PerformanceFailureCheck'
    )
    assert os.path.exists(perflogdir / 'generic' /
                          'default' / 'PerformanceFailureCheck.log')


def test_perflogdir_from_env(run_reframe, tmp_path, monkeypatch):
    monkeypatch.setenv('FOODIR', str(tmp_path / 'perflogs'))
    returncode, stdout, stderr = run_reframe(
        checkpath=['unittests/resources/checks/frontend_checks.py'],
        more_options=['-t', 'PerformanceFailureCheck'],
        perflogdir='$FOODIR'
    )
    assert returncode == 1
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert os.path.exists(tmp_path / 'perflogs' / 'generic' /
                          'default' / 'PerformanceFailureCheck.log')


def test_performance_report(run_reframe):
    returncode, stdout, _ = run_reframe(
        checkpath=['unittests/resources/checks/frontend_checks.py'],
        more_options=['-t', 'PerformanceFailureCheck', '--performance-report']
    )
    assert r'PERFORMANCE REPORT' in stdout
    assert r'perf: 10 Gflop/s' in stdout


def test_skip_system_check_option(run_reframe):
    returncode, stdout, _ = run_reframe(
        checkpath=['unittests/resources/checks/frontend_checks.py'],
        more_options=['--skip-system-check', '-t', 'NoSystemCheck']
    )
    assert 'PASSED' in stdout
    assert returncode == 0


def test_skip_prgenv_check_option(run_reframe):
    returncode, stdout, _ = run_reframe(
        checkpath=['unittests/resources/checks/frontend_checks.py'],
        more_options=['--skip-prgenv-check', '-t', 'NoPrgEnvCheck']
    )
    assert 'PASSED' in stdout
    assert returncode == 0


def test_sanity_of_checks(run_reframe, tmp_path):
    # This test will effectively load all the tests in the checks path and
    # will force a syntactic and runtime check at least for the constructor
    # of the checks
    returncode, *_ = run_reframe(
        action='list',
        checkpath=[]
    )
    assert returncode == 0


def test_unknown_system(run_reframe):
    returncode, stdout, stderr = run_reframe(
        action='list',
        system='foo',
        checkpath=[]
    )
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert returncode == 1


def test_sanity_of_optconfig(run_reframe):
    # Test the sanity of the command line options configuration
    returncode, *_ = run_reframe(
        action='help',
        checkpath=[]
    )
    assert returncode == 0


def test_checkpath_recursion(run_reframe):
    _, stdout, _ = run_reframe(action='list', checkpath=[])
    num_checks_default = re.search(r'Found (\d+) check', stdout).group(1)

    _, stdout, _ = run_reframe(action='list',
                               checkpath=['checks/'],
                               more_options=['-R'])
    num_checks_in_checkdir = re.search(r'Found (\d+) check', stdout).group(1)
    assert num_checks_in_checkdir == num_checks_default

    _, stdout, _ = run_reframe(action='list',
                               checkpath=['checks/'],
                               more_options=[])
    num_checks_in_checkdir = re.search(r'Found (\d+) check', stdout).group(1)
    assert num_checks_in_checkdir == '0'


def test_same_output_stage_dir(run_reframe, tmp_path):
    output_dir = str(tmp_path / 'foo')
    returncode, *_ = run_reframe(
        more_options=['-o', output_dir, '-s', output_dir]
    )
    assert returncode == 1

    # Retry with --keep-stage-files
    returncode, *_ = run_reframe(
        more_options=['-o', output_dir, '-s', output_dir, '--keep-stage-files']
    )
    assert returncode == 0
    assert os.path.exists(output_dir)


def test_execution_modes(run_reframe):
    returncode, stdout, stderr = run_reframe(
        checkpath=[],
        environs=[],
        local=False,
        mode='unittest'
    )
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert 'FAILED' not in stdout
    assert 'PASSED' in stdout
    assert 'Ran 2/2 test case' in stdout


def test_no_ignore_check_conflicts(run_reframe):
    returncode, *_ = run_reframe(
        checkpath=['unittests/resources/checks'],
        more_options=['-R'],
        ignore_check_conflicts=False,
        action='list'
    )
    assert returncode != 0


def test_timestamp_option(run_reframe):
    from datetime import datetime

    timefmt = datetime.now().strftime('xxx_%F')
    returncode, stdout, _ = run_reframe(
        checkpath=['unittests/resources/checks'],
        ignore_check_conflicts=False,
        action='list',
        more_options=['-R', '--timestamp=xxx_%F']
    )
    assert returncode != 0
    assert timefmt in stdout


def test_list_empty_prgenvs_check_and_options(run_reframe):
    returncode, stdout, _ = run_reframe(
        checkpath=['unittests/resources/checks/frontend_checks.py'],
        action='list',
        environs=[],
        more_options=['-n', 'NoPrgEnvCheck'],
    )
    assert 'Found 0 check(s)' in stdout
    assert returncode == 0


def test_list_check_with_empty_prgenvs(run_reframe):
    returncode, stdout, _ = run_reframe(
        checkpath=['unittests/resources/checks/frontend_checks.py'],
        action='list',
        environs=['foo'],
        more_options=['-n', 'NoPrgEnvCheck']
    )
    assert 'Found 0 check(s)' in stdout
    assert returncode == 0


def test_list_empty_prgenvs_in_check_and_options(run_reframe):
    returncode, stdout, _ = run_reframe(
        checkpath=['unittests/resources/checks/frontend_checks.py'],
        action='list',
        environs=[],
        more_options=['-n', 'NoPrgEnvCheck']
    )
    assert 'Found 0 check(s)' in stdout
    assert returncode == 0


def test_list_with_details(run_reframe):
    returncode, stdout, stderr = run_reframe(
        checkpath=['unittests/resources/checks/frontend_checks.py'],
        action='list_detailed'
    )
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert returncode == 0


def test_filtering_multiple_criteria(run_reframe):
    returncode, stdout, stderr = run_reframe(
        checkpath=['unittests/resources/checks'],
        action='list',
        more_options=['-t', 'foo', '-n', 'hellocheck',
                      '--ignore-check-conflicts']
    )
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert 'Found 1 check(s)' in stdout
    assert returncode == 0


def test_show_config_all(run_reframe):
    # Just make sure that this option does not make the frontend crash
    returncode, stdout, stderr = run_reframe(
        more_options=['--show-config'],
        system='testsys'
    )
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert returncode == 0


def test_show_config_param(run_reframe):
    # Just make sure that this option does not make the frontend crash
    returncode, stdout, stderr = run_reframe(
        more_options=['--show-config=systems'],
        system='testsys'
    )
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert returncode == 0


def test_show_config_unknown_param(run_reframe):
    # Just make sure that this option does not make the frontend crash
    returncode, stdout, stderr = run_reframe(
        more_options=['--show-config=foo'],
        system='testsys'
    )
    assert 'no such configuration parameter found' in stdout
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert returncode == 0


def test_verbosity(run_reframe):
    returncode, stdout, stderr = run_reframe(
        more_options=['-vvvvv'],
        system='testsys',
        action='list'
    )
    assert stdout != ''
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert returncode == 0


def test_verbosity_with_check(run_reframe):
    returncode, stdout, stderr = run_reframe(
        more_options=['-vvvvv'],
        system='testsys',
        action='list',
        checkpath=['unittests/resources/checks/hellocheck.py']
    )
    assert '' != stdout
    assert '--- Logging error ---' not in stdout
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert 0 == returncode


def test_load_user_modules(run_reframe, user_exec_ctx):
    with rt.module_use('unittests/modules'):
        returncode, stdout, stderr = run_reframe(
            more_options=['-m testmod_foo'],
            action='list'
        )

    assert stdout != ''
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert returncode == 0


def test_unload_module(run_reframe, user_exec_ctx):
    # This test is mostly for ensuring coverage. `run_reframe()` restores
    # the current environment, so it is not easy to verify that the modules
    # are indeed unloaded. However, this functionality is tested elsewhere
    # more exhaustively.

    ms = rt.runtime().modules_system
    if not fixtures.has_sane_modules_system():
        pytest.skip('no modules system found')

    with rt.module_use('unittests/modules'):
        ms.load_module('testmod_foo')
        returncode, stdout, stderr = run_reframe(
            more_options=['-u testmod_foo'],
            action='list'
        )
        ms.unload_module('testmod_foo')

    assert stdout != ''
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert returncode == 0


def test_unuse_module_path(run_reframe, user_exec_ctx):
    ms = rt.runtime().modules_system
    if not fixtures.has_sane_modules_system():
        pytest.skip('no modules system found')

    module_path = 'unittests/modules'
    ms.searchpath_add(module_path)
    returncode, stdout, stderr = run_reframe(
        more_options=[f'--module-path=-{module_path}', '--module=testmod_foo'],
        config_file=fixtures.USER_CONFIG_FILE, action='run',
        system=rt.runtime().system.name
    )
    ms.searchpath_remove(module_path)
    assert "could not load module 'testmod_foo' correctly" in stdout
    assert 'Traceback' not in stderr
    assert returncode == 0


def test_use_module_path(run_reframe, user_exec_ctx):
    ms = rt.runtime().modules_system
    if not fixtures.has_sane_modules_system():
        pytest.skip('no modules system found')

    module_path = 'unittests/modules'
    returncode, stdout, stderr = run_reframe(
        more_options=[f'--module-path=+{module_path}', '--module=testmod_foo'],
        config_file=fixtures.USER_CONFIG_FILE, action='run',
        system=rt.runtime().system.name
    )
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert "could not load module 'testmod_foo' correctly" not in stdout
    assert returncode == 0


def test_overwrite_module_path(run_reframe, user_exec_ctx):
    ms = rt.runtime().modules_system
    if not fixtures.has_sane_modules_system():
        pytest.skip('no modules system found')

    module_path = 'unittests/modules'
    with contextlib.suppress(KeyError):
        module_path += f':{os.environ["MODULEPATH"]}'

    returncode, stdout, stderr = run_reframe(
        more_options=[f'--module-path={module_path}', '--module=testmod_foo'],
        config_file=fixtures.USER_CONFIG_FILE, action='run',
        system=rt.runtime().system.name
    )
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert "could not load module 'testmod_foo' correctly" not in stdout
    assert returncode == 0


def test_failure_stats(run_reframe):
    returncode, stdout, stderr = run_reframe(
        checkpath=['unittests/resources/checks/frontend_checks.py'],
        more_options=['-t', 'SanityFailureCheck', '--failure-stats']
    )
    assert r'FAILURE STATISTICS' in stdout
    assert r'sanity        1     [SanityFailureCheck' in stdout
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert returncode != 0


def test_maxfail_option(run_reframe):
    returncode, stdout, stderr = run_reframe(
        more_options=['--maxfail', '1'],
        system='testsys',
        checkpath=['unittests/resources/checks/hellocheck.py']
    )
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert ('Ran 2/2 test case(s) from 2 check(s) '
            '(0 failure(s), 0 skipped)') in stdout
    assert returncode == 0


def test_maxfail_invalid_option(run_reframe):
    returncode, stdout, stderr = run_reframe(
        more_options=['--maxfail', 'foo'],
        system='testsys',
        checkpath=['unittests/resources/checks/hellocheck.py']
    )
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert "--maxfail is not a valid integer: 'foo'" in stdout
    assert returncode == 1


def test_maxfail_negative(run_reframe):
    returncode, stdout, stderr = run_reframe(
        more_options=['--maxfail', '-2'],
        system='testsys',
        checkpath=['unittests/resources/checks/hellocheck.py']
    )
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert "--maxfail should be a non-negative integer: '-2'" in stdout
    assert returncode == 1
