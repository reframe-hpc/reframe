# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import functools
import io
import itertools
import json
import os
import pytest
import re
import sys
import time

import reframe.core.environments as env
import reframe.core.logging as logging
import reframe.core.runtime as rt
import reframe.frontend.reporting as reporting
import reframe.utility.osext as osext
import unittests.utility as test_util
from reframe import INSTALL_PREFIX


# Absolute path relative to the installation path
norm_path = functools.partial(os.path.join, INSTALL_PREFIX)


def run_command_inline(argv, funct, *args, **kwargs):
    # Save current execution context
    argv_save = sys.argv
    environ_save = env.snapshot()
    sys.argv = argv
    exitcode = None

    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    print(f'pushd {os.getcwd()}; ', *sys.argv, '; popd')
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
def run_reframe(tmp_path, monkeypatch):
    def _run_reframe(system='generic:default',
                     checkpath=None,
                     environs=None,
                     local=True,
                     action='run',
                     more_options=None,
                     mode=None,
                     config_file=None,
                     perflogdir=None):
        import reframe.frontend.cli as cli

        argv = [norm_path('bin/reframe'), '--nocolor']
        if mode:
            argv += ['--mode', mode]

        if system:
            argv += ['--system', system]

        if config_file is None:
            config_file = norm_path('unittests/resources/config/settings.py')

        if checkpath is None:
            checkpath = ['unittests/resources/checks/hellocheck.py']

        if environs is None:
            environs = ['builtin']

        argv += ['-C', config_file]
        argv += itertools.chain(*(['-c', norm_path(c)] for c in checkpath))
        argv += itertools.chain(*(['-p', e] for e in environs))
        if local:
            argv += ['-S', 'local=1']

        if action == 'run':
            argv += ['-r']
        elif action == 'dry_run':
            argv += ['--dry-run']
        elif action == 'list':
            argv += ['-l']
        elif action == 'list_detailed':
            argv += ['-L']
        elif action == 'list_concretized':
            argv += ['-lC']
        elif action == 'list_detailed_concretized':
            argv += ['-LC']
        elif action == 'list_tags':
            argv += ['--list-tags']
        elif action == 'help':
            argv += ['-h']
        elif action == 'describe':
            argv += ['--describe']
        else:
            argv += [action]

        if perflogdir:
            argv += ['--perflogdir', perflogdir]

        if more_options:
            argv += more_options

        return run_command_inline(argv, cli.main)

    monkeypatch.setenv('HOME', str(tmp_path))
    with osext.change_dir(tmp_path):
        yield _run_reframe


@pytest.fixture
def user_exec_ctx(make_exec_ctx_g):
    if test_util.USER_CONFIG_FILE is None:
        pytest.skip('no user configuration file supplied')

    yield from make_exec_ctx_g(test_util.USER_CONFIG_FILE,
                               test_util.USER_SYSTEM)


@pytest.fixture
def remote_exec_ctx(user_exec_ctx):
    partition = test_util.partition_by_scheduler()
    if not partition:
        pytest.skip('job submission not supported')

    return partition, partition.environs[0]


@pytest.fixture(params=['run', 'dry_run'])
def run_action(request):
    return request.param


def test_check_success(run_reframe, tmp_path, run_action):
    returncode, stdout, _ = run_reframe(more_options=['--save-log-files'],
                                        action=run_action)
    assert 'PASSED' in stdout
    assert 'FAILED' not in stdout
    assert returncode == 0

    if run_action == 'dry_run':
        assert 'DRY' in stdout

    logfile = logging.log_files()[0]
    assert os.path.exists(tmp_path / 'output' / logfile)
    assert os.path.exists(tmp_path / '.reframe' /
                          'reports' / 'run-report-0.json')


def test_check_restore_session_failed(run_reframe, tmp_path):
    run_reframe(
        checkpath=['unittests/resources/checks_unlisted/deps_complex.py']
    )
    run_reframe(checkpath=[],  more_options=['--restore-session', '--failed'])
    report = reporting.restore_session(
        f'{tmp_path}/.reframe/reports/latest.json'
    )
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
    run_reframe(checkpath=[], more_options=['--restore-session', '-n', 'T1'])
    report = reporting.restore_session(
        f'{tmp_path}/.reframe/reports/latest.json'
    )
    assert report['runs'][-1]['num_cases'] == 1
    assert report['runs'][-1]['testcases'][0]['name'] == 'T1'

    restored = {r['name'] for r in report['restored_cases']}
    assert restored == {'T4', 'T5'}


def test_check_restore_session_check_search_path(run_reframe, tmp_path):
    run_reframe(
        checkpath=['unittests/resources/checks_unlisted/deps_complex.py']
    )
    returncode, stdout, _ = run_reframe(
        checkpath=['foo/'],
        more_options=['--restore-session', '-n', 'T1', '-R'],
        action='list'
    )
    assert returncode == 0
    assert 'Found 0 check(s)' in stdout


def test_check_success_force_local(run_reframe, tmp_path, run_action):
    # We explicitly use a system here with a non-local scheduler and we run
    # with `-S local=1`
    returncode, stdout, _ = run_reframe(system='testsys:gpu', local=True,
                                        action=run_action)
    assert 'PASSED' in stdout
    assert 'FAILED' not in stdout
    assert returncode == 0


def test_report_file_with_sessionid(run_reframe, tmp_path, run_action):
    returncode, *_ = run_reframe(
        more_options=['--report-file=report-{sessionid}.json'],
        action=run_action
    )
    assert returncode == 0
    assert os.path.exists(tmp_path / 'report-0.json')

    # Assert no symlink is created when the `--report-file` is passed
    assert not os.path.exists(tmp_path / 'latest.json')
    assert not os.path.exists(tmp_path / '.reframe' /
                              'reports' / 'latest.json')


def test_report_ends_with_newline(run_reframe, tmp_path, run_action):
    returncode, stdout, _ = run_reframe(action=run_action)
    assert returncode == 0
    with open(tmp_path / '.reframe' / 'reports' / 'run-report-0.json') as fp:
        assert fp.read()[-1] == '\n'


def test_report_file_symlink_latest(run_reframe, tmp_path, run_action):
    returncode, stdout, _ = run_reframe(action=run_action)
    assert returncode == 0
    assert os.path.exists(tmp_path / '.reframe' /
                          'reports' / 'run-report-0.json')

    symlink = tmp_path / '.reframe' / 'reports' / 'latest.json'
    assert os.readlink(symlink) == 'run-report-0.json'


def test_check_submit_success(run_reframe, remote_exec_ctx, run_action):
    # This test will run on the auto-detected system
    partition, environ = remote_exec_ctx
    returncode, stdout, _ = run_reframe(
        config_file=test_util.USER_CONFIG_FILE,
        local=False,
        system=partition.fullname,
        # Pick up the programming environment of the partition
        # Prepend ^ and append $ so as to much exactly the given name
        environs=[f'^{environ.name}$'],
        action=run_action
    )

    assert 'FAILED' not in stdout
    assert 'PASSED' in stdout

    # Assert that we have run only one test case
    assert 'Ran 2/2 test case(s)' in stdout
    assert 0 == returncode


def test_check_failure(run_reframe):
    returncode, stdout, _ = run_reframe(
        checkpath=['unittests/resources/checks/frontend_checks.py'],
        more_options=['-n', 'BadSetupCheck$']
    )
    assert 'FAILED' in stdout
    assert returncode != 0


def test_check_setup_failure(run_reframe):
    returncode, stdout, stderr = run_reframe(
        checkpath=['unittests/resources/checks/frontend_checks.py'],
        more_options=['-n', 'BadSetupCheckEarly'],
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
        more_options=['-n', 'KeyboardInterruptCheck'],
        local=False,
    )
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert 'FAILED' in stdout
    assert returncode != 0


def test_check_sanity_failure(run_reframe, tmp_path, run_action):
    returncode, stdout, stderr = run_reframe(
        checkpath=['unittests/resources/checks/frontend_checks.py'],
        more_options=['-n', 'SanityFailureCheck'],
        action=run_action
    )
    # This is a normal failure, it should not raise any exception
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr

    assert os.path.exists(
        tmp_path / 'stage' / 'generic' / 'default' /
        'builtin' / 'SanityFailureCheck' / 'rfm_job.sh'
    )
    assert not os.path.exists(
        tmp_path / 'output' / 'generic' / 'default' /
        'builtin' / 'SanityFailureCheck' / 'rfm_job.sh'
    )

    if run_action == 'dry_run':
        assert 'PASSED' in stdout
        assert returncode == 0
        assert not os.path.exists(
            tmp_path / 'stage' / 'generic' / 'default' /
            'builtin' / 'SanityFailureCheck' / 'rfm_job.out'
        )
    else:
        assert 'FAILED' in stdout
        assert returncode != 0


def test_dont_restage(run_reframe, tmp_path):
    run_reframe(
        checkpath=['unittests/resources/checks/frontend_checks.py'],
        more_options=['-n', 'SanityFailureCheck']
    )

    # Place a random file in the test's stage directory and rerun with
    # `--dont-restage` and `--max-retries`
    stagedir = (tmp_path / 'stage' / 'generic' / 'default' /
                'builtin' / 'SanityFailureCheck')
    (stagedir / 'foobar').touch()
    returncode, stdout, stderr = run_reframe(
        checkpath=['unittests/resources/checks/frontend_checks.py'],
        more_options=['-n', 'SanityFailureCheck',
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


def test_performance_check_failure(run_reframe, tmp_path, run_action):
    returncode, stdout, stderr = run_reframe(
        checkpath=['unittests/resources/checks/frontend_checks.py'],
        more_options=['-n', 'PerformanceFailureCheck'],
        action=run_action
    )
    if run_action == 'dry_run':
        assert returncode == 0
        assert 'PASSED' in stdout
    else:
        assert returncode != 0
        assert 'FAILED' in stdout

    # This is a normal failure, it should not raise any exception
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert os.path.exists(
        tmp_path / 'stage' / 'generic' / 'default' /
        'builtin' / 'PerformanceFailureCheck'
    )
    if run_action == 'run':
        assert os.path.exists(tmp_path / 'perflogs' / 'generic' /
                              'default' / 'PerformanceFailureCheck.log')


def test_perflogdir_from_env(run_reframe, tmp_path, monkeypatch):
    monkeypatch.setenv('FOODIR', str(tmp_path / 'perflogs'))
    returncode, stdout, stderr = run_reframe(
        checkpath=['unittests/resources/checks/frontend_checks.py'],
        more_options=['-n', 'PerformanceFailureCheck'],
        perflogdir='$FOODIR'
    )
    assert returncode == 1
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert os.path.exists(tmp_path / 'perflogs' / 'generic' /
                          'default' / 'PerformanceFailureCheck.log')


def test_performance_report(run_reframe, run_action):
    returncode, stdout, stderr = run_reframe(
        checkpath=['unittests/resources/checks/frontend_checks.py'],
        more_options=['-n', '^PerformanceFailureCheck',
                      '--performance-report'],
        action=run_action
    )

    if run_action == 'run':
        assert returncode == 1
    else:
        assert returncode == 0

    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr


def test_skip_system_check_option(run_reframe, run_action):
    returncode, stdout, _ = run_reframe(
        checkpath=['unittests/resources/checks/frontend_checks.py'],
        more_options=['--skip-system-check', '-n', 'NoSystemCheck'],
        action=run_action
    )
    assert 'PASSED' in stdout
    assert returncode == 0


def test_skip_prgenv_check_option(run_reframe, run_action):
    returncode, stdout, _ = run_reframe(
        checkpath=['unittests/resources/checks/frontend_checks.py'],
        more_options=['--skip-prgenv-check', '-n', 'NoPrgEnvCheck'],
        action=run_action
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


def test_execution_modes(run_reframe, run_action):
    returncode, stdout, stderr = run_reframe(
        mode='unittest', action=run_action
    )
    assert returncode == 0
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert 'FAILED' not in stdout
    assert 'PASSED' in stdout
    assert 'Ran 1/1 test case' in stdout


def test_invalid_mode_error(run_reframe):
    mode = 'invalid'
    returncode, stdout, stderr = run_reframe(
        action='list',
        checkpath=[],
        environs=[],
        local=False,
        mode=mode
    )
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert returncode == 1
    assert f'ERROR: invalid mode: {mode!r}' in stdout


def test_timestamp_option(run_reframe):
    timefmt = time.strftime('xxx_%F')
    returncode, stdout, _ = run_reframe(
        checkpath=['unittests/resources/checks'],
        action='list',
        more_options=['-R', '--timestamp=xxx_%F']
    )
    assert returncode == 0
    assert timefmt in stdout


def test_timestamp_option_default(run_reframe):
    returncode, stdout, _ = run_reframe(
        checkpath=['unittests/resources/checks'],
        action='list',
        more_options=['-R', '--timestamp']
    )
    assert returncode == 0

    matches = re.findall(
        r'(stage|output) directory: .*\/(\d{8}T\d{6}\+\d{4})', stdout
    )
    assert len(matches) == 2


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


def test_list_concretized(run_reframe):
    returncode, stdout, stderr = run_reframe(
        checkpath=['unittests/resources/checks/frontend_checks.py'],
        action='list_concretized'
    )
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert returncode == 0

    returncode, stdout, stderr = run_reframe(
        checkpath=['unittests/resources/checks/frontend_checks.py'],
        action='list_detailed_concretized'
    )
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert returncode == 0


def test_list_tags(run_reframe):
    returncode, stdout, stderr = run_reframe(
        checkpath=['unittests/resources/checks/hellocheck.py',
                   'unittests/resources/checks/hellocheck_make.py'],
        action='list_tags'
    )
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert 'Found 2 tag(s)' in stdout
    assert "'bar', 'foo'" in stdout
    assert returncode == 0


def test_list_tests_with_deps(run_reframe):
    returncode, stdout, _ = run_reframe(
        system='sys0',
        checkpath=['unittests/resources/checks_unlisted/deps_simple.py'],
        action='list',
        environs=[]
    )
    assert 'Found 9 check(s)' in stdout
    assert returncode == 0


def test_list_tests_with_fixtures(run_reframe):
    returncode, stdout, _ = run_reframe(
        system='sys0',
        checkpath=['unittests/resources/checks_unlisted/fixtures_simple.py'],
        action='list',
        environs=[]
    )
    assert 'Found 3 check(s)' in stdout
    assert returncode == 0


def test_filtering_multiple_criteria_name(run_reframe):
    returncode, stdout, stderr = run_reframe(
        checkpath=['unittests/resources/checks'],
        action='list',
        more_options=['-t', 'foo', '-n', 'HelloTest']
    )
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert 'Found 1 check(s)' in stdout
    assert returncode == 0


def test_filtering_multiple_criteria_hash(run_reframe):
    returncode, stdout, stderr = run_reframe(
        checkpath=['unittests/resources/checks'],
        action='list',
        more_options=['-t', 'foo', '-n', '/2b3e4546']
    )
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert 'Found 1 check(s)' in stdout
    assert returncode == 0


def test_filtering_exclude_hash(run_reframe):
    returncode, stdout, stderr = run_reframe(
        checkpath=['unittests/resources/checks'],
        action='list',
        more_options=['-x', '/2b3e4546']
    )
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert 'Found 8 check(s)' in stdout
    assert returncode == 0


def test_filtering_cpu_only(run_reframe):
    returncode, stdout, stderr = run_reframe(
        checkpath=['unittests/resources/checks/hellocheck.py'],
        action='list',
        more_options=['--cpu-only']
    )
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert 'Found 2 check(s)' in stdout
    assert returncode == 0


def test_filtering_gpu_only(run_reframe):
    returncode, stdout, stderr = run_reframe(
        checkpath=['unittests/resources/checks/hellocheck.py'],
        action='list',
        more_options=['--gpu-only']
    )
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert 'Found 0 check(s)' in stdout
    assert returncode == 0


def test_filtering_by_expr(run_reframe):
    returncode, stdout, stderr = run_reframe(
        checkpath=['unittests/resources/checks/hellocheck.py'],
        action='list',
        more_options=['-E num_tasks==1']
    )
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert 'Found 2 check(s)' in stdout
    assert returncode == 0


def test_show_config_all(run_reframe):
    # Just make sure that this option does not make the frontend crash
    returncode, stdout, stderr = run_reframe(
        action='--show-config',
        system='testsys'
    )
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert returncode == 0


def test_show_config_param(run_reframe):
    # Just make sure that this option does not make the frontend crash
    returncode, stdout, stderr = run_reframe(
        action='--show-config=systems',
        system='testsys'
    )
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert returncode == 0


def test_show_config_unknown_param(run_reframe):
    # Just make sure that this option does not make the frontend crash
    returncode, stdout, stderr = run_reframe(
        action='--show-config=foo',
        system='testsys'
    )
    assert 'no such configuration parameter found' in stdout
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert returncode == 0


def test_show_config_null_param(run_reframe):
    returncode, stdout, stderr = run_reframe(
        action='--show-config=general/report_junit',
        system='testsys'
    )
    assert 'null' in stdout
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


def test_quiesce_with_check(run_reframe):
    returncode, stdout, stderr = run_reframe(
        more_options=['-v', '-qqq'],    # Show only errors
        system='testsys',
        action='list',
        checkpath=['unittests/resources/checks/hellocheck.py']
    )
    assert stdout == ''
    assert 'Traceback' not in stderr
    assert 0 == returncode


def test_load_user_modules(run_reframe, user_exec_ctx):
    with rt.module_use(norm_path('unittests/modules')):
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
    if not test_util.has_sane_modules_system():
        pytest.skip('no modules system found')

    with rt.module_use(norm_path('unittests/modules')):
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


def test_unuse_module_path(run_reframe, user_exec_ctx, run_action):
    ms = rt.runtime().modules_system
    if not test_util.has_sane_modules_system():
        pytest.skip('no modules system found')

    module_path = norm_path('unittests/modules')
    ms.searchpath_add(module_path)
    returncode, stdout, stderr = run_reframe(
        more_options=[f'--module-path=-{module_path}', '--module=testmod_foo'],
        config_file=test_util.USER_CONFIG_FILE, action=run_action,
        system=rt.runtime().system.name
    )
    ms.searchpath_remove(module_path)
    assert "could not load module 'testmod_foo' correctly" in stdout
    assert 'Traceback' not in stderr
    assert returncode == 1


def test_use_module_path(run_reframe, user_exec_ctx, run_action):
    if not test_util.has_sane_modules_system():
        pytest.skip('no modules system found')

    module_path = norm_path('unittests/modules')
    returncode, stdout, stderr = run_reframe(
        more_options=[f'--module-path=+{module_path}', '--module=testmod_foo'],
        config_file=test_util.USER_CONFIG_FILE, action=run_action,
        system=rt.runtime().system.name
    )
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert "could not load module 'testmod_foo' correctly" not in stdout
    assert returncode == 0


def test_overwrite_module_path(run_reframe, user_exec_ctx, run_action):
    if not test_util.has_sane_modules_system():
        pytest.skip('no modules system found')

    module_path = norm_path('unittests/modules')
    with contextlib.suppress(KeyError):
        module_path += f':{os.environ["MODULEPATH"]}'

    returncode, stdout, stderr = run_reframe(
        more_options=[f'--module-path={module_path}', '--module=testmod_foo'],
        config_file=test_util.USER_CONFIG_FILE, action=run_action,
        system=rt.runtime().system.name
    )
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert "could not load module 'testmod_foo' correctly" not in stdout
    assert returncode == 0


def test_failure_stats(run_reframe, run_action):
    returncode, stdout, stderr = run_reframe(
        checkpath=['unittests/resources/checks/frontend_checks.py'],
        more_options=['-n', 'SanityFailureCheck', '--failure-stats'],
        action=run_action
    )
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    if run_action == 'dry_run':
        assert returncode == 0
        assert r'FAILURE STATISTICS' not in stdout
    else:
        assert returncode != 0
        assert r'FAILURE STATISTICS' in stdout
        assert r'sanity        1     SanityFailureCheck' in stdout


def test_maxfail_option(run_reframe):
    returncode, stdout, stderr = run_reframe(
        more_options=['--maxfail', '1'],
        system='testsys',
        checkpath=['unittests/resources/checks/hellocheck.py']
    )
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert ('Ran 2/2 test case(s) from 2 check(s) '
            '(0 failure(s), 0 skipped, 0 aborted)') in stdout
    assert returncode == 0


def test_maxfail_invalid_option(run_reframe):
    returncode, stdout, stderr = run_reframe(
        more_options=['--maxfail', 'foo'],
        system='testsys',
        checkpath=['unittests/resources/checks/hellocheck.py']
    )
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert "--maxfail: invalid int value: 'foo'" in stderr
    assert returncode == 2


def test_maxfail_negative(run_reframe):
    returncode, stdout, stderr = run_reframe(
        more_options=['--maxfail', '-2'],
        system='testsys',
        checkpath=['unittests/resources/checks/hellocheck.py']
    )
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert "--maxfail should be a non-negative integer: -2" in stdout
    assert returncode == 1


def test_distribute_build_remotely_runonly(run_reframe, run_action):
    returncode, stdout, stderr = run_reframe(
        local=False,
        more_options=['--distribute', '-S', 'build_locally=0'],
        checkpath=['unittests/resources/checks_unlisted/distribute.py'],
        action=run_action
    )
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert returncode == 0


def test_repeat_option(run_reframe, run_action):
    returncode, stdout, stderr = run_reframe(
        more_options=['--repeat', '2', '-n', '^HelloTest'],
        checkpath=['unittests/resources/checks/hellocheck.py'],
        action=run_action
    )
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert ('Ran 2/2 test case(s) from 2 check(s) '
            '(0 failure(s), 0 skipped, 0 aborted)') in stdout
    assert returncode == 0


def test_repeat_invalid_option(run_reframe):
    returncode, stdout, stderr = run_reframe(
        more_options=['--repeat', 'foo'],
        checkpath=['unittests/resources/checks/hellocheck.py']
    )
    errmsg = "argument to '--repeat' option must be a non-negative integer"
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert errmsg in stdout
    assert returncode == 1


def test_repeat_negative(run_reframe):
    returncode, stdout, stderr = run_reframe(
        more_options=['--repeat', '-1'],
        checkpath=['unittests/resources/checks/hellocheck.py']
    )
    errmsg = "argument to '--repeat' option must be a non-negative integer"
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert errmsg in stdout
    assert returncode == 1


def test_parameterize_tests(run_reframe):
    returncode, stdout, _ = run_reframe(
        more_options=['-P', 'num_tasks=2,4,8', '-n', '^HelloTest'],
        checkpath=['unittests/resources/checks/hellocheck.py'],
        action='describe'
    )
    assert returncode == 0

    test_descr = json.loads(stdout)
    print(json.dumps(test_descr, indent=2))
    num_tasks = {t['num_tasks'] for t in test_descr}
    assert num_tasks == {2, 4, 8}


def test_parameterize_tests_invalid_params(run_reframe):
    returncode, stdout, stderr = run_reframe(
        more_options=['-P', 'num_tasks', '-n', '^HelloTest'],
        checkpath=['unittests/resources/checks/hellocheck.py'],
        action='list'
    )
    assert returncode == 1
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert 'invalid parameter spec' in stdout


def test_reruns_negative(run_reframe):
    returncode, stdout, stderr = run_reframe(
        more_options=['--reruns', '-1'],
        checkpath=['unittests/resources/checks/hellocheck.py']
    )
    errmsg = "'--reruns' should be a non-negative integer"
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert errmsg in stdout
    assert returncode == 1


def test_reruns_with_duration(run_reframe):
    returncode, stdout, stderr = run_reframe(
        more_options=['--reruns=-1', '--duration=5s'],
        checkpath=['unittests/resources/checks/hellocheck.py']
    )
    errmsg = "'--reruns' option cannot be combined with '--duration'"
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert errmsg in stdout
    assert returncode == 1


@pytest.fixture(params=['name', 'rname', 'uid', 'ruid', 'random'])
def exec_order(request):
    return request.param


def test_exec_order(run_reframe, exec_order):
    import reframe.utility.sanity as sn

    returncode, stdout, stderr = run_reframe(
        more_options=['--repeat', '11', '-n', '^HelloTest',
                      f'--exec-order={exec_order}'],
        checkpath=['unittests/resources/checks/hellocheck.py'],
        action='list_detailed',
    )
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert 'Found 11 check(s)' in stdout
    assert returncode == 0

    # Verify the order
    if exec_order == 'name':
        repeat_no = sn.extractsingle_s(r'- HelloTest.*repeat_no=(\d+)',
                                       stdout, 1, int, 2).evaluate()
        assert repeat_no == 10
    elif exec_order == 'rname':
        repeat_no = sn.extractsingle_s(r'- HelloTest.*repeat_no=(\d+)',
                                       stdout, 1, int, -3).evaluate()
        assert repeat_no == 10
    elif exec_order == 'uid':
        repeat_no = sn.extractsingle_s(r'- HelloTest.*repeat_no=(\d+)',
                                       stdout, 1, int, -1).evaluate()
        assert repeat_no == 10
    elif exec_order == 'ruid':
        repeat_no = sn.extractsingle_s(r'- HelloTest.*repeat_no=(\d+)',
                                       stdout, 1, int, 0).evaluate()
        assert repeat_no == 10


def test_detect_host_topology(run_reframe):
    from reframe.utility.cpuinfo import cpuinfo

    returncode, stdout, stderr = run_reframe(action='--detect-host-topology')
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert returncode == 0
    assert stdout == json.dumps(cpuinfo(), indent=2) + '\n'


def test_detect_host_topology_file(run_reframe, tmp_path):
    from reframe.utility.cpuinfo import cpuinfo

    topo_file = tmp_path / 'topo.json'
    returncode, stdout, stderr = run_reframe(
        action=f'--detect-host-topology={topo_file}',
    )
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert returncode == 0
    with open(topo_file) as fp:
        assert json.load(fp) == cpuinfo()


def test_external_vars(run_reframe, run_action):
    returncode, stdout, stderr = run_reframe(
        checkpath=['unittests/resources/checks_unlisted/externalvars.py'],
        more_options=['-S', 'external_x.foo=3',
                      '-S', 'external_x.ham=true',
                      '-S', 'external_x.spam.eggs.bacon=10',
                      '-S', 'external_y.foo=2',
                      '-S', 'external_y.baz=false',
                      '-S', 'foolist=3,4',
                      '-S', 'bar=@none'],
        action=run_action
    )
    assert 'PASSED' in stdout
    assert 'Ran 6/6 test case(s)' in stdout
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert returncode == 0


def test_external_vars_invalid_expr(run_reframe, run_action):
    returncode, stdout, stderr = run_reframe(
        more_options=['-S', 'foo'], action=run_action
    )
    assert 'Traceback' not in stdout
    assert 'Traceback' not in stderr
    assert 'invalid test variable assignment' in stdout
    assert returncode == 0


def test_fixture_registry_env_sys(run_reframe):
    '''Test that the fixtures see the cli options.

    The loaded checks have a system scope fixture named HelloFixture. Hence,
    this fixture will take any partition+environ combination from the parent
    tests. So by restricting the partition and the environment to only one
    single option, here we test that the fixture has the valid_systems and
    valid_prog_environs as passed throught the cli options.
    '''
    returncode, stdout, stderr = run_reframe(
        system='sys1:p0',
        environs=['e3'],
        checkpath=['unittests/resources/checks_unlisted/fixtures_simple.py'],
        more_options=['-n', '^HelloFixture'],
        action='list_detailed'
    )
    assert returncode == 0
    assert 'e3' in stdout
    assert 'sys1:p0' in stdout
    returncode, stdout, stderr = run_reframe(
        system='sys1:p0',
        environs=['e1'],
        checkpath=['unittests/resources/checks_unlisted/fixtures_simple.py'],
        more_options=['-n', '^HelloFixture'],
        action='list_detailed'
    )
    assert returncode == 0
    assert 'e1' in stdout
    assert 'sys1:p0' in stdout
    returncode, stdout, _ = run_reframe(
        system='sys1:p1',
        environs=['e1'],
        checkpath=['unittests/resources/checks_unlisted/fixtures_simple.py'],
        more_options=['-n', '^HelloFixture'],
        action='list_detailed'
    )
    assert returncode == 0
    assert 'e1' in stdout
    assert 'sys1:p1' in stdout
    returncode, stdout, _ = run_reframe(
        system='sys1:p1',
        environs=['e2'],
        checkpath=['unittests/resources/checks_unlisted/fixtures_simple.py'],
        more_options=['-n', '^HelloFixture'],
        action='list_detailed'
    )
    assert returncode == 0
    assert 'e2' in stdout
    assert 'sys1:p1' in stdout


def test_fixture_resolution(run_reframe, run_action):
    returncode, stdout, stderr = run_reframe(
        system='sys1',
        environs=[],
        checkpath=['unittests/resources/checks_unlisted/fixtures_complex.py'],
        action=run_action
    )
    assert returncode == 0


def test_dynamic_tests(run_reframe, run_action):
    returncode, stdout, _ = run_reframe(
        system='sys0',
        environs=[],
        checkpath=['unittests/resources/checks_unlisted/distribute.py'],
        more_options=['-n', '^Complex', '--distribute=idle'],
        action=run_action
    )
    assert returncode == 0
    assert 'Ran 10/10 test case(s)' in stdout
    assert 'FAILED' not in stdout


def test_dynamic_tests_filtering(run_reframe, run_action):
    returncode, stdout, _ = run_reframe(
        system='sys1',
        environs=[],
        checkpath=['unittests/resources/checks_unlisted/distribute.py'],
        more_options=['-n', 'Complex@1', '--distribute=idle'],
        action=run_action
    )
    assert returncode == 0
    assert 'Ran 7/7 test case(s)' in stdout
    assert 'FAILED' not in stdout


def test_testlib_inherit_fixture_in_different_files(run_reframe):
    returncode, stdout, _ = run_reframe(
        checkpath=[
            'unittests/resources/checks_unlisted/testlib_inheritance_foo.py',
            'unittests/resources/checks_unlisted/testlib_inheritance_bar.py'
        ],
        action='run',
    )
    assert returncode == 0
    assert 'Ran 3/3 test case(s)' in stdout
    assert 'FAILED' not in stdout


def test_storage_options(run_reframe, tmp_path):
    def assert_no_crash(returncode, stdout, stderr, exitcode=0):
        assert returncode == exitcode
        assert 'Traceback' not in stdout
        assert 'Traceback' not in stderr
        return returncode, stdout, stderr

    run_reframe2 = functools.partial(
        run_reframe,
        checkpath=['unittests/resources/checks/frontend_checks.py']
    )

    # Run first a normal run with a performance test to initialize the DB
    run_reframe2(action='run')
    assert os.path.exists(tmp_path / '.reframe' / 'reports' / 'results.db')

    stdout = assert_no_crash(
        *run_reframe(action='--list-stored-sessions')
    )[1]

    # Get the session uuid for later queries
    uuid = re.search(r'(\w{8}-\w{4}-\w{4}-\w{4}-\w{12})', stdout).group(1)

    # Get details from the last session
    stdout = assert_no_crash(
        *run_reframe2(action=f'--describe-stored-session={uuid}')
    )[1]
    session_json = json.loads(stdout)

    # List test cases by session
    assert_no_crash(*run_reframe2(action=f'--list-stored-testcases={uuid}'))
    assert_no_crash(
        *run_reframe2(action=f'--describe-stored-testcases={uuid}')
    )

    # List test cases by time period
    ts_start = session_json['session_info']['time_start']
    assert_no_crash(
        *run_reframe2(action=f'--list-stored-testcases={ts_start}:now')
    )
    assert_no_crash(
        *run_reframe2(action=f'--describe-stored-testcases={ts_start}:now')
    )

    # Check that invalid argument do not crash CLI
    assert_no_crash(*run_reframe2(action='--describe-stored-session=0'),
                    exitcode=1)
    assert_no_crash(*run_reframe2(action='--describe-stored-testcases=0'),
                    exitcode=1)
    assert_no_crash(*run_reframe2(action='--list-stored-testcases=0'),
                    exitcode=1)

    # Remove session
    assert_no_crash(*run_reframe2(action=f'--delete-stored-session={uuid}'))


def test_performance_compare(run_reframe):
    def assert_no_crash(returncode, stdout, stderr, exitcode=0):
        assert returncode == exitcode
        assert 'Traceback' not in stdout
        assert 'Traceback' not in stderr
        return returncode, stdout, stderr

    run_reframe2 = functools.partial(
        run_reframe,
        checkpath=['unittests/resources/checks/frontend_checks.py']
    )
    run_reframe2(action='run')

    # Rerun with various arguments
    assert_no_crash(
        *run_reframe2(
            action='--performance-compare=now-1m:now/now-1d:now/mean:/+result'
        )
    )

    # Check that invalid arguments do not crash the CLI
    assert_no_crash(
        *run_reframe2(
            action='--performance-compare=now-1m:now/now-1d:now/mean:+foo/+bar'
        ), exitcode=1
    )
