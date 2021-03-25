# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import json
import jsonschema
import os
import pytest
import socket
import sys
import time

import reframe.core.runtime as rt
import reframe.frontend.dependencies as dependencies
import reframe.frontend.executors as executors
import reframe.frontend.executors.policies as policies
import reframe.frontend.runreport as runreport
import reframe.utility.jsonext as jsonext
import reframe.utility.osext as osext
from reframe.core.exceptions import (AbortTaskError,
                                     FailureLimitError,
                                     ReframeError,
                                     ForceExitError,
                                     TaskDependencyError)
from reframe.frontend.loader import RegressionCheckLoader

import unittests.fixtures as fixtures
from unittests.resources.checks.hellocheck import HelloTest
from unittests.resources.checks.frontend_checks import (
    BadSetupCheck,
    BadSetupCheckEarly,
    CompileFailureCheck,
    KeyboardInterruptCheck,
    RetriesCheck,
    SelfKillCheck,
    SleepCheck,
    SleepCheckPollFail,
    SleepCheckPollFailLate,
    SystemExitCheck,
)


# NOTE: We could move this to utility
class timer:
    '''Context manager for timing'''

    def __init__(self):
        self._time_start = None
        self._time_end = None

    def __enter__(self):
        self._time_start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._time_end = time.time()

    def timestamps(self):
        return self._time_start, self._time_end


@pytest.fixture
def temp_runtime(tmp_path):
    def _temp_runtime(site_config, system=None, options={}):
        options.update({'systems/prefix': str(tmp_path)})
        with rt.temp_runtime(site_config, system, options):
            yield rt.runtime

    yield _temp_runtime


@pytest.fixture
def make_loader():
    def _make_loader(check_search_path):
        return RegressionCheckLoader(check_search_path,
                                     ignore_conflicts=True)

    return _make_loader


@pytest.fixture
def common_exec_ctx(temp_runtime):
    yield from temp_runtime(fixtures.TEST_CONFIG_FILE, 'generic')


@pytest.fixture
def testsys_exec_ctx(temp_runtime):
    yield from temp_runtime(fixtures.TEST_CONFIG_FILE, 'testsys:gpu')


@pytest.fixture(params=[policies.SerialExecutionPolicy,
                        policies.AsynchronousExecutionPolicy])
def make_runner(request):
    def _make_runner(*args, **kwargs):
        # Use a much higher poll rate for the unit tests
        policy = request.param()
        policy._pollctl.SLEEP_MIN = 0.001
        return executors.Runner(policy, *args, **kwargs)

    return _make_runner


@pytest.fixture
def make_cases(make_loader):
    def _make_cases(checks=None, sort=False, *args, **kwargs):
        if checks is None:
            checks = make_loader(['unittests/resources/checks']).load_all()

        cases = executors.generate_testcases(checks, *args, **kwargs)
        if sort:
            depgraph, _ = dependencies.build_deps(cases)
            dependencies.validate_deps(depgraph)
            cases = dependencies.toposort(depgraph)

        return cases

    return _make_cases


@pytest.fixture(params=['pre_setup', 'post_setup',
                        'pre_compile', 'post_compile',
                        'pre_run', 'post_run',
                        'pre_sanity', 'post_sanity',
                        'pre_performance', 'post_performance',
                        'pre_cleanup', 'post_cleanup'])
def make_cases_for_skipping(request):
    import reframe as rfm
    import reframe.utility.sanity as sn

    def _make_cases():
        @fixtures.custom_prefix('unittests/resources/checks')
        class _T0(rfm.RegressionTest):
            valid_systems = ['*']
            valid_prog_environs = ['*']
            sourcepath = 'hello.c'
            executable = 'echo'
            sanity_patterns = sn.assert_true(1)

            def check_and_skip(self):
                self.skip_if(True)

            # Attach the hook manually based on the request.param
            when, stage = request.param.split('_', maxsplit=1)
            hook = rfm.run_before if when == 'pre' else rfm.run_after
            check_and_skip = hook(stage)(check_and_skip)

        class _T1(rfm.RunOnlyRegressionTest):
            valid_systems = ['*']
            valid_prog_environs = ['*']
            sanity_patterns = sn.assert_true(1)

            def __init__(self):
                self.depends_on(_T0.__qualname__)

        cases = executors.generate_testcases([_T0(), _T1()])
        depgraph, _ = dependencies.build_deps(cases)
        return dependencies.toposort(depgraph), request.param

    return _make_cases


def assert_runall(runner):
    # Make sure that all cases finished, failed or
    # were aborted
    for t in runner.stats.tasks():
        assert t.succeeded or t.failed or t.aborted or t.skipped


def assert_all_dead(runner):
    stats = runner.stats
    for t in runner.stats.tasks():
        job = t.check.job
        if job:
            assert job.finished


def num_failures_stage(runner, stage):
    stats = runner.stats
    return len([t for t in stats.failed() if t.failed_stage == stage])


def _validate_runreport(report):
    schema_filename = 'reframe/schemas/runreport.json'
    with open(schema_filename) as fp:
        schema = json.loads(fp.read())

    jsonschema.validate(json.loads(report), schema)


def _generate_runreport(run_stats, time_start, time_end):
    return {
        'session_info': {
            'cmdline': ' '.join(sys.argv),
            'config_file': rt.runtime().site_config.filename,
            'data_version': runreport.DATA_VERSION,
            'hostname': socket.gethostname(),
            'num_cases': run_stats[0]['num_cases'],
            'num_failures': run_stats[-1]['num_failures'],
            'prefix_output': rt.runtime().output_prefix,
            'prefix_stage': rt.runtime().stage_prefix,
            'time_elapsed': time_end - time_start,
            'time_end': time.strftime(
                '%FT%T%z', time.localtime(time_end),
            ),
            'time_start': time.strftime(
                '%FT%T%z', time.localtime(time_start),
            ),
            'user': osext.osuser(),
            'version': osext.reframe_version(),
            'workdir': os.getcwd()
        },
        'restored_cases': [],
        'runs': run_stats
    }


def test_runall(make_runner, make_cases, common_exec_ctx, tmp_path):
    runner = make_runner()
    with timer() as tm:
        runner.runall(make_cases())

    assert 9 == runner.stats.num_cases()
    assert_runall(runner)
    assert 5 == len(runner.stats.failed())
    assert 2 == num_failures_stage(runner, 'setup')
    assert 1 == num_failures_stage(runner, 'sanity')
    assert 1 == num_failures_stage(runner, 'performance')
    assert 1 == num_failures_stage(runner, 'cleanup')

    # Create a run report and validate it
    report = _generate_runreport(runner.stats.json(), *tm.timestamps())

    # We dump the report first, in order to get any object conversions right
    report_file = tmp_path / 'report.json'
    with open(report_file, 'w') as fp:
        jsonext.dump(report, fp)

    # Read and validate the report using the runreport module
    runreport.load_report(report_file)

    # Try to load a non-existent report
    with pytest.raises(ReframeError, match='failed to load report file'):
        runreport.load_report(tmp_path / 'does_not_exist.json')

    # Generate an invalid JSON
    with open(tmp_path / 'invalid.json', 'w') as fp:
        jsonext.dump(report, fp)
        fp.write('invalid')

    with pytest.raises(ReframeError, match=r'is not a valid JSON file'):
        runreport.load_report(tmp_path / 'invalid.json')

    # Generate a report with an incorrect data version
    report['session_info']['data_version'] = '10.0.0'
    with open(tmp_path / 'invalid-version.json', 'w') as fp:
        jsonext.dump(report, fp)

    with pytest.raises(ReframeError,
                       match=r'incompatible report data versions'):
        runreport.load_report(tmp_path / 'invalid-version.json')


def test_runall_skip_system_check(make_runner, make_cases, common_exec_ctx):
    runner = make_runner()
    runner.runall(make_cases(skip_system_check=True))
    stats = runner.stats
    assert 10 == stats.num_cases()
    assert_runall(runner)
    assert 5 == len(stats.failed())
    assert 2 == num_failures_stage(runner, 'setup')
    assert 1 == num_failures_stage(runner, 'sanity')
    assert 1 == num_failures_stage(runner, 'performance')
    assert 1 == num_failures_stage(runner, 'cleanup')


def test_runall_skip_prgenv_check(make_runner, make_cases, common_exec_ctx):
    runner = make_runner()
    runner.runall(make_cases(skip_environ_check=True))
    stats = runner.stats
    assert 10 == stats.num_cases()
    assert_runall(runner)
    assert 5 == len(stats.failed())
    assert 2 == num_failures_stage(runner, 'setup')
    assert 1 == num_failures_stage(runner, 'sanity')
    assert 1 == num_failures_stage(runner, 'performance')
    assert 1 == num_failures_stage(runner, 'cleanup')


def test_runall_skip_sanity_check(make_runner, make_cases, common_exec_ctx):
    runner = make_runner()
    runner.policy.skip_sanity_check = True
    runner.runall(make_cases())
    stats = runner.stats
    assert 9 == stats.num_cases()
    assert_runall(runner)
    assert 4 == len(stats.failed())
    assert 2 == num_failures_stage(runner, 'setup')
    assert 0 == num_failures_stage(runner, 'sanity')
    assert 1 == num_failures_stage(runner, 'performance')
    assert 1 == num_failures_stage(runner, 'cleanup')


def test_runall_skip_performance_check(make_runner, make_cases,
                                       common_exec_ctx):
    runner = make_runner()
    runner.policy.skip_performance_check = True
    runner.runall(make_cases())
    stats = runner.stats
    assert 9 == stats.num_cases()
    assert_runall(runner)
    assert 4 == len(stats.failed())
    assert 2 == num_failures_stage(runner, 'setup')
    assert 1 == num_failures_stage(runner, 'sanity')
    assert 0 == num_failures_stage(runner, 'performance')
    assert 1 == num_failures_stage(runner, 'cleanup')


def test_runall_maxfail(make_runner, make_cases, common_exec_ctx):
    runner = make_runner(max_failures=2)
    with contextlib.suppress(FailureLimitError):
        runner.runall(make_cases())

    assert_runall(runner)
    stats = runner.stats
    assert 2 == len(stats.failed())


def test_strict_performance_check(make_runner, make_cases, common_exec_ctx):
    runner = make_runner()
    runner.policy.strict_check = True
    runner.runall(make_cases())
    stats = runner.stats
    assert 9 == stats.num_cases()
    assert_runall(runner)
    assert 6 == len(stats.failed())
    assert 2 == num_failures_stage(runner, 'setup')
    assert 1 == num_failures_stage(runner, 'sanity')
    assert 2 == num_failures_stage(runner, 'performance')
    assert 1 == num_failures_stage(runner, 'cleanup')


def test_runall_skip_tests(make_runner, make_cases,
                           make_cases_for_skipping,
                           common_exec_ctx):
    runner = make_runner(max_retries=1)
    more_cases, stage = make_cases_for_skipping()
    cases = make_cases() + more_cases
    runner.runall(cases)

    def assert_reported_skipped(num_skipped):
        report = runner.stats.json()
        assert report[0]['num_skipped'] == num_skipped

        num_reported = 0
        for tc in report[0]['testcases']:
            if tc['result'] == 'skipped':
                num_reported += 1

        assert num_reported == num_skipped

    assert_runall(runner)
    assert 11 == runner.stats.num_cases(0)
    assert 5 == runner.stats.num_cases(1)
    assert 5 == len(runner.stats.failed(0))
    assert 5 == len(runner.stats.failed(1))
    if stage.endswith('cleanup'):
        assert 0 == len(runner.stats.skipped(0))
        assert 0 == len(runner.stats.skipped(1))
        assert_reported_skipped(0)
    else:
        assert 2 == len(runner.stats.skipped(0))
        assert 0 == len(runner.stats.skipped(1))
        assert_reported_skipped(2)


# We explicitly ask for a system with a non-local scheduler here, to make sure
# that the execution policies behave correctly with forced local tests
def test_force_local_execution(make_runner, make_cases, testsys_exec_ctx):
    runner = make_runner()
    runner.policy.force_local = True
    test = HelloTest()
    test.valid_prog_environs = ['builtin']

    runner.runall(make_cases([test]))
    assert_runall(runner)
    stats = runner.stats
    for t in stats.tasks():
        assert t.check.local

    assert not stats.failed()


def test_kbd_interrupt_within_test(make_runner, make_cases, common_exec_ctx):
    runner = make_runner()
    check = KeyboardInterruptCheck()
    with pytest.raises(KeyboardInterrupt):
        runner.runall(make_cases([KeyboardInterruptCheck()]))

    stats = runner.stats
    assert 1 == len(stats.failed())
    assert_all_dead(runner)


def test_system_exit_within_test(make_runner, make_cases, common_exec_ctx):
    # This should not raise and should not exit
    runner = make_runner()
    runner.runall(make_cases([SystemExitCheck()]))
    stats = runner.stats
    assert 1 == len(stats.failed())


def test_retries_bad_check(make_runner, make_cases, common_exec_ctx):
    runner = make_runner(max_retries=2)
    runner.runall(make_cases([BadSetupCheck(), BadSetupCheckEarly()]))

    # Ensure that the test was retried #max_retries times and failed
    assert 2 == runner.stats.num_cases()
    assert_runall(runner)
    assert runner.max_retries == rt.runtime().current_run
    assert 2 == len(runner.stats.failed())

    # Ensure that the report does not raise any exception
    runner.stats.retry_report()


def test_retries_good_check(make_runner, make_cases, common_exec_ctx):
    runner = make_runner(max_retries=2)
    runner.runall(make_cases([HelloTest()]))

    # Ensure that the test passed without retries.
    assert 1 == runner.stats.num_cases()
    assert_runall(runner)
    assert 0 == rt.runtime().current_run
    assert 0 == len(runner.stats.failed())


def test_pass_in_retries(make_runner, make_cases, tmp_path, common_exec_ctx):
    tmpfile = tmp_path / 'out.txt'
    tmpfile.write_text('0\n')
    runner = make_runner(max_retries=3)
    pass_run_no = 2
    runner.runall(make_cases([RetriesCheck(pass_run_no, tmpfile)]))

    # Ensure that the test passed after retries in run `pass_run_no`
    assert 1 == runner.stats.num_cases()
    assert_runall(runner)
    assert 1 == len(runner.stats.failed(run=0))
    assert pass_run_no == rt.runtime().current_run
    assert 0 == len(runner.stats.failed())


def test_sigterm_handling(make_runner, make_cases, common_exec_ctx):
    runner = make_runner()
    with pytest.raises(ForceExitError,
                       match='received TERM signal'):
        runner.runall(make_cases([SelfKillCheck()]))

    assert_all_dead(runner)
    assert runner.stats.num_cases() == 1
    assert len(runner.stats.failed()) == 1


@pytest.fixture
def dep_checks(make_loader):
    return make_loader(
        ['unittests/resources/checks_unlisted/deps_complex.py']
    ).load_all()


@pytest.fixture
def dep_cases(dep_checks, make_cases):
    return make_cases(dep_checks, sort=True)


def assert_dependency_run(runner):
    assert_runall(runner)
    stats = runner.stats
    assert 10 == stats.num_cases(0)
    assert 4  == len(stats.failed())
    for tf in stats.failed():
        check = tf.testcase.check
        _, exc_value, _ = tf.exc_info
        if check.name == 'T7' or check.name == 'T9':
            assert isinstance(exc_value, TaskDependencyError)

    # Check that cleanup is executed properly for successful tests as well
    for t in stats.tasks():
        check = t.testcase.check
        if t.failed:
            continue

        if t.ref_count == 0:
            assert os.path.exists(os.path.join(check.outputdir, 'out.txt'))


def test_dependencies(make_runner, dep_cases, common_exec_ctx):
    runner = make_runner()
    runner.runall(dep_cases)
    assert_dependency_run(runner)


def test_dependencies_with_retries(make_runner, dep_cases, common_exec_ctx):
    runner = make_runner(max_retries=2)
    runner.runall(dep_cases)
    assert_dependency_run(runner)


class _TaskEventMonitor(executors.TaskEventListener):
    '''Event listener for monitoring the execution of the asynchronous
    execution policy.

    We need to make sure two things for the async policy:

    1. The number of running tasks never exceed the max job size per partition.
    2. Given a set of regression tests with a reasonably long runtime, the
       execution policy must be able to reach the maximum concurrency. By
       reasonably long runtime, we mean that that the regression tests must run
       enough time, so as to allow the policy to execute all the tests until
       their "run" phase, before the first submitted test finishes.
    '''

    def __init__(self):
        super().__init__()

        # timeline of num_tasks
        self.num_tasks = [0]
        self.tasks = []

    def on_task_run(self, task):
        super().on_task_run(task)
        last = self.num_tasks[-1]
        self.num_tasks.append(last + 1)
        self.tasks.append(task)

    def on_task_exit(self, task):
        last = self.num_tasks[-1]
        self.num_tasks.append(last - 1)

    def on_task_success(self, task):
        pass

    def on_task_failure(self, task):
        pass

    def on_task_skip(self, task):
        pass

    def on_task_setup(self, task):
        pass


@pytest.fixture
def make_async_exec_ctx(temp_runtime):
    def _make_async_exec_ctx(max_jobs):
        yield from temp_runtime(fixtures.TEST_CONFIG_FILE, 'generic',
                                {'systems/partitions/max_jobs': max_jobs})

    return _make_async_exec_ctx


@pytest.fixture
def async_runner():
    evt_monitor = _TaskEventMonitor()
    ret = executors.Runner(policies.AsynchronousExecutionPolicy())
    ret.policy.keep_stage_files = True
    ret.policy.task_listeners.append(evt_monitor)
    return ret, evt_monitor


def _read_timestamps(tasks):
    '''Read the timestamps and sort them to permit simple
    concurrency tests.'''
    from reframe.utility.sanity import evaluate

    begin_stamps = []
    end_stamps = []
    for t in tasks:
        with osext.change_dir(t.check.stagedir):
            with open(evaluate(t.check.stdout), 'r') as f:
                begin_stamps.append(float(f.readline().strip()))
                end_stamps.append(float(f.readline().strip()))

    begin_stamps.sort()
    end_stamps.sort()
    return begin_stamps, end_stamps


def test_concurrency_unlimited(async_runner, make_cases, make_async_exec_ctx):
    num_checks = 3

    # Trigger evaluation of the execution context
    ctx = make_async_exec_ctx(num_checks)
    next(ctx)

    runner, monitor = async_runner
    runner.runall(make_cases([SleepCheck(.5) for i in range(num_checks)]))

    # Ensure that all tests were run and without failures.
    assert num_checks == runner.stats.num_cases()
    assert_runall(runner)
    assert 0 == len(runner.stats.failed())

    # Ensure that maximum concurrency was reached as fast as possible
    assert num_checks == max(monitor.num_tasks)
    assert num_checks == monitor.num_tasks[num_checks]
    begin_stamps, end_stamps = _read_timestamps(monitor.tasks)

    # Warn if not all tests were run in parallel; the corresponding strict
    # check would be:
    #
    #     assert begin_stamps[-1] <= end_stamps[0]
    #
    if begin_stamps[-1] > end_stamps[0]:
        pytest.skip('the system seems too much loaded.')


def test_concurrency_limited(async_runner, make_cases, make_async_exec_ctx):
    # The number of checks must be <= 2*max_jobs.
    num_checks, max_jobs = 5, 3
    ctx = make_async_exec_ctx(max_jobs)
    next(ctx)

    runner, monitor = async_runner
    runner.runall(make_cases([SleepCheck(.5) for i in range(num_checks)]))

    # Ensure that all tests were run and without failures.
    assert num_checks == runner.stats.num_cases()
    assert_runall(runner)
    assert 0 == len(runner.stats.failed())

    # Ensure that maximum concurrency was reached as fast as possible
    assert max_jobs == max(monitor.num_tasks)
    assert max_jobs == monitor.num_tasks[max_jobs]

    begin_stamps, end_stamps = _read_timestamps(monitor.tasks)

    # Ensure that the jobs after the first #max_jobs were each run after
    # one of the previous #max_jobs jobs had finished
    # (e.g. begin[max_jobs] > end[0]).
    # Note: we may ensure this strictly as we may ensure serial behaviour.
    begin_after_end = (b > e for b, e in zip(begin_stamps[max_jobs:],
                                             end_stamps[:-max_jobs]))
    assert all(begin_after_end)

    # NOTE: to ensure that these remaining jobs were also run
    # in parallel one could do the command hereafter; however, it would
    # require to substantially increase the sleep time (in SleepCheck),
    # because of the delays in rescheduling (1s, 2s, 3s, 1s, 2s,...).
    # We currently prefer not to do this last concurrency test to avoid an
    # important prolongation of the unit test execution time.
    # self.assertTrue(self.begin_stamps[-1] < self.end_stamps[max_jobs])

    # Warn if the first #max_jobs jobs were not run in parallel; the
    # corresponding strict check would be:
    # self.assertTrue(self.begin_stamps[max_jobs-1] <= self.end_stamps[0])
    if begin_stamps[max_jobs-1] > end_stamps[0]:
        pytest.skip('the system seems too loaded.')


def test_concurrency_none(async_runner, make_cases, make_async_exec_ctx):
    num_checks = 3
    ctx = make_async_exec_ctx(1)
    next(ctx)

    runner, monitor = async_runner
    runner.runall(make_cases([SleepCheck(.5) for i in range(num_checks)]))

    # Ensure that all tests were run and without failures.
    assert num_checks == runner.stats.num_cases()
    assert_runall(runner)
    assert 0 == len(runner.stats.failed())

    # Ensure that a single task was running all the time
    assert 1 == max(monitor.num_tasks)

    # Read the timestamps sorted to permit simple concurrency tests.
    begin_stamps, end_stamps = _read_timestamps(monitor.tasks)

    # Ensure that the jobs were run after the previous job had finished
    # (e.g. begin[1] > end[0]).
    begin_after_end = (b > e
                       for b, e in zip(begin_stamps[1:], end_stamps[:-1]))
    assert all(begin_after_end)


def assert_interrupted_run(runner):
    assert 4 == runner.stats.num_cases()
    assert_runall(runner)
    assert 1 == len(runner.stats.failed())
    assert 3 == len(runner.stats.aborted())
    assert_all_dead(runner)

    # Verify that failure reasons for the different tasks are correct
    for t in runner.stats.tasks():
        if isinstance(t.check, KeyboardInterruptCheck):
            assert t.exc_info[0] == KeyboardInterrupt
        else:
            assert t.exc_info[0] == AbortTaskError


def test_kbd_interrupt_in_wait_with_concurrency(async_runner, make_cases,
                                                make_async_exec_ctx):
    ctx = make_async_exec_ctx(4)
    next(ctx)

    runner, _ = async_runner
    with pytest.raises(KeyboardInterrupt):
        runner.runall(make_cases([
            KeyboardInterruptCheck(), SleepCheck(10),
            SleepCheck(10), SleepCheck(10)
        ]))

    assert_interrupted_run(runner)


def test_kbd_interrupt_in_wait_with_limited_concurrency(
        async_runner, make_cases, make_async_exec_ctx):
    # The general idea for this test is to allow enough time for all the
    # four checks to be submitted and at the same time we need the
    # KeyboardInterruptCheck to finish first (the corresponding wait should
    # trigger the failure), so as to make the framework kill the remaining
    # three.
    ctx = make_async_exec_ctx(2)
    next(ctx)

    runner, _ = async_runner
    with pytest.raises(KeyboardInterrupt):
        runner.runall(make_cases([
            KeyboardInterruptCheck(), SleepCheck(10),
            SleepCheck(10), SleepCheck(10)
        ]))

    assert_interrupted_run(runner)


def test_kbd_interrupt_in_setup_with_concurrency(async_runner, make_cases,
                                                 make_async_exec_ctx):
    ctx = make_async_exec_ctx(4)
    next(ctx)

    runner, _ = async_runner
    with pytest.raises(KeyboardInterrupt):
        runner.runall(make_cases([
            SleepCheck(1), SleepCheck(1), SleepCheck(1),
            KeyboardInterruptCheck(phase='setup')
        ]))

    assert_interrupted_run(runner)


def test_kbd_interrupt_in_setup_with_limited_concurrency(
        async_runner, make_cases, make_async_exec_ctx):
    ctx = make_async_exec_ctx(2)
    next(ctx)

    runner, _ = async_runner
    with pytest.raises(KeyboardInterrupt):
        runner.runall(make_cases([
            SleepCheck(1), SleepCheck(1), SleepCheck(1),
            KeyboardInterruptCheck(phase='setup')
        ]))

    assert_interrupted_run(runner)


def test_run_complete_fails_main_loop(async_runner, make_cases,
                                      make_async_exec_ctx):
    ctx = make_async_exec_ctx(1)
    next(ctx)

    runner, _ = async_runner
    num_checks = 3
    runner.runall(make_cases([SleepCheckPollFail(10),
                              SleepCheck(0.1), SleepCheckPollFail(10)]))
    assert_runall(runner)
    stats = runner.stats
    assert stats.num_cases() == num_checks
    assert len(stats.failed()) == 2

    # Verify that the succeeded test is the SleepCheck
    for t in stats.tasks():
        if not t.failed:
            assert isinstance(t.check, SleepCheck)


def test_run_complete_fails_busy_loop(async_runner, make_cases,
                                      make_async_exec_ctx):
    ctx = make_async_exec_ctx(1)
    next(ctx)

    runner, _ = async_runner
    num_checks = 3
    runner.runall(make_cases([SleepCheckPollFailLate(1),
                              SleepCheck(0.1), SleepCheckPollFailLate(0.5)]))
    assert_runall(runner)
    stats = runner.stats
    assert stats.num_cases() == num_checks
    assert len(stats.failed()) == 2

    # Verify that the succeeded test is the SleepCheck
    for t in stats.tasks():
        if not t.failed:
            assert isinstance(t.check, SleepCheck)


def test_compile_fail_reschedule_main_loop(async_runner, make_cases,
                                           make_async_exec_ctx):
    ctx = make_async_exec_ctx(1)
    next(ctx)

    runner, _ = async_runner
    num_checks = 2
    runner.runall(make_cases([SleepCheckPollFail(.1), CompileFailureCheck()]))

    stats = runner.stats
    assert num_checks == stats.num_cases()
    assert_runall(runner)
    assert num_checks == len(stats.failed())


def test_compile_fail_reschedule_busy_loop(async_runner, make_cases,
                                           make_async_exec_ctx):
    ctx = make_async_exec_ctx(1)
    next(ctx)

    runner, _ = async_runner
    num_checks = 2
    runner.runall(
        make_cases([SleepCheckPollFailLate(1.5), CompileFailureCheck()])
    )
    stats = runner.stats
    assert num_checks == stats.num_cases()
    assert_runall(runner)
    assert num_checks == len(stats.failed())


@pytest.fixture
def report_file(make_runner, dep_cases, common_exec_ctx, tmp_path):
    runner = make_runner()
    runner.policy.keep_stage_files = True
    with timer() as tm:
        runner.runall(dep_cases)

    report = _generate_runreport(runner.stats.json(), *tm.timestamps())
    filename = tmp_path / 'report.json'
    with open(filename, 'w') as fp:
        jsonext.dump(report, fp)

    return filename


def test_restore_session(report_file, make_runner,
                         dep_cases, common_exec_ctx, tmp_path):
    # Select a single test to run and create the pruned graph
    selected = [tc for tc in dep_cases if tc.check.name == 'T1']
    testgraph = dependencies.prune_deps(
        dependencies.build_deps(dep_cases)[0], selected, max_depth=1
    )

    # Restore the required test cases
    report = runreport.load_report(report_file)
    testgraph, restored_cases = report.restore_dangling(testgraph)

    assert {tc.check.name for tc in restored_cases} == {'T4', 'T5'}

    # Run the selected test cases
    runner = make_runner()
    with timer() as tm:
        runner.runall(selected, restored_cases)

    new_report = _generate_runreport(runner.stats.json(), *tm.timestamps())
    assert new_report['runs'][0]['num_cases'] == 1
    assert new_report['runs'][0]['testcases'][0]['name'] == 'T1'

    # Remove the test case dump file and retry
    os.remove(tmp_path / 'stage' / 'generic' / 'default' /
              'builtin' / 'T4' / '.rfm_testcase.json')

    with pytest.raises(ReframeError, match=r'could not restore testcase'):
        report.restore_dangling(testgraph)
