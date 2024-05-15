# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import io
import json
import jsonschema
import os
import pytest
import socket
import sys
import time

import reframe as rfm
import reframe.core.logging as logging
import reframe.core.runtime as rt
import reframe.frontend.dependencies as dependencies
import reframe.frontend.executors as executors
import reframe.frontend.executors.policies as policies
import reframe.frontend.runreport as runreport
import reframe.utility.jsonext as jsonext
import reframe.utility.osext as osext
import reframe.utility.sanity as sn
import unittests.utility as test_util

from lxml import etree
from reframe.core.exceptions import (AbortTaskError,
                                     FailureLimitError,
                                     ForceExitError,
                                     ReframeError,
                                     RunSessionTimeout,
                                     TaskDependencyError)
from reframe.frontend.loader import RegressionCheckLoader
from unittests.resources.checks.hellocheck import HelloTest
from unittests.resources.checks.frontend_checks import (
    BadSetupCheck,
    BadSetupCheckEarly,
    CompileFailureCheck,
    KeyboardInterruptCheck,
    RetriesCheck,
    SleepCheck,
    SelfKillCheck,
    SystemExitCheck
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
def make_loader():
    def _make_loader(check_search_path, *args, **kwargs):
        return RegressionCheckLoader(check_search_path, *args, **kwargs)

    return _make_loader


@pytest.fixture
def common_exec_ctx(make_exec_ctx_g):
    yield from make_exec_ctx_g(system='generic')


@pytest.fixture
def testsys_exec_ctx(make_exec_ctx_g):
    yield from make_exec_ctx_g(system='testsys:gpu')


def make_kbd_check(phase='wait'):
    return test_util.make_check(KeyboardInterruptCheck, phase=phase)


@pytest.fixture
def make_sleep_check():
    test_id = 0

    def _do_make_check(sleep_time, poll_fail=None):
        nonlocal test_id
        test = test_util.make_check(SleepCheck,
                                    sleep_time=sleep_time,
                                    poll_fail=poll_fail,
                                    alt_name=f'SleepCheck_{test_id}')
        test_id += 1
        return test

    return _do_make_check


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
            checks = make_loader(
                ['unittests/resources/checks'], *args, **kwargs
            ).load_all(force=True)

        cases = executors.generate_testcases(checks)
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
        @test_util.custom_prefix('unittests/resources/checks')
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
            hook = run_before if when == 'pre' else run_after
            check_and_skip = hook(stage)(check_and_skip)

        class _T1(rfm.RunOnlyRegressionTest):
            valid_systems = ['*']
            valid_prog_environs = ['*']
            executable = 'echo'
            sanity_patterns = sn.assert_true(1)

            def __init__(self):
                self.depends_on('_T0')

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


def _validate_junit_report(report):
    # Cloned from
    # https://raw.githubusercontent.com/windyroad/JUnit-Schema/master/JUnit.xsd
    schema_file = 'reframe/schemas/junit.xsd'
    with open(schema_file, encoding='utf-8') as fp:
        schema = etree.XMLSchema(etree.parse(fp))

    schema.assert_(report)


def _generate_runreport(run_stats, time_start, time_end):
    return {
        'session_info': {
            'cmdline': ' '.join(sys.argv),
            'config_files': rt.runtime().site_config.sources,
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

    # We explicitly set `time_total` to `None` in the last test case, in order
    # to test the proper handling of `None`.`
    report['runs'][0]['testcases'][-1]['time_total'] = None

    # Validate the junit report
    xml_report = runreport.junit_xml_report(report)
    _validate_junit_report(xml_report)

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

    # Generate a report that does not comply to the schema
    del report['session_info']['data_version']
    with open(tmp_path / 'invalid-version.json', 'w') as fp:
        jsonext.dump(report, fp)

    with pytest.raises(ReframeError,
                       match=r'invalid report'):
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
    runner.runall(make_cases(skip_prgenv_check=True))
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
    cases = make_cases()
    for c in cases:
        c.check.strict_check = True

    runner.runall(cases)
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
    test = HelloTest()
    test.local = True
    test.valid_prog_environs = ['builtin']

    runner.runall(make_cases([test]))
    assert_runall(runner)
    stats = runner.stats
    for t in stats.tasks():
        assert t.check.local

    assert not stats.failed()


def test_kbd_interrupt_within_test(make_runner, make_cases, common_exec_ctx):
    runner = make_runner()
    with pytest.raises(KeyboardInterrupt):
        runner.runall(make_cases([make_kbd_check()]))

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
    runner.runall(make_cases([
        test_util.make_check(RetriesCheck, filename=str(tmpfile), num_runs=2)
    ]))

    # Ensure that the test passed after retries in run `pass_run_no`
    assert 1 == runner.stats.num_cases()
    assert_runall(runner)
    assert 1 == len(runner.stats.failed(run=0))
    assert 2 == rt.runtime().current_run
    assert 0 == len(runner.stats.failed())


def test_sigterm_handling(make_runner, make_cases, common_exec_ctx):
    runner = make_runner()
    with pytest.raises(ForceExitError,
                       match='received TERM signal'):
        runner.runall(make_cases([SelfKillCheck()]))

    assert_all_dead(runner)
    assert runner.stats.num_cases() == 1
    assert len(runner.stats.failed()) == 1


def test_reruns(make_runner, make_cases, common_exec_ctx):
    runner = make_runner(reruns=2)
    test = HelloTest()

    runner.runall(make_cases([test]))
    stats = runner.stats
    assert stats.num_runs == 3
    assert not stats.failed(run=None)


def test_duration_limit(make_runner, make_cases, common_exec_ctx):
    runner = make_runner(timeout=2)
    test = HelloTest()

    with pytest.raises(RunSessionTimeout):
        runner.runall(make_cases([test]))
        stats = runner.stats

        assert not stats.failed(run=None)

        # A task may or may not be aborted depending on when the timeout
        # expires, but if it gets aborted, only a single test can be aborted
        # in this case for both policies.
        num_aborted = stats.aborted(run=None)
        if num_aborted:
            assert num_aborted == 1


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

    def on_task_compile(self, task):
        pass

    def on_task_compile_exit(self, task):
        pass

    def on_task_success(self, task):
        pass

    def on_task_failure(self, task):
        pass

    def on_task_abort(self, task):
        pass

    def on_task_skip(self, task):
        pass

    def on_task_setup(self, task):
        pass


def max_jobs_opts(n):
    return {'systems/partitions/max_jobs': n,
            'systems/max_local_jobs': n}


@pytest.fixture
def make_async_runner():
    # We need to have control in the unit tests where the policy is created,
    # because in some cases we need it to be initialized after the execution
    # context. For this reason, we use a constructor fixture here.

    def _make_runner():
        evt_monitor = _TaskEventMonitor()
        ret = executors.Runner(policies.AsynchronousExecutionPolicy())
        ret.policy.keep_stage_files = True
        ret.policy.task_listeners.append(evt_monitor)
        return ret, evt_monitor

    return _make_runner


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


def test_concurrency_unlimited(make_async_runner, make_cases,
                               make_sleep_check, make_exec_ctx):
    num_checks = 3
    make_exec_ctx(options=max_jobs_opts(num_checks))
    runner, monitor = make_async_runner()
    runner.runall(make_cases([make_sleep_check(.5)
                              for i in range(num_checks)]))

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


def test_concurrency_limited(make_async_runner, make_cases,
                             make_sleep_check, make_exec_ctx):
    # The number of checks must be <= 2*max_jobs.
    num_checks, max_jobs = 5, 3
    make_exec_ctx(options=max_jobs_opts(max_jobs))

    runner, monitor = make_async_runner()
    runner.runall(make_cases([make_sleep_check(.5)
                              for i in range(num_checks)]))

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

    # NOTE: to ensure that these remaining jobs were also run in parallel one
    # could do the command hereafter; however, it would require to
    # substantially increase the sleep time, because of the delays in
    # rescheduling (1s, 2s, 3s, 1s, 2s,...). We currently prefer not to do
    # this last concurrency test to avoid an important prolongation of the
    # unit test execution time. self.assertTrue(self.begin_stamps[-1] <
    # self.end_stamps[max_jobs])

    # Warn if the first #max_jobs jobs were not run in parallel; the
    # corresponding strict check would be:
    # self.assertTrue(self.begin_stamps[max_jobs-1] <= self.end_stamps[0])
    if begin_stamps[max_jobs-1] > end_stamps[0]:
        pytest.skip('the system seems too loaded.')


def test_concurrency_none(make_async_runner, make_cases,
                          make_sleep_check, make_exec_ctx):
    num_checks = 3
    make_exec_ctx(options=max_jobs_opts(1))

    runner, monitor = make_async_runner()
    runner.runall(make_cases([make_sleep_check(.5)
                              for i in range(num_checks)]))

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


def test_kbd_interrupt_in_wait_with_concurrency(
        make_async_runner, make_cases, make_sleep_check, make_exec_ctx
):
    make_exec_ctx(options=max_jobs_opts(4))
    runner, _ = make_async_runner()
    with pytest.raises(KeyboardInterrupt):
        runner.runall(make_cases([
            make_kbd_check(), make_sleep_check(10),
            make_sleep_check(10), make_sleep_check(10)
        ]))

    assert_interrupted_run(runner)


def test_kbd_interrupt_in_wait_with_limited_concurrency(
        make_async_runner, make_cases, make_sleep_check, make_exec_ctx
):
    # The general idea for this test is to allow enough time for all the
    # four checks to be submitted and at the same time we need the
    # KeyboardInterruptCheck to finish first (the corresponding wait should
    # trigger the failure), so as to make the framework kill the remaining
    # three.
    make_exec_ctx(options=max_jobs_opts(2))
    runner, _ = make_async_runner()
    with pytest.raises(KeyboardInterrupt):
        runner.runall(make_cases([
            make_kbd_check(), make_sleep_check(10),
            make_sleep_check(10), make_sleep_check(10)
        ]))

    assert_interrupted_run(runner)


def test_kbd_interrupt_in_setup_with_concurrency(
        make_async_runner, make_cases, make_sleep_check, make_exec_ctx
):
    make_exec_ctx(options=max_jobs_opts(4))
    runner, _ = make_async_runner()
    with pytest.raises(KeyboardInterrupt):
        runner.runall(make_cases([
            make_sleep_check(1), make_sleep_check(1), make_sleep_check(1),
            make_kbd_check(phase='setup')
        ]))

    assert_interrupted_run(runner)


def test_kbd_interrupt_in_setup_with_limited_concurrency(
        make_async_runner, make_sleep_check, make_cases, make_exec_ctx
):
    make_exec_ctx(options=max_jobs_opts(2))
    runner, _ = make_async_runner()
    with pytest.raises(KeyboardInterrupt):
        runner.runall(make_cases([
            make_sleep_check(1), make_sleep_check(1), make_sleep_check(1),
            make_kbd_check(phase='setup')
        ]))

    assert_interrupted_run(runner)


def test_run_complete_fails_main_loop(make_async_runner, make_cases,
                                      make_sleep_check, make_exec_ctx):
    make_exec_ctx(options=max_jobs_opts(1))
    runner, _ = make_async_runner()
    num_checks = 3
    runner.runall(make_cases([make_sleep_check(10, poll_fail='early'),
                              make_sleep_check(0.1),
                              make_sleep_check(10, poll_fail='early')]))
    assert_runall(runner)
    stats = runner.stats
    assert stats.num_cases() == num_checks
    assert len(stats.failed()) == 2

    # Verify that the succeeded test is a SleepCheck
    for t in stats.tasks():
        if not t.failed:
            assert t.check.name.startswith('SleepCheck')


def test_run_complete_fails_busy_loop(make_async_runner, make_cases,
                                      make_sleep_check, make_exec_ctx):
    make_exec_ctx(options=max_jobs_opts(1))
    runner, _ = make_async_runner()
    num_checks = 3
    runner.runall(make_cases([make_sleep_check(1, poll_fail='late'),
                              make_sleep_check(0.1),
                              make_sleep_check(0.5, poll_fail='late')]))
    assert_runall(runner)
    stats = runner.stats
    assert stats.num_cases() == num_checks
    assert len(stats.failed()) == 2

    # Verify that the succeeded test is a SleepCheck
    for t in stats.tasks():
        if not t.failed:
            assert t.check.name.startswith('SleepCheck')


def test_compile_fail_reschedule_main_loop(make_async_runner, make_cases,
                                           make_sleep_check, make_exec_ctx):
    make_exec_ctx(options=max_jobs_opts(1))
    runner, _ = make_async_runner()
    num_checks = 2
    runner.runall(make_cases([make_sleep_check(.1, poll_fail='early'),
                              CompileFailureCheck()]))

    stats = runner.stats
    assert num_checks == stats.num_cases()
    assert_runall(runner)
    assert num_checks == len(stats.failed())


def test_compile_fail_reschedule_busy_loop(make_async_runner, make_cases,
                                           make_sleep_check, make_exec_ctx):
    make_exec_ctx(options=max_jobs_opts(1))
    runner, _ = make_async_runner()
    num_checks = 2
    runner.runall(
        make_cases([make_sleep_check(1.5, poll_fail='late'),
                    CompileFailureCheck()])
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

    # Generate an empty report and load it as primary with the original report
    # as a fallback, in order to test if the dependencies are still resolved
    # correctly
    empty_report = tmp_path / 'empty.json'

    with open(empty_report, 'w') as fp:
        empty_run = [
            {
                'num_cases': 0,
                'num_failures': 0,
                'num_aborted': 0,
                'num_skipped': 0,
                'runid': 0,
                'testcases': []
            }
        ]
        jsonext.dump(_generate_runreport(empty_run, *tm.timestamps()), fp)

    report2 = runreport.load_report(empty_report, report_file)
    restored_cases = report2.restore_dangling(testgraph)[1]
    assert {tc.check.name for tc in restored_cases} == {'T4', 'T5'}

    # Remove the test case dump file and retry
    os.remove(tmp_path / 'stage' / 'generic' / 'default' /
              'builtin' / 'T4' / '.rfm_testcase.json')

    with pytest.raises(ReframeError, match=r'could not restore testcase'):
        report.restore_dangling(testgraph)


def test_config_params(make_runner, make_exec_ctx):
    '''Test that configuration parameters are properly retrieved with the
    various execution policies.
    '''

    class T(rfm.RunOnlyRegressionTest):
        valid_systems = ['sys2']
        valid_prog_environs = ['*']
        executable = 'echo'

        @sanity_function
        def validate(self):
            return True

        @run_after('setup')
        def assert_git_timeout(self):
            expected = 10 if self.current_partition.name == 'part1' else 20
            timeout = rt.runtime().get_option('general/0/git_timeout')
            assert timeout == expected

    make_exec_ctx(system='sys2')
    runner = make_runner()
    testcases = executors.generate_testcases([T()])
    runner.runall(testcases)
    assert runner.stats.num_cases() == 2
    assert not runner.stats.failed()


class _MyPerfTest(rfm.RunOnlyRegressionTest):
    valid_systems = ['*']
    valid_prog_environs = ['*']
    executable = 'echo perf0=100 && echo perf1=50'

    @sanity_function
    def validate(self):
        return sn.assert_found(r'perf0', self.stdout)

    @performance_function('unit0')
    def perf0(self):
        return sn.extractsingle(r'perf0=(\S+)', self.stdout, 1, float)

    @performance_function('unit1')
    def perf1(self):
        return sn.extractsingle(r'perf1=(\S+)', self.stdout, 1, float)


class _MyPerfParamTest(_MyPerfTest):
    p = parameter([1, 2])


class _MyFailingTest(rfm.RunOnlyRegressionTest):
    valid_systems = ['*']
    valid_prog_environs = ['*']
    executable = 'echo perf0=100'

    @sanity_function
    def validate(self):
        return False

    @performance_function('unit0')
    def perf0(self):
        return sn.extractsingle(r'perf0=(\S+)', self.stdout, 1, float)


class _LazyPerfTest(rfm.RunOnlyRegressionTest):
    valid_systems = ['*']
    valid_prog_environs = ['*']
    executable = 'echo perf0=100'

    @sanity_function
    def validate(self):
        return True

    @run_before('performance')
    def set_perf_vars(self):
        self.perf_variables = {
            'perf0': sn.make_performance_function(
                sn.extractsingle(r'perf0=(\S+)', self.stdout, 1, float),
                'unit0'
            )
        }


@pytest.fixture
def perf_test():
    return _MyPerfTest()


@pytest.fixture
def perf_param_tests():
    return [_MyPerfParamTest(variant_num=v)
            for v in range(_MyPerfParamTest.num_variants)]


@pytest.fixture
def failing_perf_test():
    return _MyFailingTest()


@pytest.fixture
def lazy_perf_test():
    return _LazyPerfTest()


@pytest.fixture
def simple_test():
    class _MySimpleTest(rfm.RunOnlyRegressionTest):
        valid_systems = ['*']
        valid_prog_environs = ['*']
        executable = 'echo hello'

        @sanity_function
        def validate(self):
            return sn.assert_found(r'hello', self.stdout)

    return _MySimpleTest()


@pytest.fixture
def config_perflog(make_config_file):
    def _config_perflog(fmt, perffmt=None, logging_opts=None):
        logging_config = {
            'level': 'debug2',
            'handlers': [{
                'type': 'stream',
                'name': 'stdout',
                'level': 'info',
                'format': '%(message)s'
            }],
            'handlers_perflog': [{
                'type': 'filelog',
                'prefix': '%(check_system)s/%(check_partition)s',
                'level': 'info',
                'format': fmt
            }]
        }
        if logging_opts:
            logging_config.update(logging_opts)

        if perffmt is not None:
            logging_config['handlers_perflog'][0]['format_perfvars'] = perffmt

        return make_config_file({'logging': [logging_config]})

    return _config_perflog


def _count_lines(filepath):
    count = 0
    with open(filepath) as fp:
        for line in fp:
            count += 1

    return count


def _assert_header(filepath, header):
    with open(filepath) as fp:
        assert fp.readline().strip() == header


def _assert_no_logging_error(fn, *args, **kwargs):
    captured_stderr = io.StringIO()
    with contextlib.redirect_stderr(captured_stderr):
        fn(*args, **kwargs)

    assert 'Logging error' not in captured_stderr.getvalue()


def test_perf_logging(make_runner, make_exec_ctx, perf_test,
                      config_perflog, tmp_path):
    make_exec_ctx(
        config_perflog(
            fmt=(
                '%(check_job_completion_time)s,%(version)s,'
                '%(check_display_name)s,%(check_system)s,'
                '%(check_partition)s,%(check_environ)s,'
                '%(check_jobid)s,%(check_result)s,%(check_perfvalues)s'
            ),
            perffmt=(
                '%(check_perf_value)s,%(check_perf_unit)s,'
                '%(check_perf_ref)s,%(check_perf_lower_thres)s,'
                '%(check_perf_upper_thres)s,'
            )
        )
    )
    logging.configure_logging(rt.runtime().site_config)
    runner = make_runner()
    testcases = executors.generate_testcases([perf_test])
    runner.runall(testcases)

    logfile = tmp_path / 'perflogs' / 'generic' / 'default' / '_MyPerfTest.log'
    assert os.path.exists(logfile)
    assert _count_lines(logfile) == 2

    # Rerun with the same configuration and check that new entry is appended
    testcases = executors.generate_testcases([perf_test])
    runner = make_runner()
    _assert_no_logging_error(runner.runall, testcases)
    assert _count_lines(logfile) == 3

    # Change the configuration and rerun
    make_exec_ctx(
        config_perflog(
            fmt=(
                '%(check_job_completion_time)s,%(version)s,'
                '%(check_display_name)s,%(check_system)s,'
                '%(check_partition)s,%(check_environ)s,'
                '%(check_jobid)s,%(check_result)s,%(check_perfvalues)s'
            ),
            perffmt='%(check_perf_value)s,%(check_perf_unit)s,'
        )
    )
    logging.configure_logging(rt.runtime().site_config)
    testcases = executors.generate_testcases([perf_test])
    runner = make_runner()
    _assert_no_logging_error(runner.runall, testcases)
    assert _count_lines(logfile) == 2
    _assert_header(logfile,
                   'job_completion_time,version,display_name,system,partition,'
                   'environ,jobid,result,perf0_value,perf0_unit,'
                   'perf1_value,perf1_unit')

    logfile_prev = [(str(logfile) + '.h0', 3)]
    for f, num_lines in logfile_prev:
        assert os.path.exists(f)
        _count_lines(f) == num_lines

    # Change the test and rerun
    perf_test.perf_variables['perfN'] = perf_test.perf_variables['perf1']

    # We reconfigure the logging in order for the filelog handler to start
    # from a clean state
    logging.configure_logging(rt.runtime().site_config)
    testcases = executors.generate_testcases([perf_test])
    runner = make_runner()
    _assert_no_logging_error(runner.runall, testcases)
    assert _count_lines(logfile) == 2
    _assert_header(logfile,
                   'job_completion_time,version,display_name,system,partition,'
                   'environ,jobid,result,perf0_value,perf0_unit,'
                   'perf1_value,perf1_unit,perfN_value,perfN_unit')

    logfile_prev = [(str(logfile) + '.h0', 3), (str(logfile) + '.h1', 2)]
    for f, num_lines in logfile_prev:
        assert os.path.exists(f)
        _count_lines(f) == num_lines


def test_perf_logging_no_end_delim(make_runner, make_exec_ctx, perf_test,
                                   config_perflog, tmp_path):
    make_exec_ctx(
        config_perflog(
            fmt=(
                '%(check_job_completion_time)s,%(version)s,'
                '%(check_display_name)s,%(check_system)s,'
                '%(check_partition)s,%(check_environ)s,'
                '%(check_jobid)s,%(check_result)s,%(check_perfvalues)s'
            ),
            perffmt='%(check_perf_value)s,%(check_perf_unit)s'
        )
    )
    logging.configure_logging(rt.runtime().site_config)
    runner = make_runner()
    testcases = executors.generate_testcases([perf_test])
    _assert_no_logging_error(runner.runall, testcases)

    logfile = tmp_path / 'perflogs' / 'generic' / 'default' / '_MyPerfTest.log'
    assert os.path.exists(logfile)
    assert _count_lines(logfile) == 2

    with open(logfile) as fp:
        lines = fp.readlines()

    assert len(lines) == 2
    assert lines[0] == (
        'job_completion_time,version,display_name,system,partition,'
        'environ,jobid,result,perf0_value,perf0_unitperf1_value,perf1_unit\n'
    )
    assert '<error formatting the performance record' in lines[1]


def test_perf_logging_no_perfvars(make_runner, make_exec_ctx, perf_test,
                                  config_perflog, tmp_path):
    make_exec_ctx(
        config_perflog(
            fmt=(
                '%(check_job_completion_time)s,%(version)s,'
                '%(check_display_name)s,%(check_system)s,'
                '%(check_partition)s,%(check_environ)s,'
                '%(check_jobid)s,%(check_result)s,%(check_perfvalues)s'
            )
        )
    )
    logging.configure_logging(rt.runtime().site_config)
    runner = make_runner()
    testcases = executors.generate_testcases([perf_test])
    _assert_no_logging_error(runner.runall, testcases)

    logfile = tmp_path / 'perflogs' / 'generic' / 'default' / '_MyPerfTest.log'
    assert os.path.exists(logfile)
    assert _count_lines(logfile) == 2

    with open(logfile) as fp:
        lines = fp.readlines()

    assert len(lines) == 2
    assert lines[0] == (
        'job_completion_time,version,display_name,system,partition,'
        'environ,jobid,result,\n'
    )
    assert 'error' not in lines[1]


def test_perf_logging_multiline(make_runner, make_exec_ctx, perf_test,
                                simple_test, failing_perf_test,
                                config_perflog, tmp_path):
    make_exec_ctx(
        config_perflog(
            fmt=(
                '%(check_job_completion_time)s,%(version)s,'
                '%(check_display_name)s,%(check_system)s,'
                '%(check_partition)s,%(check_environ)s,'
                '%(check_jobid)s,%(check_result)s,'
                '%(check_perf_var)s=%(check_perf_value)s,%(check_perf_unit)s'
            ),
            logging_opts={'perflog_compat': True}
        )
    )
    logging.configure_logging(rt.runtime().site_config)
    runner = make_runner()
    testcases = executors.generate_testcases(
        [perf_test, simple_test, failing_perf_test]
    )
    _assert_no_logging_error(runner.runall, testcases)

    logfile = tmp_path / 'perflogs' / 'generic' / 'default' / '_MyPerfTest.log'
    assert os.path.exists(logfile)
    assert _count_lines(logfile) == 3

    # assert that the emitted lines are correct
    with open(logfile) as fp:
        lines = fp.readlines()
        assert ',perf0=100.0,unit0' in lines[1]
        assert ',perf1=50.0,unit1'  in lines[2]


def test_perf_logging_lazy(make_runner, make_exec_ctx, lazy_perf_test,
                           config_perflog, tmp_path):
    make_exec_ctx(
        config_perflog(
            fmt=(
                '%(check_job_completion_time)s,%(version)s,'
                '%(check_display_name)s,%(check_system)s,'
                '%(check_partition)s,%(check_environ)s,'
                '%(check_jobid)s,%(check_result)s,%(check_perfvalues)s'
            ),
            perffmt=(
                '%(check_perf_value)s,%(check_perf_unit)s,'
                '%(check_perf_ref)s,%(check_perf_lower_thres)s,'
                '%(check_perf_upper_thres)s,'
            )
        )
    )
    logging.configure_logging(rt.runtime().site_config)
    runner = make_runner()
    testcases = executors.generate_testcases([lazy_perf_test])
    _assert_no_logging_error(runner.runall, testcases)

    logfile = tmp_path / 'perflogs' / 'generic' / 'default' / '_LazyPerfTest.log'
    assert os.path.exists(logfile)


def test_perf_logging_all_attrs(make_runner, make_exec_ctx, perf_test,
                                config_perflog, tmp_path):
    make_exec_ctx(config_perflog(fmt='%(check_result)s|%(check_#ALL)s'))
    logging.configure_logging(rt.runtime().site_config)
    runner = make_runner()
    testcases = executors.generate_testcases([perf_test])
    _assert_no_logging_error(runner.runall, testcases)

    logfile = tmp_path / 'perflogs' / 'generic' / 'default' / '_MyPerfTest.log'
    assert os.path.exists(logfile)
    with open(logfile) as fp:
        header = fp.readline()

    loggable_attrs = type(perf_test).loggable_attrs()
    assert len(header.split('|')) == len(loggable_attrs) + 1


def test_perf_logging_custom_vars(make_runner, make_exec_ctx,
                                  config_perflog, tmp_path):
    # Create two tests with different loggable variables
    class _X(_MyPerfTest):
        x = variable(int, value=1, loggable=True)

    class _Y(_MyPerfTest):
        y = variable(int, value=2, loggable=True)

    make_exec_ctx(config_perflog(fmt='%(check_result)s|%(check_#ALL)s'))
    logging.configure_logging(rt.runtime().site_config)
    runner = make_runner()
    testcases = executors.generate_testcases([_X(), _Y()])
    _assert_no_logging_error(runner.runall, testcases)

    logfiles = [tmp_path / 'perflogs' / 'generic' / 'default' / '_X.log',
                tmp_path / 'perflogs' / 'generic' / 'default' / '_Y.log']
    for f in logfiles:
        with open(f) as fp:
            header = fp.readline().strip()
            if os.path.basename(f).startswith('_X'):
                assert 'x' in header.split('|')
            else:
                assert 'y' in header.split('|')


def test_perf_logging_param_test(make_runner, make_exec_ctx, perf_param_tests,
                                 config_perflog, tmp_path):
    make_exec_ctx(config_perflog(fmt='%(check_result)s|%(check_#ALL)s'))
    logging.configure_logging(rt.runtime().site_config)
    runner = make_runner()
    testcases = executors.generate_testcases(perf_param_tests)
    _assert_no_logging_error(runner.runall, testcases)

    logfile = (tmp_path / 'perflogs' / 'generic' /
               'default' / '_MyPerfParamTest.log')
    assert os.path.exists(logfile)
    assert _count_lines(logfile) == 3


def test_perf_logging_sanity_failure(make_runner, make_exec_ctx,
                                     config_perflog, tmp_path):
    class _X(_MyPerfTest):
        @sanity_function
        def validate(self):
            return sn.assert_true(0, msg='no way')

    make_exec_ctx(config_perflog(
        fmt='%(check_result)s|%(check_fail_reason)s|%(check_perfvalues)s',
        perffmt='%(check_perf_value)s|'
    ))
    logging.configure_logging(rt.runtime().site_config)
    runner = make_runner()
    testcases = executors.generate_testcases([_X()])
    _assert_no_logging_error(runner.runall, testcases)

    logfile = tmp_path / 'perflogs' / 'generic' / 'default' / '_X.log'
    assert os.path.exists(logfile)
    with open(logfile) as fp:
        lines = fp.readlines()

    assert len(lines) == 2
    assert lines[1] == 'fail|sanity error: no way|None|None\n'
