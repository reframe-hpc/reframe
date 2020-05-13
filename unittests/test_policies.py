# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import pytest

import reframe.core.runtime as rt
import reframe.frontend.dependency as dependency
import reframe.frontend.executors as executors
import reframe.frontend.executors.policies as policies
import reframe.utility.os_ext as os_ext
from reframe.core.exceptions import (JobNotStartedError,
                                     ReframeForceExitError,
                                     TaskDependencyError)
from reframe.frontend.loader import RegressionCheckLoader

import unittests.fixtures as fixtures
from unittests.resources.checks.hellocheck import HelloTest
from unittests.resources.checks.frontend_checks import (
    BadSetupCheck,
    BadSetupCheckEarly,
    KeyboardInterruptCheck,
    RetriesCheck,
    SelfKillCheck,
    SleepCheck,
    SleepCheckPollFail,
    SleepCheckPollFailLate,
    SystemExitCheck,
)


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


@pytest.fixture(params=[policies.SerialExecutionPolicy,
                        policies.AsynchronousExecutionPolicy])
def make_runner(request):
    def _make_runner(*args, **kwargs):
        return executors.Runner(request.param(), *args, **kwargs)

    return _make_runner


@pytest.fixture
def make_cases(make_loader):
    def _make_cases(checks=None, sort=False, *args, **kwargs):
        if checks is None:
            checks = make_loader(['unittests/resources/checks']).load_all()

        cases = executors.generate_testcases(checks, *args, **kwargs)
        if sort:
            depgraph = dependency.build_deps(cases)
            dependency.validate_deps(depgraph)
            cases = dependency.toposort(depgraph)

        return cases

    return _make_cases


def assert_runall(runner):
    # Make sure that all cases finished or failed
    for t in runner.stats.tasks():
        assert t.succeeded or t.failed


def assert_all_dead(runner):
    stats = runner.stats
    for t in runner.stats.tasks():
        try:
            finished = t.check.poll()
        except JobNotStartedError:
            finished = True

        assert finished


def num_failures_stage(runner, stage):
    stats = runner.stats
    return len([t for t in stats.failures() if t.failed_stage == stage])


def test_runall(make_runner, make_cases, common_exec_ctx):
    runner = make_runner()
    runner.runall(make_cases())
    stats = runner.stats
    assert 8 == stats.num_cases()
    assert_runall(runner)
    assert 5 == len(stats.failures())
    assert 2 == num_failures_stage(runner, 'setup')
    assert 1 == num_failures_stage(runner, 'sanity')
    assert 1 == num_failures_stage(runner, 'performance')
    assert 1 == num_failures_stage(runner, 'cleanup')


def test_runall_skip_system_check(make_runner, make_cases, common_exec_ctx):
    runner = make_runner()
    runner.runall(make_cases(skip_system_check=True))
    stats = runner.stats
    assert 9 == stats.num_cases()
    assert_runall(runner)
    assert 5 == len(stats.failures())
    assert 2 == num_failures_stage(runner, 'setup')
    assert 1 == num_failures_stage(runner, 'sanity')
    assert 1 == num_failures_stage(runner, 'performance')
    assert 1 == num_failures_stage(runner, 'cleanup')


def test_runall_skip_prgenv_check(make_runner, make_cases, common_exec_ctx):
    runner = make_runner()
    runner.runall(make_cases(skip_environ_check=True))
    stats = runner.stats
    assert 9 == stats.num_cases()
    assert_runall(runner)
    assert 5 == len(stats.failures())
    assert 2 == num_failures_stage(runner, 'setup')
    assert 1 == num_failures_stage(runner, 'sanity')
    assert 1 == num_failures_stage(runner, 'performance')
    assert 1 == num_failures_stage(runner, 'cleanup')


def test_runall_skip_sanity_check(make_runner, make_cases, common_exec_ctx):
    runner = make_runner()
    runner.policy.skip_sanity_check = True
    runner.runall(make_cases())
    stats = runner.stats
    assert 8 == stats.num_cases()
    assert_runall(runner)
    assert 4 == len(stats.failures())
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
    assert 8 == stats.num_cases()
    assert_runall(runner)
    assert 4 == len(stats.failures())
    assert 2 == num_failures_stage(runner, 'setup')
    assert 1 == num_failures_stage(runner, 'sanity')
    assert 0 == num_failures_stage(runner, 'performance')
    assert 1 == num_failures_stage(runner, 'cleanup')


def test_strict_performance_check(make_runner, make_cases, common_exec_ctx):
    runner = make_runner()
    runner.policy.strict_check = True
    runner.runall(make_cases())
    stats = runner.stats
    assert 8 == stats.num_cases()
    assert_runall(runner)
    assert 6 == len(stats.failures())
    assert 2 == num_failures_stage(runner, 'setup')
    assert 1 == num_failures_stage(runner, 'sanity')
    assert 2 == num_failures_stage(runner, 'performance')
    assert 1 == num_failures_stage(runner, 'cleanup')


def test_force_local_execution(make_runner, make_cases, common_exec_ctx):
    runner = make_runner()
    runner.policy.force_local = True
    runner.runall(make_cases([HelloTest()]))
    assert_runall(runner)
    stats = runner.stats
    for t in stats.tasks():
        assert t.check.local


def test_kbd_interrupt_within_test(make_runner, make_cases, common_exec_ctx):
    runner = make_runner()
    check = KeyboardInterruptCheck()
    with pytest.raises(KeyboardInterrupt):
        runner.runall(make_cases([KeyboardInterruptCheck()]))

    stats = runner.stats
    assert 1 == len(stats.failures())
    assert_all_dead(runner)


def test_system_exit_within_test(make_runner, make_cases, common_exec_ctx):
    # This should not raise and should not exit
    runner = make_runner()
    runner.runall(make_cases([SystemExitCheck()]))
    stats = runner.stats
    assert 1 == len(stats.failures())


def test_retries_bad_check(make_runner, make_cases, common_exec_ctx):
    runner = make_runner(max_retries=2)
    runner.runall(make_cases([BadSetupCheck(), BadSetupCheckEarly()]))

    # Ensure that the test was retried #max_retries times and failed
    assert 2 == runner.stats.num_cases()
    assert_runall(runner)
    assert runner.max_retries == rt.runtime().current_run
    assert 2 == len(runner.stats.failures())

    # Ensure that the report does not raise any exception
    runner.stats.retry_report()


def test_retries_good_check(make_runner, make_cases, common_exec_ctx):
    runner = make_runner(max_retries=2)
    runner.runall(make_cases([HelloTest()]))

    # Ensure that the test passed without retries.
    assert 1 == runner.stats.num_cases()
    assert_runall(runner)
    assert 0 == rt.runtime().current_run
    assert 0 == len(runner.stats.failures())


def test_pass_in_retries(make_runner, make_cases, tmp_path, common_exec_ctx):
    tmpfile = tmp_path / 'out.txt'
    tmpfile.write_text('0\n')
    runner = make_runner(max_retries=3)
    pass_run_no = 2
    runner.runall(make_cases([RetriesCheck(pass_run_no, tmpfile)]))

    # Ensure that the test passed after retries in run `pass_run_no`
    assert 1 == runner.stats.num_cases()
    assert_runall(runner)
    assert 1 == len(runner.stats.failures(run=0))
    assert pass_run_no == rt.runtime().current_run
    assert 0 == len(runner.stats.failures())


def test_sigterm_handling(make_runner, make_cases, common_exec_ctx):
    runner = make_runner()
    with pytest.raises(ReframeForceExitError,
                       match='received TERM signal'):
        runner.runall(make_cases([SelfKillCheck()]))

    assert_all_dead(runner)
    assert runner.stats.num_cases() == 1
    assert len(runner.stats.failures()) == 1


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
    assert 4  == len(stats.failures())
    for tf in stats.failures():
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
        with os_ext.change_dir(t.check.stagedir):
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
    assert 0 == len(runner.stats.failures())

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
    assert 0 == len(runner.stats.failures())

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
    assert 0 == len(runner.stats.failures())

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
    assert 4 == len(runner.stats.failures())
    assert_all_dead(runner)


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


def test_poll_fails_in_main_loop(async_runner, make_cases,
                                 make_async_exec_ctx):
    ctx = make_async_exec_ctx(1)
    next(ctx)

    runner, _ = async_runner
    num_checks = 3
    runner.runall(make_cases([SleepCheckPollFail(10)
                              for i in range(num_checks)]))

    stats = runner.stats
    assert num_checks == stats.num_cases()
    assert_runall(runner)
    assert num_checks == len(stats.failures())


def test_poll_fails_in_busy_loop(async_runner, make_cases,
                                 make_async_exec_ctx):
    ctx = make_async_exec_ctx(1)
    next(ctx)

    runner, _ = async_runner
    num_checks = 3
    runner.runall(make_cases([SleepCheckPollFailLate(1/i)
                              for i in range(1, num_checks+1)]))

    stats = runner.stats
    assert num_checks == stats.num_cases()
    assert_runall(runner)
    assert num_checks == len(stats.failures())
