# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import os
import pytest

import reframe as rfm
import reframe.core.runtime as rt
import reframe.frontend.dependencies as dependencies
import reframe.frontend.executors as executors
import reframe.frontend.executors.policies as policies
import reframe.utility.osext as osext
import unittests.utility as test_util

from reframe.core.exceptions import (AbortTaskError,
                                     FailureLimitError,
                                     ForceExitError,
                                     RunSessionTimeout)
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


def test_runall(make_runner, make_cases, common_exec_ctx, tmp_path):
    runner = make_runner()
    runner.runall(make_cases())

    assert 9 == runner.stats.num_cases()
    assert_runall(runner)
    assert 5 == len(runner.stats.failed())
    assert 2 == num_failures_stage(runner, 'setup')
    assert 1 == num_failures_stage(runner, 'sanity')
    assert 1 == num_failures_stage(runner, 'performance')
    assert 1 == num_failures_stage(runner, 'cleanup')


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
    assert_runall(runner)
    assert 11 == runner.stats.num_cases(0)
    assert 5 == runner.stats.num_cases(1)
    assert 5 == len(runner.stats.failed(0))
    assert 5 == len(runner.stats.failed(1))
    if stage.endswith('cleanup'):
        assert 0 == len(runner.stats.skipped(0))
        assert 0 == len(runner.stats.skipped(1))
    else:
        assert 2 == len(runner.stats.skipped(0))
        assert 0 == len(runner.stats.skipped(1))


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
    assert 3 == runner.stats.num_runs


def test_retries_threshold(make_runner, make_cases, common_exec_ctx):
    runner = make_runner(max_retries=2, retries_threshold=1)
    runner.runall(make_cases([BadSetupCheck(), BadSetupCheckEarly()]))
    assert 1 == runner.stats.num_runs


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


def assert_dependency_run(runner):
    assert_runall(runner)
    stats = runner.stats
    assert 10 == stats.num_cases(0)
    assert 2  == len(stats.failed())
    assert 2  == len(stats.skipped())

    # Check that cleanup is executed properly for successful tests as well
    for t in stats.tasks():
        check = t.testcase.check
        if t.failed or t.skipped:
            continue

        if t.ref_count == 0:
            assert os.path.exists(os.path.join(check.outputdir, 'out.txt'))


def test_dependencies(make_runner, cases_with_deps, common_exec_ctx):
    runner = make_runner()
    runner.runall(cases_with_deps)
    assert_dependency_run(runner)


def test_dependencies_with_retries(make_runner, cases_with_deps,
                                   common_exec_ctx):
    runner = make_runner(max_retries=2)
    runner.runall(cases_with_deps)
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
        pytest.skip('the system seems too loaded')


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
        pytest.skip('the system seems too loaded')


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
