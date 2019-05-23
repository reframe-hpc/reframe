import collections
import os
import pytest
import tempfile
import unittest

import reframe as rfm
import reframe.core.runtime as rt
import reframe.frontend.dependency as dependency
import reframe.frontend.executors as executors
import reframe.frontend.executors.policies as policies
import reframe.utility.os_ext as os_ext
from reframe.core.exceptions import DependencyError, JobNotStartedError
from reframe.frontend.loader import RegressionCheckLoader
import unittests.fixtures as fixtures
from unittests.resources.checks.hellocheck import HelloTest
from unittests.resources.checks.frontend_checks import (
    BadSetupCheck,
    BadSetupCheckEarly,
    KeyboardInterruptCheck,
    RetriesCheck,
    SleepCheck,
    SleepCheckPollFail,
    SleepCheckPollFailLate,
    SystemExitCheck,
)


class TestSerialExecutionPolicy(unittest.TestCase):
    def setUp(self):
        self.loader = RegressionCheckLoader(['unittests/resources/checks'],
                                            ignore_conflicts=True)

        # Setup the runner
        self.runner = executors.Runner(policies.SerialExecutionPolicy())
        self.checks = self.loader.load_all()

        # Set runtime prefix
        rt.runtime().resources.prefix = tempfile.mkdtemp(dir='unittests')

        # Reset current_run
        rt.runtime()._current_run = 0

    def tearDown(self):
        os_ext.rmtree(rt.runtime().resources.prefix)

    def runall(self, checks, *args, **kwargs):
        cases = executors.generate_testcases(checks, *args, **kwargs)
        self.runner.runall(cases)

    def _num_failures_stage(self, stage):
        stats = self.runner.stats
        return len([t for t in stats.failures() if t.failed_stage == stage])

    def assert_all_dead(self):
        stats = self.runner.stats
        for t in self.runner.stats.tasks():
            try:
                finished = t.check.poll()
            except JobNotStartedError:
                finished = True

            self.assertTrue(finished)

    def test_runall(self):
        self.runall(self.checks)

        stats = self.runner.stats
        self.assertEqual(7, stats.num_cases())
        self.assertEqual(4, len(stats.failures()))
        self.assertEqual(2, self._num_failures_stage('setup'))
        self.assertEqual(1, self._num_failures_stage('sanity'))
        self.assertEqual(1, self._num_failures_stage('performance'))

    def test_runall_skip_system_check(self):
        self.runall(self.checks, skip_system_check=True)

        stats = self.runner.stats
        self.assertEqual(8, stats.num_cases())
        self.assertEqual(4, len(stats.failures()))
        self.assertEqual(2, self._num_failures_stage('setup'))
        self.assertEqual(1, self._num_failures_stage('sanity'))
        self.assertEqual(1, self._num_failures_stage('performance'))

    def test_runall_skip_prgenv_check(self):
        self.runall(self.checks, skip_environ_check=True)

        stats = self.runner.stats
        self.assertEqual(8, stats.num_cases())
        self.assertEqual(4, len(stats.failures()))
        self.assertEqual(2, self._num_failures_stage('setup'))
        self.assertEqual(1, self._num_failures_stage('sanity'))
        self.assertEqual(1, self._num_failures_stage('performance'))

    def test_runall_skip_sanity_check(self):
        self.runner.policy.skip_sanity_check = True
        self.runall(self.checks)

        stats = self.runner.stats
        self.assertEqual(7, stats.num_cases())
        self.assertEqual(3, len(stats.failures()))
        self.assertEqual(2, self._num_failures_stage('setup'))
        self.assertEqual(0, self._num_failures_stage('sanity'))
        self.assertEqual(1, self._num_failures_stage('performance'))

    def test_runall_skip_performance_check(self):
        self.runner.policy.skip_performance_check = True
        self.runall(self.checks)

        stats = self.runner.stats
        self.assertEqual(7, stats.num_cases())
        self.assertEqual(3, len(stats.failures()))
        self.assertEqual(2, self._num_failures_stage('setup'))
        self.assertEqual(1, self._num_failures_stage('sanity'))
        self.assertEqual(0, self._num_failures_stage('performance'))

    def test_strict_performance_check(self):
        self.runner.policy.strict_check = True
        self.runall(self.checks)

        stats = self.runner.stats
        self.assertEqual(7, stats.num_cases())
        self.assertEqual(5, len(stats.failures()))
        self.assertEqual(2, self._num_failures_stage('setup'))
        self.assertEqual(1, self._num_failures_stage('sanity'))
        self.assertEqual(2, self._num_failures_stage('performance'))

    def test_force_local_execution(self):
        self.runner.policy.force_local = True
        self.runall([HelloTest()])
        stats = self.runner.stats
        for t in stats.tasks():
            self.assertTrue(t.check.local)

    def test_kbd_interrupt_within_test(self):
        check = KeyboardInterruptCheck()
        self.assertRaises(KeyboardInterrupt, self.runall, [check])
        stats = self.runner.stats
        self.assertEqual(1, len(stats.failures()))
        self.assert_all_dead()

    def test_system_exit_within_test(self):
        check = SystemExitCheck()

        # This should not raise and should not exit
        self.runall([check])
        stats = self.runner.stats
        self.assertEqual(1, len(stats.failures()))

    def test_retries_bad_check(self):
        max_retries = 2
        checks = [BadSetupCheck(), BadSetupCheckEarly()]
        self.runner._max_retries = max_retries
        self.runall(checks)

        # Ensure that the test was retried #max_retries times and failed.
        self.assertEqual(2, self.runner.stats.num_cases())
        self.assertEqual(max_retries, rt.runtime().current_run)
        self.assertEqual(2, len(self.runner.stats.failures()))

        # Ensure that the report does not raise any exception.
        self.runner.stats.retry_report()

    def test_retries_good_check(self):
        max_retries = 2
        checks = [HelloTest()]
        self.runner._max_retries = max_retries
        self.runall(checks)

        # Ensure that the test passed without retries.
        self.assertEqual(1, self.runner.stats.num_cases())
        self.assertEqual(0, rt.runtime().current_run)
        self.assertEqual(0, len(self.runner.stats.failures()))

    def test_pass_in_retries(self):
        max_retries = 3
        run_to_pass = 2
        # Create a file containing the current_run; Run 0 will set it to 0,
        # run 1 to 1 and so on.
        with tempfile.NamedTemporaryFile(mode='wt', delete=False) as fp:
            fp.write('0\n')

        checks = [RetriesCheck(run_to_pass, fp.name)]
        self.runner._max_retries = max_retries
        self.runall(checks)

        # Ensure that the test passed after retries in run #run_to_pass.
        self.assertEqual(1, self.runner.stats.num_cases())
        self.assertEqual(1, len(self.runner.stats.failures(run=0)))
        self.assertEqual(run_to_pass, rt.runtime().current_run)
        self.assertEqual(0, len(self.runner.stats.failures()))
        os.remove(fp.name)


class TaskEventMonitor(executors.TaskEventListener):
    """Event listener for monitoring the execution of the asynchronous
    execution policy.

    We need to make sure two things for the async policy:

    1. The number of running tasks never exceed the max job size per partition.
    2. Given a set of regression tests with a reasonably long runtime, the
       execution policy must be able to reach the maximum concurrency. By
       reasonably long runtime, we mean that that the regression tests must run
       enough time, so as to allow the policy to execute all the tests until
       their "run" phase, before the first submitted test finishes.
    """

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


class TestAsynchronousExecutionPolicy(TestSerialExecutionPolicy):
    def setUp(self):
        super().setUp()
        self.runner = executors.Runner(policies.AsynchronousExecutionPolicy())
        self.runner.policy.keep_stage_files = True
        self.monitor = TaskEventMonitor()
        self.runner.policy.task_listeners.append(self.monitor)

    def set_max_jobs(self, value):
        for p in rt.runtime().system.partitions:
            p._max_jobs = value

    def read_timestamps(self, tasks):
        """Read the timestamps and sort them to permit simple
        concurrency tests."""
        from reframe.core.deferrable import evaluate

        self.begin_stamps = []
        self.end_stamps = []
        for t in tasks:
            with os_ext.change_dir(t.check.stagedir):
                with open(evaluate(t.check.stdout), 'r') as f:
                    self.begin_stamps.append(float(f.readline().strip()))
                    self.end_stamps.append(float(f.readline().strip()))

        self.begin_stamps.sort()
        self.end_stamps.sort()

    def test_concurrency_unlimited(self):
        checks = [SleepCheck(0.5) for i in range(3)]
        self.set_max_jobs(len(checks))
        self.runall(checks)

        # Ensure that all tests were run and without failures.
        self.assertEqual(len(checks), self.runner.stats.num_cases())
        self.assertEqual(0, len(self.runner.stats.failures()))

        # Ensure that maximum concurrency was reached as fast as possible
        self.assertEqual(len(checks), max(self.monitor.num_tasks))
        self.assertEqual(len(checks), self.monitor.num_tasks[len(checks)])

        self.read_timestamps(self.monitor.tasks)

        # Warn if not all tests were run in parallel; the corresponding strict
        # check would be:
        #
        #     self.assertTrue(self.begin_stamps[-1] <= self.end_stamps[0])
        #
        if self.begin_stamps[-1] > self.end_stamps[0]:
            self.skipTest('the system seems too much loaded.')

    def test_concurrency_limited(self):
        # The number of checks must be <= 2*max_jobs.
        checks = [SleepCheck(0.5) for i in range(5)]
        max_jobs = len(checks) - 2
        self.set_max_jobs(max_jobs)
        self.runall(checks)

        # Ensure that all tests were run and without failures.
        self.assertEqual(len(checks), self.runner.stats.num_cases())
        self.assertEqual(0, len(self.runner.stats.failures()))

        # Ensure that maximum concurrency was reached as fast as possible
        self.assertEqual(max_jobs, max(self.monitor.num_tasks))
        self.assertEqual(max_jobs, self.monitor.num_tasks[max_jobs])

        self.read_timestamps(self.monitor.tasks)

        # Ensure that the jobs after the first #max_jobs were each run after
        # one of the previous #max_jobs jobs had finished
        # (e.g. begin[max_jobs] > end[0]).
        # Note: we may ensure this strictly as we may ensure serial behaviour.
        begin_after_end = (b > e for b, e in zip(self.begin_stamps[max_jobs:],
                                                 self.end_stamps[:-max_jobs]))
        self.assertTrue(all(begin_after_end))

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
        if self.begin_stamps[max_jobs-1] > self.end_stamps[0]:
            self.skipTest('the system seems too loaded.')

    def test_concurrency_none(self):
        checks = [SleepCheck(0.5) for i in range(3)]
        num_checks = len(checks)
        self.set_max_jobs(1)
        self.runall(checks)

        # Ensure that all tests were run and without failures.
        self.assertEqual(len(checks), self.runner.stats.num_cases())
        self.assertEqual(0, len(self.runner.stats.failures()))

        # Ensure that a single task was running all the time
        self.assertEqual(1, max(self.monitor.num_tasks))

        # Read the timestamps sorted to permit simple concurrency tests.
        self.read_timestamps(self.monitor.tasks)

        # Ensure that the jobs were run after the previous job had finished
        # (e.g. begin[1] > end[0]).
        begin_after_end = (b > e for b, e in zip(self.begin_stamps[1:],
                                                 self.end_stamps[:-1]))
        self.assertTrue(all(begin_after_end))

    def _run_checks(self, checks, max_jobs):
        self.set_max_jobs(max_jobs)
        self.assertRaises(KeyboardInterrupt, self.runall, checks)

        self.assertEqual(4, self.runner.stats.num_cases())
        self.assertEqual(4, len(self.runner.stats.failures()))
        self.assert_all_dead()

    def test_kbd_interrupt_in_wait_with_concurrency(self):
        checks = [KeyboardInterruptCheck(),
                  SleepCheck(10), SleepCheck(10), SleepCheck(10)]
        self._run_checks(checks, 4)

    def test_kbd_interrupt_in_wait_with_limited_concurrency(self):
        # The general idea for this test is to allow enough time for all the
        # four checks to be submitted and at the same time we need the
        # KeyboardInterruptCheck to finish first (the corresponding wait should
        # trigger the failure), so as to make the framework kill the remaining
        # three.
        checks = [KeyboardInterruptCheck(),
                  SleepCheck(10), SleepCheck(10), SleepCheck(10)]
        self._run_checks(checks, 2)

    def test_kbd_interrupt_in_setup_with_concurrency(self):
        checks = [SleepCheck(1), SleepCheck(1), SleepCheck(1),
                  KeyboardInterruptCheck(phase='setup')]
        self._run_checks(checks, 4)

    def test_kbd_interrupt_in_setup_with_limited_concurrency(self):
        checks = [SleepCheck(1), SleepCheck(1), SleepCheck(1),
                  KeyboardInterruptCheck(phase='setup')]
        self._run_checks(checks, 2)

    def test_poll_fails_main_loop(self):
        num_tasks = 3
        checks = [SleepCheckPollFail(10) for i in range(num_tasks)]
        num_checks = len(checks)
        self.set_max_jobs(1)
        self.runall(checks)
        stats = self.runner.stats
        self.assertEqual(num_tasks, stats.num_cases())
        self.assertEqual(num_tasks, len(stats.failures()))

    def test_poll_fails_busy_loop(self):
        num_tasks = 3
        checks = [SleepCheckPollFailLate(1/i) for i in range(1, num_tasks+1)]
        num_checks = len(checks)
        self.set_max_jobs(1)
        self.runall(checks)
        stats = self.runner.stats
        self.assertEqual(num_tasks, stats.num_cases())
        self.assertEqual(num_tasks, len(stats.failures()))


class TestDependencies(unittest.TestCase):
    class Node:
        """A node in the test case graph.

        It's simply a wrapper to a (test_name, partition, environment) tuple
        that can interact seemlessly with a real test case.
        It's meant for convenience in unit testing.
        """

        def __init__(self, cname, pname, ename):
            self.cname, self.pname, self.ename = cname, pname, ename

        def __eq__(self, other):
            if isinstance(other, type(self)):
                return (self.cname == other.cname and
                        self.pname == other.pname and
                        self.ename == other.ename)

            if isinstance(other, executors.TestCase):
                return (self.cname == other.check.name and
                        self.pname == other.partition.fullname and
                        self.ename == other.environ.name)

            return NotImplemented

        def __hash__(self):
            return hash(self.cname) ^ hash(self.pname) ^ hash(self.ename)

        def __repr__(self):
            return 'Node(%r, %r, %r)' % (self.cname, self.pname, self.ename)

    def has_edge(graph, src, dst):
        return dst in graph[src]

    def num_deps(graph, cname):
        return sum(len(deps) for c, deps in graph.items()
                   if c.check.name == cname)

    def find_check(name, checks):
        for c in checks:
            if c.name == name:
                return c

        return None

    def find_case(cname, ename, cases):
        for c in cases:
            if c.check.name == cname and c.environ.name == ename:
                return c

    def setUp(self):
        self.loader = RegressionCheckLoader([
            'unittests/resources/checks_unlisted/dependencies/normal.py'
        ])

        # Set runtime prefix
        rt.runtime().resources.prefix = tempfile.mkdtemp(dir='unittests')

    def tearDown(self):
        os_ext.rmtree(rt.runtime().resources.prefix)

    @rt.switch_runtime(fixtures.TEST_SITE_CONFIG, 'sys0')
    def test_eq_hash(self):
        find_case = TestDependencies.find_case
        cases = executors.generate_testcases(self.loader.load_all())

        case0 = find_case('Test0', 'e0', cases)
        case1 = find_case('Test0', 'e1', cases)
        case0_copy = case0.clone()

        assert case0 == case0_copy
        assert hash(case0) == hash(case0_copy)
        assert case1 != case0
        assert hash(case1) != hash(case0)

    @rt.switch_runtime(fixtures.TEST_SITE_CONFIG, 'sys0')
    def test_build_deps(self):
        Node = TestDependencies.Node
        has_edge = TestDependencies.has_edge
        num_deps = TestDependencies.num_deps
        find_check = TestDependencies.find_check
        find_case = TestDependencies.find_case

        checks = self.loader.load_all()
        cases = executors.generate_testcases(checks)

        # Test calling getdep() before having built the graph
        t = find_check('Test1_exact', checks)
        with pytest.raises(DependencyError):
            t.getdep('Test0', 'e0')

        # Build dependencies and continue testing
        deps = dependency.build_deps(cases)
        dependency.validate_deps(deps)

        # Check DEPEND_FULLY dependencies
        assert num_deps(deps, 'Test1_fully') == 8
        for p in ['sys0:p0', 'sys0:p1']:
            for e0 in ['e0', 'e1']:
                for e1 in ['e0', 'e1']:
                    assert has_edge(deps,
                                    Node('Test1_fully', p, e0),
                                    Node('Test0', p, e1))

        # Check DEPEND_BY_ENV
        assert num_deps(deps, 'Test1_by_env') == 4
        assert num_deps(deps, 'Test1_default') == 4
        for p in ['sys0:p0', 'sys0:p1']:
            for e in ['e0', 'e1']:
                assert has_edge(deps,
                                Node('Test1_by_env', p, e),
                                Node('Test0', p, e))
                assert has_edge(deps,
                                Node('Test1_default', p, e),
                                Node('Test0', p, e))

        # Check DEPEND_EXACT
        assert num_deps(deps, 'Test1_exact') == 6
        for p in ['sys0:p0', 'sys0:p1']:
            assert has_edge(deps,
                            Node('Test1_exact', p, 'e0'),
                            Node('Test0', p, 'e0'))
            assert has_edge(deps,
                            Node('Test1_exact', p, 'e0'),
                            Node('Test0', p, 'e1'))
            assert has_edge(deps,
                            Node('Test1_exact', p, 'e1'),
                            Node('Test0', p, 'e1'))

        # Pick a check to test getdep()
        check_e0 = find_case('Test1_exact', 'e0', cases).check
        check_e1 = find_case('Test1_exact', 'e1', cases).check
        assert check_e0.getdep('Test0', 'e0').name == 'Test0'
        assert check_e0.getdep('Test0', 'e1').name == 'Test0'
        assert check_e1.getdep('Test0', 'e1').name == 'Test0'
        with pytest.raises(DependencyError):
            check_e0.getdep('TestX', 'e0')

        with pytest.raises(DependencyError):
            check_e0.getdep('Test0', 'eX')

        with pytest.raises(DependencyError):
            check_e1.getdep('Test0', 'e0')

    @rt.switch_runtime(fixtures.TEST_SITE_CONFIG, 'sys0')
    def test_build_deps_unknown_test(self):
        find_check = TestDependencies.find_check
        checks = self.loader.load_all()

        # Add some inexistent dependencies
        test0 = find_check('Test0', checks)
        for depkind in ('default', 'fully', 'by_env', 'exact'):
            test1 = find_check('Test1_' + depkind, checks)
            if depkind == 'default':
                test1.depends_on('TestX')
            elif depkind == 'exact':
                test1.depends_on('TestX', rfm.DEPEND_EXACT, {'e0': ['e0']})
            elif depkind == 'fully':
                test1.depends_on('TestX', rfm.DEPEND_FULLY)
            elif depkind == 'by_env':
                test1.depends_on('TestX', rfm.DEPEND_BY_ENV)

            with pytest.raises(DependencyError):
                dependency.build_deps(executors.generate_testcases(checks))

    @rt.switch_runtime(fixtures.TEST_SITE_CONFIG, 'sys0')
    def test_build_deps_unknown_target_env(self):
        find_check = TestDependencies.find_check
        checks = self.loader.load_all()

        # Add some inexistent dependencies
        test0 = find_check('Test0', checks)
        test1 = find_check('Test1_default', checks)
        test1.depends_on('Test0', rfm.DEPEND_EXACT, {'e0': ['eX']})
        with pytest.raises(DependencyError):
            dependency.build_deps(executors.generate_testcases(checks))

    @rt.switch_runtime(fixtures.TEST_SITE_CONFIG, 'sys0')
    def test_build_deps_unknown_source_env(self):
        find_check = TestDependencies.find_check
        num_deps = TestDependencies.num_deps
        checks = self.loader.load_all()

        # Add some inexistent dependencies
        test0 = find_check('Test0', checks)
        test1 = find_check('Test1_default', checks)
        test1.depends_on('Test0', rfm.DEPEND_EXACT, {'eX': ['e0']})

        # Unknown source is ignored, because it might simply be that the test
        # is not executed for eX
        deps = dependency.build_deps(executors.generate_testcases(checks))
        assert num_deps(deps, 'Test1_default') == 4

    def create_test(self, name):
        test = rfm.RegressionTest()
        test.name = name
        test.valid_systems = ['*']
        test.valid_prog_environs = ['*']
        test.executable = 'echo'
        test.executable_opts = [name]
        return test

    @rt.switch_runtime(fixtures.TEST_SITE_CONFIG, 'sys0')
    def test_valid_deps(self):
        #
        #       t0
        #       ^
        #       |
        #   +-->t1<--+
        #   |        |
        #   t2<------t3
        #   ^        ^
        #   |        |
        #   +---t4---+
        #
        t0 = self.create_test('t0')
        t1 = self.create_test('t1')
        t2 = self.create_test('t2')
        t3 = self.create_test('t3')
        t4 = self.create_test('t4')
        t1.depends_on('t0')
        t2.depends_on('t1')
        t3.depends_on('t1')
        t3.depends_on('t2')
        t4.depends_on('t2')
        t4.depends_on('t3')
        dependency.validate_deps(
            dependency.build_deps(
                executors.generate_testcases([t0, t1, t2, t3, t4])
            )
        )

    @rt.switch_runtime(fixtures.TEST_SITE_CONFIG, 'sys0')
    def test_cyclic_deps(self):
        #
        #       t0
        #       ^
        #       |
        #   +-->t1<--+
        #   |   |    |
        #   t2  |    t3
        #   ^   |    ^
        #   |   v    |
        #   +---t4---+
        #
        t0 = self.create_test('t0')
        t1 = self.create_test('t1')
        t2 = self.create_test('t2')
        t3 = self.create_test('t3')
        t4 = self.create_test('t4')
        t1.depends_on('t0')
        t1.depends_on('t4')
        t2.depends_on('t1')
        t3.depends_on('t1')
        t3.depends_on('t2')
        t4.depends_on('t2')
        t4.depends_on('t3')
        deps = dependency.build_deps(
            executors.generate_testcases([t0, t1, t2, t3, t4])
        )

        with pytest.raises(DependencyError) as exc_info:
            dependency.validate_deps(deps)

        assert ('t4->t2->t1->t4' in str(exc_info.value) or
                't2->t1->t4->t2' in str(exc_info.value) or
                't1->t4->t2->t1' in str(exc_info.value) or
                't1->t4->t3->t1' in str(exc_info.value) or
                't4->t3->t1->t4' in str(exc_info.value) or
                't3->t1->t4->t3' in str(exc_info.value))

    @rt.switch_runtime(fixtures.TEST_SITE_CONFIG, 'sys0')
    def test_cyclic_deps_by_env(self):
        t0 = self.create_test('t0')
        t1 = self.create_test('t1')
        t1.depends_on('t0', rfm.DEPEND_EXACT, {'e0': ['e0']})
        t0.depends_on('t1', rfm.DEPEND_EXACT, {'e1': ['e1']})
        deps = dependency.build_deps(
            executors.generate_testcases([t0, t1])
        )
        with pytest.raises(DependencyError) as exc_info:
            dependency.validate_deps(deps)

        assert ('t1->t0->t1' in str(exc_info.value) or
                't0->t1->t0' in str(exc_info.value))
