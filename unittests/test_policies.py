import shutil
import tempfile
import unittest

import reframe.frontend.executors as executors
import reframe.frontend.executors.policies as policies
from reframe.core.modules import init_modules_system
from reframe.frontend.loader import RegressionCheckLoader, SiteConfiguration
from reframe.frontend.resources import ResourcesManager
from reframe.settings import settings
from unittests.resources.frontend_checks import (KeyboardInterruptCheck,
                                                 SleepCheck,
                                                 SystemExitCheck)


class TestSerialExecutionPolicy(unittest.TestCase):
    def setUp(self):
        # Load a system configuration
        self.site_config = SiteConfiguration()
        self.site_config.load_from_dict(settings.site_configuration)
        self.system = self.site_config.systems['generic']
        self.resourcesdir = tempfile.mkdtemp(dir='unittests')
        self.resources = ResourcesManager(prefix=self.resourcesdir)
        self.loader = RegressionCheckLoader(['unittests/resources'])

        # Init modules system
        init_modules_system(self.system.modules_system)

        # Setup the runner
        self.runner = executors.Runner(policies.SerialExecutionPolicy())
        self.checks = self.loader.load_all(system=self.system,
                                           resources=self.resources)

    def tearDown(self):
        shutil.rmtree(self.resourcesdir, ignore_errors=True)

    def test_runall(self):
        self.runner.runall(self.checks, self.system)

        stats = self.runner.stats
        self.assertEqual(8, stats.num_cases())
        self.assertEqual(5, stats.num_failures())
        self.assertEqual(3, stats.num_failures_stage('setup'))
        self.assertEqual(1, stats.num_failures_stage('sanity'))
        self.assertEqual(1, stats.num_failures_stage('performance'))

    def test_runall_skip_system_check(self):
        self.runner.policy.skip_system_check = True
        self.runner.runall(self.checks, self.system)

        stats = self.runner.stats
        self.assertEqual(9, stats.num_cases())
        self.assertEqual(5, stats.num_failures())
        self.assertEqual(3, stats.num_failures_stage('setup'))
        self.assertEqual(1, stats.num_failures_stage('sanity'))
        self.assertEqual(1, stats.num_failures_stage('performance'))

    def test_runall_skip_prgenv_check(self):
        self.runner.policy.skip_environ_check = True
        self.runner.runall(self.checks, self.system)

        stats = self.runner.stats
        self.assertEqual(9, stats.num_cases())
        self.assertEqual(5, stats.num_failures())
        self.assertEqual(3, stats.num_failures_stage('setup'))
        self.assertEqual(1, stats.num_failures_stage('sanity'))
        self.assertEqual(1, stats.num_failures_stage('performance'))

    def test_runall_skip_sanity_check(self):
        self.runner.policy.skip_sanity_check = True
        self.runner.runall(self.checks, self.system)

        stats = self.runner.stats
        self.assertEqual(8, stats.num_cases())
        self.assertEqual(4, stats.num_failures())
        self.assertEqual(3, stats.num_failures_stage('setup'))
        self.assertEqual(0, stats.num_failures_stage('sanity'))
        self.assertEqual(1, stats.num_failures_stage('performance'))

    def test_runall_skip_performance_check(self):
        self.runner.policy.skip_performance_check = True
        self.runner.runall(self.checks, self.system)

        stats = self.runner.stats
        self.assertEqual(8, stats.num_cases())
        self.assertEqual(4, stats.num_failures())
        self.assertEqual(3, stats.num_failures_stage('setup'))
        self.assertEqual(1, stats.num_failures_stage('sanity'))
        self.assertEqual(0, stats.num_failures_stage('performance'))

    def test_strict_performance_check(self):
        self.runner.policy.strict_check = True
        self.runner.runall(self.checks, self.system)

        stats = self.runner.stats
        self.assertEqual(8, stats.num_cases())
        self.assertEqual(6, stats.num_failures())
        self.assertEqual(3, stats.num_failures_stage('setup'))
        self.assertEqual(1, stats.num_failures_stage('sanity'))
        self.assertEqual(2, stats.num_failures_stage('performance'))

    def test_kbd_interrupt_within_test(self):
        check = KeyboardInterruptCheck(system=self.system,
                                       resources=self.resources)
        self.assertRaises(KeyboardInterrupt, self.runner.runall,
                          [check], self.system)
        stats = self.runner.stats
        self.assertEqual(1, stats.num_failures())

    def test_system_exit_within_test(self):
        check = SystemExitCheck(system=self.system, resources=self.resources)

        # This should not raise and should not exit
        self.runner.runall([check], self.system)
        stats = self.runner.stats
        self.assertEqual(1, stats.num_failures())


class TaskEventMonitor(executors.TaskEventListener):
    """Event listener for monitoring the execution of the asynchronous execution
    policy.

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
        for p in self.system.partitions:
            p._max_jobs = value

    def read_timestamps(self, tasks):
        """Read the timestamps and sort them to permit simple
        concurrency tests."""
        from reframe.core.deferrable import evaluate

        self.begin_stamps = []
        self.end_stamps = []
        for t in tasks:
            with open(evaluate(t.check.stdout), 'r') as f:
                self.begin_stamps.append(float(f.readline().strip()))
                self.end_stamps.append(float(f.readline().strip()))

        self.begin_stamps.sort()
        self.end_stamps.sort()

    def test_concurrency_unlimited(self):
        checks = [
            SleepCheck(0.5, system=self.system, resources=self.resources),
            SleepCheck(0.5, system=self.system, resources=self.resources),
            SleepCheck(0.5, system=self.system, resources=self.resources)
        ]
        self.set_max_jobs(len(checks))
        self.runner.runall(checks, self.system)

        # Ensure that all tests were run and without failures.
        self.assertEqual(len(checks), self.runner.stats.num_cases())
        self.assertEqual(0, self.runner.stats.num_failures())

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
        checks = [
            SleepCheck(0.5, system=self.system, resources=self.resources),
            SleepCheck(0.5, system=self.system, resources=self.resources),
            SleepCheck(0.5, system=self.system, resources=self.resources),
            SleepCheck(0.5, system=self.system, resources=self.resources),
            SleepCheck(0.5, system=self.system, resources=self.resources)
        ]
        max_jobs = len(checks) - 2
        self.set_max_jobs(max_jobs)
        self.runner.runall(checks, self.system)

        # Ensure that all tests were run and without failures.
        self.assertEqual(len(checks), self.runner.stats.num_cases())
        self.assertEqual(0, self.runner.stats.num_failures())

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
        checks = [
            SleepCheck(0.5, system=self.system, resources=self.resources),
            SleepCheck(0.5, system=self.system, resources=self.resources),
            SleepCheck(0.5, system=self.system, resources=self.resources)
        ]

        num_checks = len(checks)
        self.set_max_jobs(1)
        self.runner.runall(checks, self.system)

        # Ensure that all tests were run and without failures.
        self.assertEqual(len(checks), self.runner.stats.num_cases())
        self.assertEqual(0, self.runner.stats.num_failures())

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
        self.assertRaises(KeyboardInterrupt, self.runner.runall,
                          checks, self.system)

        self.assertEqual(4, self.runner.stats.num_cases())
        self.assertEqual(4, self.runner.stats.num_failures())

    def test_kbd_interrupt_in_wait_with_concurrency(self):
        checks = [
            KeyboardInterruptCheck(system=self.system,
                                   resources=self.resources),
            SleepCheck(1, system=self.system, resources=self.resources),
            SleepCheck(1, system=self.system, resources=self.resources),
            SleepCheck(1, system=self.system, resources=self.resources)
        ]
        self._run_checks(checks, 4)

    def test_kbd_interrupt_in_wait_with_limited_concurrency(self):
        checks = [
            KeyboardInterruptCheck(system=self.system,
                                   resources=self.resources),
            SleepCheck(1, system=self.system, resources=self.resources),
            SleepCheck(1, system=self.system, resources=self.resources),
            SleepCheck(1, system=self.system, resources=self.resources)
        ]
        self._run_checks(checks, 2)

    def test_kbd_interrupt_in_setup_with_concurrency(self):
        checks = [
            SleepCheck(1, system=self.system, resources=self.resources),
            SleepCheck(1, system=self.system, resources=self.resources),
            SleepCheck(1, system=self.system, resources=self.resources),
            KeyboardInterruptCheck(phase='setup',
                                   system=self.system,
                                   resources=self.resources),
        ]
        self._run_checks(checks, 4)

    def test_kbd_interrupt_in_setup_with_limited_concurrency(self):
        checks = [
            SleepCheck(1, system=self.system, resources=self.resources),
            SleepCheck(1, system=self.system, resources=self.resources),
            SleepCheck(1, system=self.system, resources=self.resources),
            KeyboardInterruptCheck(phase='setup',
                                   system=self.system,
                                   resources=self.resources),
        ]
        self._run_checks(checks, 2)
