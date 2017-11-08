import shutil
import tempfile
import unittest
import warnings

from reframe.core.exceptions import ReframeDeprecationWarning
from reframe.frontend.executors import *
from reframe.frontend.executors.policies import *
from reframe.frontend.loader import *
from reframe.frontend.resources import ResourcesManager
from reframe.settings import settings
from unittests.resources.frontend_checks import (KeyboardInterruptCheck,
                                                 SleepCheck,
                                                 SystemExitCheck)


# In order to reliably test the AsynchronousExecutionPolicy we need to
# monitor its internal state. That is why we subclass it, adding some
# variables for monitoring. We also test the observed behaviour,
# but not all of these tests are deterministic. Non-deterministic tests may
# not lead to a test failure, but only to a test skip.

class DebugAsynchronousExecutionPolicy(AsynchronousExecutionPolicy):
    def __init__(self):
        super().__init__()
        self.keep_stage_files = True
        self.checks = []
        # Storage for monitoring of the number of active cases
        # (active case := running cases from the perspective of the policy)
        self.num_active_cases = []

    def exit_environ(self, c, p, e):
        super().exit_environ(c, p, e)
        self.checks.append(c)

    # We overwrite _reschedule, because all jobs are submitted from here and
    # the running_cases_counts are updated at submission.
    def _reschedule(self, ready_testcase, load_env=True):
        super()._reschedule(ready_testcase, load_env)
        self.num_active_cases.append(
            self._running_cases_counts['generic:login'])


class TestSerialExecutionPolicy(unittest.TestCase):
    def setUp(self):
        # Ignore deprecation warnings
        warnings.simplefilter('ignore', ReframeDeprecationWarning)

        # Load a system configuration
        self.site_config = SiteConfiguration()
        self.site_config.load_from_dict(settings.site_configuration)
        self.system = self.site_config.systems['generic']
        self.resourcesdir = tempfile.mkdtemp(dir='unittests')
        self.resources = ResourcesManager(prefix=self.resourcesdir)
        self.loader = RegressionCheckLoader(['unittests/resources'])

        # Setup the runner
        self.runner = Runner(SerialExecutionPolicy())
        self.checks = self.loader.load_all(system=self.system,
                                           resources=self.resources)

    def tearDown(self):
        shutil.rmtree(self.resourcesdir, ignore_errors=True)
        warnings.simplefilter('default', ReframeDeprecationWarning)

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


class TestAsynchronousExecutionPolicy(TestSerialExecutionPolicy):
    def setUp(self):
        super().setUp()
        self.debug_policy = DebugAsynchronousExecutionPolicy()
        self.runner       = Runner(self.debug_policy)

    def set_max_jobs(self, value):
        for p in self.system.partitions:
            p._max_jobs = value

    def read_timestamps_sorted(self):
        from reframe.core.deferrable import evaluate

        self.begin_stamps = []
        self.end_stamps   = []
        for c in self.debug_policy.checks:
            with open(evaluate(c.stdout), 'r') as f:
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
        num_checks = len(checks)
        self.set_max_jobs(num_checks)
        self.runner.runall(checks, self.system)

        # Ensure that all tests were run and without failures.
        self.assertEqual(num_checks, self.runner.stats.num_cases())
        self.assertEqual(0, self.runner.stats.num_failures())

        # Ensure that all tests were simultaneously active.
        self.assertEqual(len(self.debug_policy.num_active_cases), num_checks)
        self.assertEqual(self.debug_policy.num_active_cases[-1], num_checks)

        # Read the timestamps sorted to permit simple concurrency tests.
        self.read_timestamps_sorted()

        # Warn if not all tests were run in parallel; the corresponding strict
        # check would be:
        # self.assertTrue(self.begin_stamps[-1] <= self.end_stamps[0])
        if self.begin_stamps[-1] > self.end_stamps[0]:
            self.skipTest('the system seems too loaded.')

    def test_concurrency_limited(self):
        # The number of checks must be <= 2*max_jobs.
        checks = [
            SleepCheck(0.5, system=self.system, resources=self.resources),
            SleepCheck(0.5, system=self.system, resources=self.resources),
            SleepCheck(0.5, system=self.system, resources=self.resources),
            SleepCheck(0.5, system=self.system, resources=self.resources),
            SleepCheck(0.5, system=self.system, resources=self.resources)
        ]

        num_checks = len(checks)
        max_jobs   = num_checks - 2
        self.set_max_jobs(max_jobs)
        self.runner.runall(checks, self.system)

        # Ensure that all tests were run and without failures.
        self.assertEqual(num_checks, self.runner.stats.num_cases())
        self.assertEqual(0, self.runner.stats.num_failures())

        # Ensure that #max_jobs tests were simultaneously active.
        self.assertEqual(len(self.debug_policy.num_active_cases), num_checks)
        self.assertEqual(self.debug_policy.num_active_cases[max_jobs-1],
                         max_jobs)

        # Read the timestamps sorted to permit simple concurrency tests.
        self.read_timestamps_sorted()

        # Ensure that the jobs after the first #max_jobs were each run after
        # one of the previous #max_jobs jobs had finished
        # (e.g. begin[max_jobs] > end[0]).
        # Note: we may ensure this strictly as we may ensure serial behaviour.
        begin_after_end = [b > e for b, e in zip(self.begin_stamps[max_jobs:],
                                                 self.end_stamps[:-max_jobs])]
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
        self.assertEqual(num_checks, self.runner.stats.num_cases())
        self.assertEqual(0, self.runner.stats.num_failures())

        # Ensure that there was only one active job at a time.
        self.assertEqual(len(self.debug_policy.num_active_cases), num_checks)
        self.assertEqual(max(self.debug_policy.num_active_cases), 1)

        # Read the timestamps sorted to permit simple concurrency tests.
        self.read_timestamps_sorted()

        # Ensure that the jobs were run after the previous job had finished
        # (e.g. begin[1] > end[0]).
        begin_after_end = [b > e for b, e in zip(self.begin_stamps[1:],
                                                 self.end_stamps[:-1])]
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
