import shutil
import tempfile
import unittest

from datetime import datetime
from reframe.frontend.executors import *
from reframe.frontend.executors.policies import *
from reframe.frontend.loader import *
from reframe.frontend.resources import ResourcesManager
from reframe.settings import settings

from unittests.fixtures import TEST_SITE_CONFIG

class TestSerialExecutionPolicy(unittest.TestCase):
    def setUp(self):
        # Load a system configuration
        self.site_config = SiteConfiguration()
        self.site_config.load_from_dict(settings.site_configuration)
        self.system = self.site_config.systems['generic']
        self.resourcesdir = tempfile.mkdtemp(dir='unittests')
        self.resources    = ResourcesManager(prefix=self.resourcesdir)
        self.loader       = RegressionCheckLoader(['unittests/resources'])

        # Setup the runner
        self.runner = Runner(SerialExecutionPolicy())
        self.checks = self.loader.load_all(system=self.system,
                                           resources=self.resources)

    def tearDown(self):
        shutil.rmtree(self.resourcesdir, ignore_errors=True)


    def test_runall(self):
        self.runner.runall(self.checks, self.system)

        stats = self.runner.stats
        self.assertEqual(7, stats.num_cases())
        self.assertEqual(5, stats.num_failures())
        self.assertEqual(2, stats.num_failures_stage('setup'))
        self.assertEqual(1, stats.num_failures_stage('sanity'))
        self.assertEqual(2, stats.num_failures_stage('performance'))


    def test_runall_skip_system_check(self):
        self.runner.policy.skip_system_check = True
        self.runner.runall(self.checks, self.system)

        stats = self.runner.stats
        self.assertEqual(8, stats.num_cases())
        self.assertEqual(5, stats.num_failures())
        self.assertEqual(2, stats.num_failures_stage('setup'))
        self.assertEqual(1, stats.num_failures_stage('sanity'))
        self.assertEqual(2, stats.num_failures_stage('performance'))


    def test_runall_skip_prgenv_check(self):
        self.runner.policy.skip_environ_check = True
        self.runner.runall(self.checks, self.system)

        stats = self.runner.stats
        self.assertEqual(8, stats.num_cases())
        self.assertEqual(5, stats.num_failures())
        self.assertEqual(2, stats.num_failures_stage('setup'))
        self.assertEqual(1, stats.num_failures_stage('sanity'))
        self.assertEqual(2, stats.num_failures_stage('performance'))


    def test_runall_skip_sanity_check(self):
        self.runner.policy.skip_sanity_check = True
        self.runner.runall(self.checks, self.system)

        stats = self.runner.stats
        self.assertEqual(7, stats.num_cases())
        self.assertEqual(4, stats.num_failures())
        self.assertEqual(2, stats.num_failures_stage('setup'))
        self.assertEqual(0, stats.num_failures_stage('sanity'))
        self.assertEqual(2, stats.num_failures_stage('performance'))


    def test_runall_skip_performance_check(self):
        self.runner.policy.skip_performance_check = True
        self.runner.runall(self.checks, self.system)

        stats = self.runner.stats
        self.assertEqual(7, stats.num_cases())
        self.assertEqual(3, stats.num_failures())
        self.assertEqual(2, stats.num_failures_stage('setup'))
        self.assertEqual(1, stats.num_failures_stage('sanity'))
        self.assertEqual(0, stats.num_failures_stage('performance'))


    def test_run_relaxed_performance_check(self):
        self.runner.policy.relax_performance_check = True
        self.runner.runall(self.checks, self.system)

        stats = self.runner.stats
        self.assertEqual(7, stats.num_cases())
        self.assertEqual(3, stats.num_failures())
        self.assertEqual(2, stats.num_failures_stage('setup'))
        self.assertEqual(1, stats.num_failures_stage('sanity'))
        self.assertEqual(0, stats.num_failures_stage('performance'))


    def test_kbd_interrupt_within_test(self):
        from unittests.resources.frontend_checks import KeyboardInterruptCheck

        check = KeyboardInterruptCheck(system=self.system,
                                       resources=self.resources)
        self.assertRaises(KeyboardInterrupt, self.runner.runall,
                          [ check ], self.system)
        stats = self.runner.stats
        self.assertEqual(1, stats.num_failures())


    def test_system_exit_within_test(self):
        from unittests.resources.frontend_checks import SystemExitCheck

        check = SystemExitCheck(system=self.system, resources=self.resources)

        # This should not raise and should not exit
        self.runner.runall([ check ], self.system)
        stats = self.runner.stats
        self.assertEqual(1, stats.num_failures())


class TestAsynchronousExecutionPolicy(TestSerialExecutionPolicy):
    def setUp(self):
        super().setUp()
        self.debug_policy = DebugAsynchronousExecutionPolicy()
        self.runner       = Runner(self.debug_policy)


    def set_max_jobs(self, value):
        for p in self.system.partitions:
            p.max_jobs = value


    def read_timestamps_sorted(self):
        self.begin_stamps = []
        self.end_stamps   = []
        for c in self.debug_policy.checks:
            with open(c.stdout, 'r') as f:
                self.begin_stamps.append(float(f.readline().strip()))
                self.end_stamps.append(float(f.readline().strip()))

        self.begin_stamps.sort()
        self.end_stamps.sort()


    def test_concurrency_unlimited(self):
        from unittests.resources.frontend_checks import SleepCheck

        checks = [
            SleepCheck(0.5, system=self.system, resources=self.resources),
            SleepCheck(0.5, system=self.system, resources=self.resources),
            SleepCheck(0.5, system=self.system, resources=self.resources)
        ]
        num_checks = len(checks)
        self.set_max_jobs(num_checks)
        self.runner.runall(checks, self.system)

        # Assure that all tests were run and without failures
        self.assertEqual(num_checks, self.runner.stats.num_cases())
        self.assertEqual(0, self.runner.stats.num_failures())

        # Read the timestamps sorted to permit simple concurrency tests
        self.read_timestamps_sorted()

        # Assure that all tests were run in parallel
        self.assertTrue(self.begin_stamps[-1] < self.end_stamps[0])


    def test_concurrency_limited(self):
        from unittests.resources.frontend_checks import SleepCheck

        # The number of checks must be <= 2*max_jobs
        t = 0.5
        checks = [ SleepCheck(t, system=self.system, resources=self.resources),
                   SleepCheck(t, system=self.system, resources=self.resources),
                   SleepCheck(t, system=self.system, resources=self.resources),
                   SleepCheck(t, system=self.system, resources=self.resources),
                   SleepCheck(t, system=self.system, resources=self.resources) ]
        num_checks = len(checks)
        max_jobs  = num_checks - 2
        self.set_max_jobs(max_jobs)
        self.runner.runall(checks, self.system)

        # Assure that all tests were run and without failures
        self.assertEqual(num_checks, self.runner.stats.num_cases())
        self.assertEqual(0, self.runner.stats.num_failures())

        # Read the timestamps sorted to permit simple concurrency tests
        self.read_timestamps_sorted()

        # Assure that the first #max_jobs jobs were run in parallel
        self.assertTrue(self.begin_stamps[max_jobs-1] < self.end_stamps[0])

        # Assure that the remaining jobs were each run after one of the
        # previous #max_jobs jobs had finished (e.g. begin[max_jobs] > end[0])
        begin_after_end = [b > e for b, e in zip(self.begin_stamps[max_jobs:],
                                                 self.end_stamps[:-max_jobs])]
        self.assertTrue(all(begin_after_end))

        # NOTE: to assure that these remaining jobs were also run
        # in parallel one could do the command hereafter; however, it would
        # require to substantially increase the sleep time (in SleepCheck),
        # because of the delays in rescheduling (1s, 2s, 3s, 1s, 2s,...).
        # We currently prefer not to do this last concurrency test to avoid an
        # important prolongation of the unit test execution time.
        # self.assertTrue(self.begin_stamps[-1] < self.end_stamps[max_jobs])


    def test_concurrency_none(self):
        from unittests.resources.frontend_checks import SleepCheck

        t = 0.5
        checks = [ SleepCheck(t, system=self.system, resources=self.resources),
                   SleepCheck(t, system=self.system, resources=self.resources),
                   SleepCheck(t, system=self.system, resources=self.resources) ]
        num_checks = len(checks)
        self.set_max_jobs(1)
        self.runner.runall(checks, self.system)

        # Assure that all tests were run and without failures
        self.assertEqual(num_checks, self.runner.stats.num_cases())
        self.assertEqual(0, self.runner.stats.num_failures())

        # Read the timestamps sorted to permit simple concurrency tests
        self.read_timestamps_sorted()

        # Assure that the jobs were run after the previous job had finished
        # (e.g. begin[1] > end[0])
        begin_after_end = [ b > e for b, e in zip(self.begin_stamps[1:],
                                                  self.end_stamps[:-1]) ]
        self.assertTrue(all(begin_after_end))


    def _run_checks(self, checks, max_jobs):
        self.set_max_jobs(max_jobs)
        self.assertRaises(KeyboardInterrupt, self.runner.runall,
                          checks, self.system)

        self.assertEqual(4, self.runner.stats.num_cases())
        self.assertEqual(4, self.runner.stats.num_failures())


    def test_kbd_interrupt_in_wait_with_concurrency(self):
        from unittests.resources.frontend_checks import SleepCheck, \
                                                        KeyboardInterruptCheck

        checks = [
            KeyboardInterruptCheck(system=self.system,
                                   resources=self.resources),
            SleepCheck(1, system=self.system, resources=self.resources),
            SleepCheck(1, system=self.system, resources=self.resources),
            SleepCheck(1, system=self.system, resources=self.resources)
        ]
        self._run_checks(checks, 4)


    def test_kbd_interrupt_in_wait_with_limited_concurrency(self):
        from unittests.resources.frontend_checks import SleepCheck, \
                                                        KeyboardInterruptCheck

        checks = [
            KeyboardInterruptCheck(system=self.system,
                                   resources=self.resources),
            SleepCheck(1, system=self.system, resources=self.resources),
            SleepCheck(1, system=self.system, resources=self.resources),
            SleepCheck(1, system=self.system, resources=self.resources)
        ]
        self._run_checks(checks, 2)


    def test_kbd_interrupt_in_setup_with_concurrency(self):
        from unittests.resources.frontend_checks import SleepCheck, \
                                                        KeyboardInterruptCheck

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
        from unittests.resources.frontend_checks import SleepCheck, \
                                                        KeyboardInterruptCheck

        checks = [
            SleepCheck(1, system=self.system, resources=self.resources),
            SleepCheck(1, system=self.system, resources=self.resources),
            SleepCheck(1, system=self.system, resources=self.resources),
            KeyboardInterruptCheck(phase='setup',
                                   system=self.system,
                                   resources=self.resources),
        ]
        self._run_checks(checks, 2)
