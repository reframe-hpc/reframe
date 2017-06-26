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
        self.runner = Runner(AsynchronousExecutionPolicy())


    def set_max_jobs(self, value):
        for p in self.system.partitions:
            p.max_jobs = value


    def test_concurrency_unlimited(self):
        from unittests.resources.frontend_checks import SleepCheck

        checks = [ SleepCheck(1, system=self.system, resources=self.resources),
                   SleepCheck(1, system=self.system, resources=self.resources),
                   SleepCheck(1, system=self.system, resources=self.resources) ]
        self.set_max_jobs(3)

        t_run = datetime.now()
        self.runner.runall(checks, self.system)
        t_run = datetime.now() - t_run
        self.assertLess(t_run.seconds, 2)

        self.assertEqual(3, self.runner.stats.num_cases())
        self.assertEqual(0, self.runner.stats.num_failures())


    def test_concurrency_limited(self):
        from unittests.resources.frontend_checks import SleepCheck

        checks = [ SleepCheck(1, system=self.system, resources=self.resources),
                   SleepCheck(1, system=self.system, resources=self.resources),
                   SleepCheck(1, system=self.system, resources=self.resources) ]
        self.set_max_jobs(2)

        t_run = datetime.now()
        self.runner.runall(checks, self.system)
        t_run = datetime.now() - t_run
        self.assertGreaterEqual(t_run.seconds, 2)
        self.assertLess(t_run.seconds, 3)

        self.assertEqual(3, self.runner.stats.num_cases())
        self.assertEqual(0, self.runner.stats.num_failures())


    def test_concurrency_none(self):
        from unittests.resources.frontend_checks import SleepCheck

        checks = [ SleepCheck(1, system=self.system, resources=self.resources),
                   SleepCheck(1, system=self.system, resources=self.resources),
                   SleepCheck(1, system=self.system, resources=self.resources) ]
        self.set_max_jobs(1)

        t_run = datetime.now()
        self.runner.runall(checks, self.system)
        t_run = datetime.now() - t_run
        self.assertGreaterEqual(t_run.seconds, 3)

        self.assertEqual(3, self.runner.stats.num_cases())
        self.assertEqual(0, self.runner.stats.num_failures())


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
