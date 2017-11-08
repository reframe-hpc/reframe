import os
import re
import unittest

from datetime import datetime
from tempfile import NamedTemporaryFile

from reframe.core.environments import Environment
from reframe.core.launchers import *
from reframe.core.modules import module_path_add
from reframe.core.schedulers import *
from reframe.core.shell import BashScriptBuilder
from reframe.frontend.loader import autodetect_system, SiteConfiguration
from reframe.settings import settings

from unittests.fixtures import (
    force_remove_file, system_with_scheduler, TEST_MODULES, TEST_RESOURCES
)


class _TestJob(unittest.TestCase):
    def setUp(self):
        module_path_add([TEST_MODULES])
        self.site_config = SiteConfiguration()
        self.site_config.load_from_dict(settings.site_configuration)

        self.stdout_f = NamedTemporaryFile(
            dir='.', suffix='.out', delete=False)
        self.stderr_f = NamedTemporaryFile(
            dir='.', suffix='.err', delete=False)
        self.script_f = NamedTemporaryFile(dir='.', suffix='.sh', delete=False)

        # Close all files and let whoever interested to open them. Otherwise a
        # local job may fail with a 'Text file busy' error
        self.stdout_f.close()
        self.stderr_f.close()
        self.script_f.close()

    def tearDown(self):
        force_remove_file(self.stdout_f.name)
        force_remove_file(self.stderr_f.name)
        force_remove_file(self.script_f.name)

    def assertProcessDied(self, pid):
        try:
            os.kill(pid, 0)
            self.fail('process %s is still alive' % pid)
        except (ProcessLookupError, PermissionError):
            pass


class TestSlurmJob(_TestJob):
    def setUp(self):
        super().setUp()

        self.num_tasks = 4
        self.num_tasks_per_node = 2
        self.testjob = SlurmJob(
            job_name='testjob',
            job_environ_list=[
                Environment(name='foo', modules=['testmod_foo'])
            ],
            job_script_builder=BashScriptBuilder(login=True),
            script_filename=self.script_f.name,
            num_tasks=self.num_tasks,
            num_tasks_per_node=self.num_tasks_per_node,
            stdout=self.stdout_f.name,
            stderr=self.stderr_f.name,
            launcher_type=NativeSlurmLauncher
        )
        self.testjob.pre_run  = ['echo prerun', 'echo prerun']
        self.testjob.post_run = ['echo postrun']

    def setup_job(self, scheduler):
        partition = system_with_scheduler(scheduler)
        self.testjob.options += partition.access

    def _test_submission(self, ignore_lines=None):
        self.testjob.submit('hostname')
        self.testjob.wait()
        self.assertEqual(self.testjob.state, SLURM_JOB_COMPLETED)
        self.assertEqual(self.testjob.exitcode, 0)

        # Check if job has run on the correct number of nodes
        nodes = set()
        num_tasks   = 0
        num_prerun  = 0
        num_postrun = 0
        before_run = True
        with open(self.testjob.stdout) as f:
            for line in f:
                if ignore_lines and re.search(ignore_lines, line):
                    continue

                if before_run and re.search('^prerun', line):
                    num_prerun += 1
                elif not before_run and re.search('^postrun', line):
                    num_postrun += 1
                else:
                    # The rest of the lines must be from the job
                    nodes.add(line)
                    num_tasks += 1
                    before_run = False

        self.assertEqual(2, num_prerun)
        self.assertEqual(1, num_postrun)
        self.assertEqual(num_tasks, self.num_tasks)
        self.assertEqual(len(nodes), self.num_tasks / self.num_tasks_per_node)

    def _test_state_poll(self):
        t_sleep = datetime.now()
        self.testjob.submit('sleep 3')
        self.testjob.wait()
        t_sleep = datetime.now() - t_sleep

        self.assertEqual(self.testjob.state, SLURM_JOB_COMPLETED)
        self.assertEqual(self.testjob.exitcode, 0)
        self.assertGreaterEqual(t_sleep.total_seconds(), 3)

    @unittest.skipIf(not system_with_scheduler(None),
                     'job submission not supported')
    def test_cancel(self):
        self.setup_job(None)
        self.testjob.submit('sleep 5')
        self.testjob.cancel()

        # Cancel waits for job to finish
        self.assertTrue(self.testjob.finished())
        self.assertEqual(self.testjob.state, SLURM_JOB_CANCELLED)

    def test_cancel_before_submit(self):
        self.testjob.cancel()

    @unittest.skipIf(not system_with_scheduler('nativeslurm'),
                     'native SLURM not supported')
    def test_submit_slurm(self):
        self.setup_job('nativeslurm')
        self._test_submission()

    @unittest.skipIf(not system_with_scheduler('nativeslurm'),
                     'native SLURM not supported')
    def test_state_poll_slurm(self):
        self.setup_job('nativeslurm')
        self._test_state_poll()

    @unittest.skipIf(not system_with_scheduler('slurm+alps'),
                     'SLURM+ALPS not supported')
    def test_submit_alps(self):
        from reframe.launchers import AlpsLauncher

        self.setup_job('slurm+alps')
        self.testjob.launcher = AlpsLauncher(self.testjob)
        self._test_submission(ignore_lines='^Application (\d+) resources\:')

    @unittest.skipIf(not system_with_scheduler('slurm+alps'),
                     'SLURM+ALPS not supported')
    def test_state_poll_alps(self):
        from reframe.launchers import AlpsLauncher

        self.setup_job('slurm+alps')
        self.testjob.launcher = AlpsLauncher(self.testjob)
        self._test_state_poll()


class TestLocalJob(_TestJob):
    def setUp(self):
        super().setUp()
        self.testjob = LocalJob(job_name='localjob',
                                job_environ_list=[],
                                job_script_builder=BashScriptBuilder(),
                                stdout=self.stdout_f.name,
                                stderr=self.stderr_f.name,
                                script_filename=self.script_f.name)

    def test_submission(self):
        self.testjob.submit('sleep 1 && echo hello')
        t_wait = datetime.now()
        self.testjob.wait()
        t_wait = datetime.now() - t_wait

        self.assertGreaterEqual(t_wait.total_seconds(), 1)
        self.assertEqual(self.testjob.state, LOCAL_JOB_SUCCESS)
        self.assertEqual(self.testjob.exitcode, 0)
        with open(self.testjob.stdout) as f:
            self.assertEqual(f.read(), 'hello\n')

        # Double wait; job state must not change
        self.testjob.wait()
        self.assertEqual(self.testjob.state, LOCAL_JOB_SUCCESS)

    def test_submission_timelimit(self):
        self.testjob._time_limit = (0, 0, 2)

        t_job = datetime.now()
        self.testjob.submit('echo before && sleep 10 && echo after')
        self.testjob.wait()
        t_job = datetime.now() - t_job

        self.assertEqual(self.testjob.state, LOCAL_JOB_TIMEOUT)
        self.assertNotEqual(self.testjob.exitcode, 0)
        with open(self.testjob.stdout) as f:
            self.assertEqual(f.read(), 'before\n')

        self.assertGreaterEqual(t_job.total_seconds(), 2)
        self.assertLess(t_job.total_seconds(), 10)

        # Double wait; job state must not change
        self.testjob.wait()
        self.assertEqual(self.testjob.state, LOCAL_JOB_TIMEOUT)

    def test_cancel(self):
        t_job = datetime.now()
        self.testjob.submit('sleep 5')
        self.testjob.cancel()
        t_job = datetime.now() - t_job

        # Cancel waits for the job to finish
        self.assertTrue(self.testjob.finished())
        self.assertLess(t_job.total_seconds(), 5)
        self.assertEqual(self.testjob.state, LOCAL_JOB_FAILURE)

    def test_cancel_before_submit(self):
        self.testjob.cancel()

    def test_cancel_with_grace(self):
        # This test emulates a spawned process that ignores the SIGTERM signal
        # and also spawns another process:
        #
        #   reframe --- local job script --- sleep 10
        #                  (TERM IGN)
        #
        # We expect the job not to be cancelled immediately, since it ignores
        # the gracious signal we are sending it. However, we expect it to be
        # killed immediately after the grace period of 2 seconds expires.
        #
        # We also check that the additional spawned process is also killed.

        self.testjob._time_limit = (0, 1, 0)
        self.testjob.cancel_grace_period = 2
        self.testjob.pre_run = ['trap -- "" TERM']
        self.testjob.post_run = ['echo $!', 'wait']
        self.testjob.submit('sleep 5 &')

        # Stall a bit here to let the the spawned process start and install its
        # signal handler for SIGTERM
        time.sleep(1)

        t_grace = datetime.now()
        self.testjob.cancel()
        t_grace = datetime.now() - t_grace

        self.testjob.wait()
        # Read pid of spawned sleep
        with open(self.testjob.stdout) as f:
            sleep_pid = int(f.read())

        self.assertGreaterEqual(t_grace.total_seconds(), 2)
        self.assertLess(t_grace.total_seconds(), 5)
        self.assertEqual(LOCAL_JOB_TIMEOUT, self.testjob.state)

        # Verify that the spawned sleep is killed, too
        self.assertProcessDied(sleep_pid)

    def test_cancel_term_ignore(self):
        # This test emulates a descendant process of the spawned job that
        # ignores the SIGTERM signal:
        #
        #   reframe --- local job script --- sleep_deeply.sh --- sleep
        #                                      (TERM IGN)
        #
        #  Since the "local job script" does not ignore SIGTERM, it will be
        #  terminated immediately after we cancel the job. However, the deeply
        #  spawned sleep will ignore it. We need to make sure that our
        #  implementation grants the sleep process a grace period and then
        #  kills it.

        prog = os.path.join(TEST_RESOURCES, 'src', 'sleep_deeply.sh')
        self.testjob.cancel_grace_period = 2
        self.testjob.submit(prog)

        # Stall a bit here to let the the spawned process start and install its
        # signal handler for SIGTERM
        time.sleep(1)

        t_grace = datetime.now()
        self.testjob.cancel()
        t_grace = datetime.now() - t_grace
        self.testjob.wait()

        # Read pid of spawned sleep
        with open(self.testjob.stdout) as f:
            sleep_pid = int(f.read())

        self.assertGreaterEqual(t_grace.total_seconds(), 2)
        self.assertEqual(LOCAL_JOB_TIMEOUT, self.testjob.state)

        # Verify that the spawned sleep is killed, too
        self.assertProcessDied(sleep_pid)
