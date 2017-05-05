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

from unittests.fixtures import TEST_MODULES, system_with_scheduler


class TestJobSubmission(unittest.TestCase):
    def setUp(self):
        module_path_add([TEST_MODULES])
        self.site_config = SiteConfiguration()
        self.site_config.load_from_dict(settings.site_configuration)

        self.stdout_f = NamedTemporaryFile(dir='.', suffix='.out', delete=False)
        self.stderr_f = NamedTemporaryFile(dir='.', suffix='.err', delete=False)
        self.script_f = NamedTemporaryFile(dir='.', suffix='.sh', delete=False)

        # Close all files and let whoever interested to open them. Otherwise the
        # test_loca_job may fail with a 'Text file busy' error
        self.stdout_f.close()
        self.stderr_f.close()
        self.script_f.close()

        # Setup a slurm job
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
            launcher=NativeSlurmLauncher
        )
        self.testjob.pre_run  = [ 'echo prerun', 'echo prerun' ]
        self.testjob.post_run = [ 'echo postrun' ]


    def setup_job(self, scheduler):
        partition = system_with_scheduler(scheduler)
        self.testjob.options += partition.access


    def tearDown(self):
        if os.path.exists(self.stdout_f.name):
            os.remove(self.stdout_f.name)

        if os.path.exists(self.stderr_f.name):
            os.remove(self.stderr_f.name)

        if os.path.exists(self.script_f.name):
            os.remove(self.script_f.name)


    def _test_job_submission(self, ignore_lines = None):
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


    def _test_jobstate_poll(self):
        t_sleep = datetime.now()
        self.testjob.submit('sleep 3')
        self.testjob.wait()
        t_sleep = datetime.now() - t_sleep

        self.assertEqual(self.testjob.state, SLURM_JOB_COMPLETED)
        self.assertEqual(self.testjob.exitcode, 0)
        self.assertGreaterEqual(t_sleep.seconds, 3)


    @unittest.skipIf(not system_with_scheduler('nativeslurm'),
                     'native SLURM not supported')
    def test_job_submission_slurm(self):
        self.setup_job('nativeslurm')
        self._test_job_submission()


    @unittest.skipIf(not system_with_scheduler('nativeslurm'),
                     'native SLURM not supported')
    def test_jobstate_poll_slurm(self):
        self.setup_job('nativeslurm')
        self._test_jobstate_poll()


    @unittest.skipIf(not system_with_scheduler('slurm+alps'),
                     'SLURM+ALPS not supported')
    def test_job_submission_alps(self):
        from reframe.launchers import AlpsLauncher

        self.setup_job('slurm+alps')
        self.testjob.launcher = AlpsLauncher(self.testjob)
        self._test_job_submission(ignore_lines='^Application (\d+) resources\:')


    @unittest.skipIf(not system_with_scheduler('slurm+alps'),
                     'SLURM+ALPS not supported')
    def test_jobstate_poll_alps(self):
        from reframe.launchers import AlpsLauncher

        self.setup_job('slurm+alps')
        self.testjob.launcher = AlpsLauncher(self.testjob)
        self._test_jobstate_poll()


    def test_local_job(self):

        self.testjob = LocalJob(job_name='localjob',
                                job_environ_list=[],
                                job_script_builder=BashScriptBuilder(),
                                stdout=self.stdout_f.name,
                                stderr=self.stderr_f.name,
                                script_filename=self.script_f.name,
                                time_limit=(0, 0, 3))
        self.testjob.submit('sleep 2 && echo hello')
        self.testjob.wait()
        self.assertEqual(self.testjob.state, LOCAL_JOB_SUCCESS)
        self.assertEqual(self.testjob.exitcode, 0)
        with open(self.testjob.stdout) as f:
            self.assertEqual(f.read(), 'hello\n')

    def test_local_job_timelimit(self):
        self.testjob = LocalJob(job_name='localjob',
                                job_environ_list=[],
                                job_script_builder=BashScriptBuilder(),
                                stdout=self.stdout_f.name,
                                stderr=self.stderr_f.name,
                                script_filename=self.script_f.name,
                                time_limit=(0, 0, 2))
        t_job = datetime.now()
        self.testjob.submit('echo before && sleep 10 && echo after')
        self.testjob.wait()
        t_job = datetime.now() - t_job
        self.assertEqual(self.testjob.state, LOCAL_JOB_TIMEOUT)
        self.assertNotEqual(self.testjob.exitcode, 0)
        with open(self.testjob.stdout) as f:
            self.assertEqual(f.read(), 'before\n')

        self.assertGreaterEqual(t_job.seconds, 2)
        self.assertLess(t_job.seconds, 10)


    def test_launcher_wrapper_native_slurm(self):
        builder = BashScriptBuilder()
        ddt_launcher = LauncherWrapper(NativeSlurmLauncher(None),
                                       'ddt', '-o foo.out'.split())
        ddt_launcher.emit_run_command('hostname', builder)
        script_text = builder.finalise()
        self.assertIsNone(re.search('^\s*srun', script_text, re.MULTILINE))
        self.assertIsNotNone(re.search('^ddt\s+-o\s+foo\.out\s+srun\s+hostname',
                                       script_text, re.MULTILINE))


    def test_launcher_wrapper_alps(self):
        builder = BashScriptBuilder()
        ddt_launcher = LauncherWrapper(AlpsLauncher(None),
                                       'ddt', '-o foo.out'.split())
        ddt_launcher.emit_run_command('hostname', builder)
        script_text = builder.finalise()
        self.assertIsNone(re.search('^\s*aprun', script_text, re.MULTILINE))
        self.assertIsNotNone(
            re.search('^ddt\s+-o\s+foo\.out\s+aprun\s+-B\s+hostname',
                      script_text, re.MULTILINE)
        )
